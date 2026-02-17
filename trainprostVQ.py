import os, glob, re
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import pandas as pd
import numpy as np
import pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import nibabel as nib
from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandRotated,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandBiasFieldd,

)
from monai.metrics import DiceMetric, compute_hausdorff_distance, compute_average_surface_distance,  compute_meandice
from monai.data import decollate_batch

from tqdm import tqdm
from argparse import ArgumentParser

from convnet3D_utils import  VQUNet3Dposv3

image_size = (224, 224)
num_classes = 14
batch_size = 100
epochs = 300
num_workers = 4

@torch.no_grad()
def boundary_from_onehot(onehot: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    onehot: [B, C, H, W, D] float {0,1}
    return: [B, 1, H, W, D] float in [0,1]
    """

    fg = onehot[:, 1:, ...]  # [B, C-1, H, W, D]

    dil = F.max_pool3d(fg, kernel_size=k, stride=1, padding=k // 2)
    ero = -F.max_pool3d(-fg, kernel_size=k, stride=1, padding=k // 2)

    bnd = (dil - ero).clamp(0.0, 1.0)               # [B, C-1, ...]
    bnd = bnd.max(dim=1, keepdim=True).values       # [B, 1, ...]
    return bnd

def codebook_metrics(tokens: torch.Tensor, n_embed: int, dist_reduce: bool = False):
    flat = tokens.reshape(-1)
    hist = torch.bincount(flat, minlength=n_embed).float()

    if dist_reduce and dist.is_available() and dist.is_initialized():
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)

    p = hist / hist.sum().clamp_min(1.0)
    used = (hist > 0).sum().item()
    p_nz = p[p > 0]
    entropy = -(p_nz * p_nz.log()).sum()
    perplexity = entropy.exp().item()
    usage_frac = used / n_embed
    return usage_frac, perplexity, hist


class ProstateDataset(Dataset):
    def __init__(self, csv_file_img, datatype):
        self.data = pd.read_csv(csv_file_img)
        self.datatype = datatype
        self.train_transforms = Compose(
            [
                LoadImaged(
                    keys = ["image", "label"]
                ),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"], 
                    pixdim=(0.5, 0.5, 1.5), 
                    mode=("nearest", "nearest")
                ),
                CropForegroundd(
                    keys=["image", "label"],
                    source_key="label",
                    margin=10
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                #NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                #RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(192, 192, 64),
                ),

                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(192, 192, 64),
                    pos=1,
                    neg=0,    
                    num_samples=1
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandRotated(
                    keys=["image", "label"], 
                    range_x=0.2, 
                    prob=0.10, 
                    mode=("nearest","nearest")
                ),

                #RandShiftIntensityd(keys="image", offsets=0.1, prob=0.4),
                #RandGaussianNoised(keys = "image", std = 0.05, prob = 0.15),
                #RandAdjustContrastd(keys="image",prob = 0.2),
                #RandBiasFieldd(keys="image",  prob=0.2),

            ]
        )
        self.val_transforms = Compose(
            [   

                LoadImaged(
                    keys = ["image", "label"]
                ),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"], 
                    pixdim=(0.5, 0.5, 1.5), 
                    mode=("nearest", "nearest")
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                #NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                # CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(192, 192, 64),
                ),
                CenterSpatialCropd(keys=["image","label"], roi_size=(192,192,64))
                #ToTensord(keys=["image", "label"]),
            ]
        )

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            #img_path = self.data.loc[idx, 'images']
            #img_label = self.data.loc[idx, 'labels']

            #sample = {'image': img_path, 'label': img_label}

            img_label = self.data.loc[idx, 'labels']
            sample = {'image': img_label, 'label': img_label}

            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        if isinstance(sample, list):
            sample = sample[0]

        image = sample["image"]
        label = sample["label"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        return {"image": image, "label": label}

    def get_sample(self, item):
        sample = self.samples[item]
        if self.datatype == 'train':
            sample = self.train_transforms(sample)
        else:
            sample = self.val_transforms(sample)
        
        return sample

class ProstateDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.batch_size =batch_size
        self.num_workers = num_workers


        self.train_set = ProstateDataset(self.csv_train_img, datatype='train')
        self.val_set = ProstateDataset(self.csv_val_img, datatype= 'val')
        self.test_set = ProstateDataset(self.csv_test_img, datatype='test')

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

class Net(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 3,
        inputchannels: Optional[int] = None,
        use_boundary_channel: bool = True,
        boundary_k: int = 3,
        channels=16,
        dropout=0.0,
        n_embed=1024,
        embed_dim=256,
        w_d=0.8,
        w_hd=0.1,
        w_asd=0.1,
        max_epochs=50,
        check_val=1,
        output_root=".",
    ):
        super().__init__()

        if inputchannels is None:
            inputchannels = num_classes + (1 if use_boundary_channel else 0) 

        self.save_hyperparameters()
        self.n_embed = n_embed
#        self.automatic_optimization = False
       ### data = pd.read_csv(train_csv)
        #self.trainlen = len(data)
        self._model = VQUNet3Dposv3(
            inputchannels=inputchannels,
            num_classes=num_classes,
            channels=channels,
            dropout=dropout,
            n_embed=n_embed,
            embed_dim=embed_dim,
        )
        # CE weights with correct length (equal to num_classes)
        w = torch.ones(num_classes, dtype=torch.float32)
        w[0] = 0.05  # background más bajo
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=w)

        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        self.dictresults = {'dice':[], 'hd': [], 'asd': []}
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.w_d = w_d
        self.w_hd = w_hd
        self.w_asd = w_asd
        self.best_multi = -1.0
        self.best_multi_epoch = -1
        self.max_epochs = max_epochs
        self.check_val = check_val
        self.warmup_epochs = 10
        self.metric_values = []
        self.epoch_loss_values = []

        self.output_root = output_root
        self.test_output_dir = os.path.join(self.output_root, "testimagesoutput")
        self.test_results_csv = os.path.join(self.output_root, "testresults", "result.csv")

        #self.bn = int(math.ceil(self.trainlen/batch_size))

    def forward(self, input, return_indices: bool = False):
        logits, emb_loss, quant, indices = self._model(input)
        if return_indices:
            return logits, emb_loss, indices
        return logits, emb_loss

    def forward1(self, input, return_indices: bool = False):
        logits, emb_loss, quant, indices = self._model(input)
        if return_indices:
            return logits, indices
        return logits
    
    @torch.no_grad()
    def token_stats_by_class(self, indices: torch.Tensor, labels: torch.Tensor, num_classes: int):
        """
        indices: [B, Ht, Wt, Dt]  (p.ej. [B,12,12,4])
        labels:  [B, 1, H, W, D] o [B, H, W, D]
        """
        if labels.ndim == 5:
            lab = labels.float()              # [B,1,H,W,D] o [B,C,H,W,D]
            if lab.shape[1] != 1:
                lab = lab.argmax(dim=1, keepdim=True).float()
        elif labels.ndim == 4:
            lab = labels.unsqueeze(1).float() # [B,1,H,W,D]
        else:
            raise ValueError(f"labels.ndim inesperado: {labels.ndim}")

        lab_ds = F.interpolate(lab, size=indices.shape[1:], mode="nearest").long().squeeze(1)

        print(f"\n[Epoch {self.current_epoch}] Token stats by class (grid={tuple(indices.shape[1:])})")
        for c in range(num_classes):
            mask = (lab_ds == c)
            ids = indices[mask]

            if ids.numel() == 0:
                print(f"  class {c}: n_tokens=0 (It does not appear in this batch.)")
                continue

            uniq, cnt = torch.unique(ids, return_counts=True)

            k = min(5, cnt.numel())
            top_cnt, top_pos = torch.topk(cnt, k=k)

            print(f"  class {c}: n_tokens={ids.numel()}, unique={uniq.numel()}")
            for j in range(k):
                pos = top_pos[j].item()
                print(f"    id {int(uniq[pos])}  count {int(top_cnt[j])}")

    #def get_input(self, batch, k):
    #    x = batch[k]
    #    x = x.to(memory_format=torch.contiguous_format)
    #    return x.float()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        x = self.build_x(labels)

        logits, emb_loss, indices = self.forward(x, return_indices=True)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.token_stats_by_class(
                indices,
                labels,
                num_classes=self.hparams.num_classes
            )

        usage_frac, ppl, hist = codebook_metrics(indices, self.n_embed, dist_reduce=True)
        self.log("codebook/usage_frac", usage_frac, prog_bar=True, on_step=False, on_epoch=True)
        self.log("codebook/perplexity", ppl, prog_bar=False, on_step=False, on_epoch=True)

        if self.trainer.is_global_zero:
            k = min(10, hist.numel())
            top_ids = torch.topk(hist, k=k).indices
            top_counts = hist[top_ids].to(torch.int64)
            pairs = list(zip(top_ids.tolist(), top_counts.tolist()))
            print(f"[Epoch {self.current_epoch}] codebook usage={usage_frac:.3f} perplexity={ppl:.1f} top={pairs}")

        # Loss SIEMPRE con logits (tensor), no con post_pred
        seg_loss = self.loss_function(logits, labels)
        cap = 1.0
        emb_weight = min(cap, cap * float(self.current_epoch + 1) / float(self.warmup_epochs))
        loss = seg_loss + emb_weight * emb_loss

        # Métricas con discretización
        preds = [self.post_pred(i) for i in decollate_batch(logits)]
        labels1 = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=preds, y=labels1)
        self.dice_metric.reset()

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_seg_loss", seg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_emb_loss", emb_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_emb_w", emb_weight, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)     # logits tensor, OK
            gt = labels.squeeze(1)
            dices = []
            eps = 1e-6
            for c in range(1, self.hparams.num_classes):
                p = (pred == c).float()
                g = (gt == c).float()
                inter = (p * g).sum(dim=(1, 2, 3))
                denom = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3)) + eps
                dices.append((2.0 * inter / denom))
            dice_scalar = torch.stack(dices, dim=1).mean().item()

        self.log("train_dice", dice_scalar, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.current_epoch == 0 and batch_idx == 0:
            u = torch.unique(labels)
            print("[DEBUG] unique labels:", u.detach().cpu().tolist(), "max=", int(labels.max()))

        return loss


    def training_epoch_end(self, outputs):
        if len(outputs) == 0:
            return

        # We ensure that everything is Tensor
        losses = []
        for o in outputs:
            if isinstance(o, dict) and "loss" in o:
                losses.append(o["loss"])
            else:
                losses.append(o)

        losses = [l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in losses]

        avg_loss = torch.stack(losses).mean()
        self.epoch_loss_values.append(float(avg_loss.detach().cpu()))

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        x = self.build_x(labels)

        logits, emb_loss, indices = self.forward(x, return_indices=True)

        # Loss (usa logits, no "outputs")
        seg_loss = self.loss_function(logits, labels)
        cap = 1.0
        emb_weight = min(cap, cap * float(self.current_epoch + 1) / float(self.warmup_epochs))
        loss = seg_loss + emb_weight * emb_loss

        if batch_idx == 0:
            usage, perplexity, hist = codebook_metrics(indices, self.hparams.n_embed)
            self.log("codebook_usage", usage, prog_bar=True, logger=True)
            self.log("codebook_perplexity", perplexity, prog_bar=False, logger=True)

            k = min(10, hist.numel())
            topv, topi = torch.topk(hist, k=k)
            top_list = [(int(i), int(v)) for i, v in zip(topi.cpu(), topv.cpu())]
            print(f"[Epoch {self.current_epoch}] codebook usage={usage:.3f} perplexity={perplexity:.1f} top={top_list}")

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_seg_loss", seg_loss, prog_bar=False, logger=True)
        self.log("val_emb_loss", emb_loss, prog_bar=False, logger=True)
        self.log("val_emb_w", emb_weight, prog_bar=False, logger=True)

        # Métricas: discretiza a partir de logits
        preds = [self.post_pred(i) for i in decollate_batch(logits)]
        labels1 = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=preds, y=labels1)

        hd = compute_hausdorff_distance(
            y_pred=torch.stack(preds, dim=0),
            y=torch.stack(labels1, dim=0),
            include_background=False
        )
        asd = compute_average_surface_distance(
            y_pred=torch.stack(preds, dim=0),
            y=torch.stack(labels1, dim=0),
            include_background=False
        )

        return {"val_loss": loss, "hd": hd, "asd": asd, "val_number": logits.shape[0]}

    def validation_epoch_end(self, outputs):
        val_loss, num_items, hd, asd = 0.0, 0, 0.0, 0.0
        for output in outputs:
            val_loss += float(output["val_loss"].sum().item())
            hd       += float(output["hd"].sum().item())
            asd      += float(output["asd"].sum().item())
            num_items += output["val_number"]

        if num_items == 0:
            return

        mean_val_dice = float(self.dice_metric.aggregate().item())
        self.dice_metric.reset()

        mean_val_loss = val_loss / num_items
        mean_hd = hd / num_items
        mean_asd = asd / num_items

        hd_score = 1.0 / (1.0 + mean_hd)
        asd_score = 1.0 / (1.0 + mean_asd)

        multi_score = (
            self.w_d  * mean_val_dice +
            self.w_hd * hd_score +
            self.w_asd * asd_score
        )

        # Logs for TensorBoard
        self.log("val_loss",  mean_val_loss, prog_bar=True,  logger=True)
        self.log("val_dice",  mean_val_dice, prog_bar=True,  logger=True)
        self.log("val_hd",    mean_hd,       prog_bar=False, logger=True)
        self.log("val_asd",   mean_asd,      prog_bar=False, logger=True)
        self.log("val_multi", multi_score,   prog_bar=True,  logger=True)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        if multi_score > self.best_multi:
            self.best_multi = multi_score
            self.best_multi_epoch = self.current_epoch

        print(
            f"epoch {self.current_epoch}: "
            f"Dice={mean_val_dice:.4f}, "
            f"HD={mean_hd:.4f}, "
            f"ASD={mean_asd:.4f}, "
            f"Multi={multi_score:.4f}\n"
            f"best Dice={self.best_val_dice:.4f} @ epoch {self.best_val_epoch}, "
            f"best Multi={self.best_multi:.4f} @ epoch {self.best_multi_epoch}"
        )
        self.metric_values.append(mean_val_dice)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        x = self.build_x(labels)

        outputs = self.forward1(x) 

        loss = self.loss_function(outputs, labels.long())
        print('test Loss: %.3f' % (loss))

        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels1 = [self.post_label(i) for i in decollate_batch(labels)]

        dice = torch.mean(
            compute_meandice(
                y_pred=torch.stack(outputs, dim=0),
                y=torch.stack(labels1, dim=0),
                include_background=False,
            )
        )
        hd = torch.mean(
            compute_hausdorff_distance(
                y_pred=torch.stack(outputs, dim=0),
                y=torch.stack(labels1, dim=0),
                include_background=False,
            )
        )
        asd = torch.mean(
            compute_average_surface_distance(
                y_pred=torch.stack(outputs, dim=0),
                y=torch.stack(labels1, dim=0),
                include_background=False,
            )
        )

        print(f"dice: {dice:.4f} hd: {hd:.4f} asd: {asd:.4f}")

        # logs
        self.log("test_dice", dice)
        self.log("test_hd", hd)
        self.log("test_asd", asd)

        # store metrics in dict as floats
        self.dictresults['dice'].append(float(dice.detach().cpu()))
        self.dictresults['asd'].append(float(asd.detach().cpu()))
        self.dictresults['hd'].append(float(hd.detach().cpu()))

        # save images
        os.makedirs(self.test_output_dir, exist_ok=True)
        for i in range(len(outputs)):
            output = torch.squeeze(outputs[i])
            output = torch.argmax(output, dim=0)
            output = output.cpu().numpy().astype(np.float32)
            label = torch.squeeze(labels1[i])
            label = torch.argmax(label, dim=0)
            label = label.cpu().numpy().astype(np.float32)
            image = torch.squeeze(images[i]).cpu().numpy().astype(np.float32)

            affine = np.eye(4)
            nib.save(nib.Nifti1Image(output, affine),
                    os.path.join(self.test_output_dir, f"seg{i}_batch{batch_idx}.nii"))
            nib.save(nib.Nifti1Image(label, affine),
                    os.path.join(self.test_output_dir, f"label{i}_batch{batch_idx}.nii"))
            nib.save(nib.Nifti1Image(image, affine),
                    os.path.join(self.test_output_dir, f"image{i}_batch{batch_idx}.nii"))

        return loss

    def test_epoch_end(self, outputs):
        # Create results folder (if it does not exist)
        results_dir = os.path.dirname(self.test_results_csv)
        os.makedirs(results_dir, exist_ok=True)
        df = pd.DataFrame(self.dictresults)
        df.to_csv(self.test_results_csv, index=False)

    def build_x(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        onehot = F.one_hot(
            labels.squeeze(1),
            num_classes=self.hparams.num_classes
        ).permute(0, 4, 1, 2, 3).float()  # [B, C, H, W, D]

        if not self.hparams.use_boundary_channel:
            x = onehot
        else:
            bnd = boundary_from_onehot(onehot, k=self.hparams.boundary_k)  # [B,1,...]
            x = torch.cat([onehot, bnd], dim=1)                            # [B,C+1,...]

        # chequeo defensivo (tu error actual queda prevenido aquí)
        expected = self.hparams.inputchannels
        if x.shape[1] != expected:
            raise RuntimeError(
                f"Channel mismatch: x has {x.shape[1]} channels but model expects "
                f"{expected}. (num_classes={self.hparams.num_classes}, "
                f"use_boundary_channel={self.hparams.use_boundary_channel})"
            )
        return x

def _extract_epoch_from_name(path: str) -> int:
    name = os.path.basename(path)

    m = re.search(r"epoch=(\d+)", name)
    if m:
        return int(m.group(1))

    m = re.search(r"epoch(\d+)", name)
    if m:
        return int(m.group(1))

    return -1

def get_highest_epoch_ckpt(root_dir: str, patterns: Optional[List[str]] = None) -> Optional[str]:
    if patterns is None:
        patterns = ["best_*.ckpt"]

    ckpts = []
    for p in patterns:
        ckpts.extend(glob.glob(os.path.join(root_dir, p)))

    if not ckpts:
        return None

    parsed = [(c, _extract_epoch_from_name(c)) for c in ckpts]
    parsed_ok = [(c, e) for (c, e) in parsed if e >= 0]

    if parsed_ok:
        parsed_ok.sort(key=lambda x: x[1])
        return parsed_ok[-1][0]

    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

def export_token_indices(trainer, model: Net, dataloader, out_path: str):
    model.eval()
    all_tokens = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(model.device)
            x = model.build_x(labels)
            _, _, indices = model.forward(x, return_indices=True)
            all_tokens.append(indices.cpu())

    tokens = torch.cat(all_tokens, dim=0)
    torch.save(tokens, out_path)
    print(f"Saved tokens: {tokens.shape} -> {out_path}")

if __name__ == '__main__':
        pl.seed_everything(42, workers=True)

        NUM_CLASSES = 3

        RESUME_TRAINING = False          # False -> start from scratch, True -> resume
        RESUME_MODE = "highest_epoch"   # "highest_epoch" (recommended) or "none"

        root_dir = "./data/Prostate/outputprostatefinal"

        # Pick checkpoint with highest epoch among best_*.ckpt (best_dice + best_multi)
        resume_ckpt = None
        if RESUME_TRAINING and RESUME_MODE == "highest_epoch":
            resume_ckpt = get_highest_epoch_ckpt(root_dir, patterns=["best_*.ckpt"])
            if resume_ckpt is None:
                print(f"[WARN] RESUME_TRAINING=True but no best_*.ckpt found in {root_dir}. Training from scratch.")
            else:
                print(f"[INFO] Resuming training from highest-epoch ckpt: {resume_ckpt}")

        net = Net(
            num_classes=3,               
            inputchannels=None,           
            use_boundary_channel=True,
            boundary_k=3,
            channels=16,
            dropout=0.0,
            n_embed=32,
            embed_dim=256,
            w_d=0.8,
            w_hd=0.1,
            w_asd=0.1,
            max_epochs=10,
            check_val=1,
            output_root=root_dir,
        )
        data = ProstateDataModule(
            batch_size=1,
            num_workers=4,
            csv_train_img="./train.csv",
            csv_val_img="./validation.csv",
            csv_test_img="./test.csv",
        )

        # set up checkpoints

        checkpoint_dice = ModelCheckpoint(
            dirpath=root_dir,
            filename="best_dice_epoch{epoch:02d}_dice{val_dice:.4f}",
            monitor="val_dice",
            mode="max",
            save_top_k=1,
        )
        checkpoint_multi = ModelCheckpoint(
            dirpath=root_dir,
            filename="best_multi_epoch{epoch:02d}_multi{val_multi:.4f}",
            monitor="val_multi",
            mode="max",
            save_top_k=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_dice",
            mode="max",
            patience=15,
            min_delta=1e-4,
            verbose=True,
        )
        # initialise Lightning's trainer.
        trainer = pytorch_lightning.Trainer(
            gpus=[2],
            max_epochs=net.max_epochs,
            check_val_every_n_epoch=net.check_val,
            #callbacks=[checkpoint_dice, checkpoint_multi, early_stopping],
            callbacks=[checkpoint_dice,checkpoint_multi],
            default_root_dir=root_dir,
        )

        os.makedirs(net.test_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(net.test_results_csv), exist_ok=True)

        try:
            # Newer Lightning
            if resume_ckpt is not None:
                trainer.fit(net, data, ckpt_path=resume_ckpt)
            else:
                trainer.fit(net, data)

        except TypeError:
            # Older Lightning fallback
            if resume_ckpt is not None:
                trainer = pytorch_lightning.Trainer(
                    gpus=[0],
                    max_epochs=net.max_epochs,
                    check_val_every_n_epoch=net.check_val,
                    callbacks=[checkpoint_dice, checkpoint_multi],
                    default_root_dir=root_dir,
                    resume_from_checkpoint=resume_ckpt,
                )
            trainer.fit(net, data)

        # ---- select best checkpoints for testing ----
        dice_ckpts  = glob.glob(os.path.join(root_dir, "best_dice_epoch*.ckpt"))
        multi_ckpts = glob.glob(os.path.join(root_dir, "best_multi_epoch*.ckpt"))

        if not dice_ckpts:
            raise FileNotFoundError(f"No best_dice_epoch*.ckpt file was found in {root_dir}")
        if not multi_ckpts:
            raise FileNotFoundError(f"No best_multi_epoch*.ckpt file was found in {root_dir}")

        best_dice_path  = sorted(dice_ckpts)[-1]
        best_multi_path = sorted(multi_ckpts)[-1]

        print(f"Best DICE checkpoint: {best_dice_path}")
        print(f"Best MULTI checkpoint: {best_multi_path}")

        # ---- test best dice ----
        output_root_dice = os.path.join(root_dir, "test_best_dice")
        model_dice = Net.load_from_checkpoint(
            best_dice_path,
            output_root=output_root_dice,
            strict=False,
        )  
        os.makedirs(model_dice.test_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_dice.test_results_csv), exist_ok=True)

        print(f"=== Test model best_dice: results in {output_root_dice} ===")
        trainer.test(model_dice, data.test_dataloader())

        # ---- test best multi ----
        output_root_multi = os.path.join(root_dir, "test_best_multi")
        model_multi = Net.load_from_checkpoint(
            best_multi_path,
            output_root=output_root_multi,
            strict=False,
        )
        os.makedirs(model_multi.test_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_multi.test_results_csv), exist_ok=True)

        print(f"=== Test model best_multi: results in {output_root_multi} ===")
        trainer.test(model_multi, data.test_dataloader())

        export_token_indices(trainer, net, data.train_dataloader(), os.path.join(root_dir, "tokens_train.pt"))
        export_token_indices(trainer, net, data.val_dataloader(),   os.path.join(root_dir, "tokens_val.pt"))
        export_token_indices(trainer, net, data.test_dataloader(),  os.path.join(root_dir, "tokens_test.pt"))