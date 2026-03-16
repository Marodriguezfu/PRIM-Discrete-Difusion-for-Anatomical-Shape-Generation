"""Training, validation, testing, and token export for the PRIM VQ tokenizer."""

import glob
import os
import re
import sys
from typing import List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import (
    DiceMetric,
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
)
from monai.transforms import (
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    SpatialPadd,
    Spacingd,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from convnet3D_utils import VQUNet3Dposv3


@rank_zero_only
def r0(message: str) -> None:
    """Print only from rank zero when using distributed execution."""
    sys.__stdout__.write(message + "\n")
    sys.__stdout__.flush()


@torch.no_grad()
def boundary_from_onehot(onehot: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Extract a coarse boundary map from a one-hot label volume.

    Args:
        onehot: Tensor of shape [B, C, H, W, D] with binary one-hot channels.
        k: Kernel size used for the morphological approximation.

    Returns:
        Tensor of shape [B, 1, H, W, D] in [0, 1].
    """
    foreground = onehot[:, 1:, ...]
    dilated = F.max_pool3d(foreground, kernel_size=k, stride=1, padding=k // 2)
    eroded = -F.max_pool3d(-foreground, kernel_size=k, stride=1, padding=k // 2)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    boundary = boundary.max(dim=1, keepdim=True).values
    return boundary


def codebook_metrics(
    tokens: torch.Tensor,
    n_embed: int,
    dist_reduce: bool = False,
) -> tuple[float, float, torch.Tensor]:
    """
    Compute basic codebook usage statistics from token indices.

    Args:
        tokens: Token tensor with arbitrary shape.
        n_embed: Codebook size.
        dist_reduce: Whether to all-reduce histograms across distributed workers.

    Returns:
        usage_fraction, perplexity, histogram
    """
    flat = tokens.reshape(-1)
    hist = torch.bincount(flat, minlength=n_embed).float()

    if dist_reduce and dist.is_available() and dist.is_initialized():
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)

    probs = hist / hist.sum().clamp_min(1.0)
    used = (hist > 0).sum().item()
    probs_nz = probs[probs > 0]
    entropy = -(probs_nz * probs_nz.log()).sum()
    perplexity = entropy.exp().item()
    usage_fraction = used / n_embed
    return usage_fraction, perplexity, hist


class ProstateDataset(Dataset):
    """
    Dataset for the PRIM tokenizer stage.

    The tokenizer is trained on segmentation masks rather than raw MR images.
    For that reason, both the "image" and "label" MONAI keys intentionally
    point to the same mask path. This keeps the transform pipeline compatible
    with MONAI dictionary transforms while making the modeling objective explicit.
    """

    def __init__(self, csv_file: str, split: str) -> None:
        self.data = pd.read_csv(csv_file)
        self.split = split
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.5, 0.5, 1.5),
                    mode=("nearest", "nearest"),
                ),
                CropForegroundd(
                    keys=["image", "label"],
                    source_key="label",
                    margin=10,
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
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
                    num_samples=1,
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
                    mode=("nearest", "nearest"),
                ),
            ]
        )
        self.eval_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.5, 0.5, 1.5),
                    mode=("nearest", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(192, 192, 64),
                ),
                CenterSpatialCropd(keys=["image", "label"], roi_size=(192, 192, 64)),
            ]
        )
        self.samples = self._build_samples()

    def _build_samples(self) -> list[dict[str, str]]:
        samples: list[dict[str, str]] = []
        for idx in tqdm(range(len(self.data)), desc=f"Loading {self.split} data"):
            mask_path = self.data.loc[idx, "labels"]
            samples.append({"image": mask_path, "label": mask_path})
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self._get_sample(index)

        if isinstance(sample, list):
            sample = sample[0]

        image = sample["image"]
        label = sample["label"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        return {"image": image, "label": label}

    def _get_sample(self, index: int):
        sample = self.samples[index]
        if self.split == "train":
            return self.train_transforms(sample)
        return self.eval_transforms(sample)


class ProstateDataModule(pl.LightningDataModule):
    """LightningDataModule for train/validation/test splits."""

    def __init__(
        self,
        csv_train_img: str,
        csv_val_img: str,
        csv_test_img: str,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set: Optional[ProstateDataset] = None
        self.val_set: Optional[ProstateDataset] = None
        self.test_set: Optional[ProstateDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_set is None:
            self.train_set = ProstateDataset(self.csv_train_img, split="train")
            self.val_set = ProstateDataset(self.csv_val_img, split="val")
            self.test_set = ProstateDataset(self.csv_test_img, split="test")

            r0(f"#train: {len(self.train_set)}")
            r0(f"#val:   {len(self.val_set)}")
            r0(f"#test:  {len(self.test_set)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class Net(pl.LightningModule):
    """Lightning module wrapping the 3D VQ-UNet tokenizer."""

    def __init__(
        self,
        num_classes: int = 3,
        use_boundary_channel: bool = True,
        inputchannels: Optional[int] = None,
        boundary_k: int = 3,
        channels: int = 16,
        dropout: float = 0.0,
        n_embed: int = 1024,
        embed_dim: int = 256,
        w_d: float = 0.8,
        w_hd: float = 0.1,
        w_asd: float = 0.1,
        max_epochs: int = 50,
        check_val: int = 1,
        output_root: str = ".",
    ) -> None:
        super().__init__()

        if inputchannels is None:
            inputchannels = num_classes + (1 if use_boundary_channel else 0)

        self.save_hyperparameters()

        self.n_embed = n_embed
        self._model = VQUNet3Dposv3(
            inputchannels=inputchannels,
            num_classes=num_classes,
            channels=channels,
            dropout=dropout,
            n_embed=n_embed,
            embed_dim=embed_dim,
        )

        class_weights = torch.ones(num_classes, dtype=torch.float32)
        class_weights[0] = 0.05

        self.loss_function = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            weight=class_weights,
            lambda_dice=0.5,
            lambda_ce=1.2,
        )
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )

        self.dictresults = {"dice": [], "hd": [], "asd": []}
        self.best_val_dice = 0.0
        self.best_val_epoch = 0
        self.best_multi = -1.0
        self.best_multi_epoch = -1

        self.w_d = w_d
        self.w_hd = w_hd
        self.w_asd = w_asd
        self.max_epochs = max_epochs
        self.check_val = check_val
        self.warmup_epochs = 10

        self.metric_values: list[float] = []
        self.epoch_loss_values: list[float] = []

        self.output_root = output_root
        self.test_output_dir = os.path.join(self.output_root, "testimagesoutput")
        self.test_results_csv = os.path.join(self.output_root, "testresults", "result.csv")
        self._val_outputs: list[dict[str, float | int | torch.Tensor]] = []

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

    def on_test_epoch_start(self) -> None:
        self.dictresults = {"dice": [], "hd": [], "asd": []}

    def _surface_metric_mode(self) -> str:
        """
        Decide where to compute surface metrics.

        Modes:
            - "gpu": force GPU
            - "cpu": force CPU
            - "auto": try GPU if available, otherwise CPU
        """
        env_mode = os.environ.get("PRIM_SURFACE_DEVICE", "").strip().lower()
        if env_mode in {"gpu", "cpu", "auto"}:
            return env_mode

        # Default behavior:
        # - In Colab, prefer CPU for MONAI surface metrics due to CuPy/cuCIM issues.
        # - Elsewhere, use auto.
        is_colab = ("google.colab" in sys.modules) or ("COLAB_GPU" in os.environ)
        return "cpu" if is_colab else "auto"

    @torch.no_grad()
    def _compute_surface_metrics(
        self,
        pred_stack: torch.Tensor,
        label_stack: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute HD and ASD with automatic fallback.

        This keeps training/inference on GPU, but can move only the metric
        computation to CPU when needed.
        """
        mode = self._surface_metric_mode()

        def _run_metrics(p: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hd = torch.nanmean(
                compute_hausdorff_distance(
                    y_pred=p,
                    y=y,
                    include_background=False,
                )
            )
            asd = torch.nanmean(
                compute_average_surface_distance(
                    y_pred=p,
                    y=y,
                    include_background=False,
                )
            )
            return hd, asd

        # Force CPU
        if mode == "cpu":
            return _run_metrics(pred_stack.detach().cpu(), label_stack.detach().cpu())

        # Force GPU
        if mode == "gpu":
            return _run_metrics(pred_stack, label_stack)

        # Auto: try GPU first if tensors are on CUDA, otherwise CPU
        if pred_stack.is_cuda and label_stack.is_cuda:
            try:
                return _run_metrics(pred_stack, label_stack)
            except Exception as exc:
                r0(
                    f"[WARN] GPU surface metrics failed ({type(exc).__name__}). "
                    "Falling back to CPU for HD/ASD."
                )

        return _run_metrics(pred_stack.detach().cpu(), label_stack.detach().cpu())

    def forward(self, x: torch.Tensor, return_indices: bool = False):
        logits, emb_loss, _, indices = self._model(x)
        if return_indices:
            return logits, emb_loss, indices
        return logits, emb_loss

    def forward_logits(self, x: torch.Tensor, return_indices: bool = False):
        """Forward pass that only returns segmentation logits (and optionally indices)."""
        logits, _, _, indices = self._model(x)
        if return_indices:
            return logits, indices
        return logits

    def encode_x(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        Encode model input into token indices.

        Args:
            x: Input tensor [B, C, H, W, D].
            add_noise: Whether to inject encoder noise before quantization.

        Returns:
            Token indices [B, Ht, Wt, Dt].
        """
        return self._model.encode(x, add_noise=add_noise, return_quant=False)

    #def encode_labels(self, labels: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
    #    """Convenience wrapper that builds model input directly from raw labels."""
    #    x = self.build_x(labels)
    #    return self.encode_x(x, add_noise=add_noise)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices back to segmentation logits."""
        return self._model.decode_indices(indices)

    @torch.no_grad()
    def decode_indices_to_seg(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices and return argmax segmentation."""
        return self._model.decode_indices_to_seg(indices)

    @torch.no_grad()
    def token_stats_by_class(
        self,
        indices: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
    ) -> None:
        """
        Print the most frequent token ids for each class in a batch.

        Args:
            indices: Token ids [B, Ht, Wt, Dt].
            labels: Raw labels [B, 1, H, W, D] or [B, H, W, D].
            num_classes: Number of semantic classes.
        """
        if labels.ndim == 5:
            label_volume = labels.float()
            if label_volume.shape[1] != 1:
                label_volume = label_volume.argmax(dim=1, keepdim=True).float()
        elif labels.ndim == 4:
            label_volume = labels.unsqueeze(1).float()
        else:
            raise ValueError(f"Unexpected label rank: {labels.ndim}")

        label_ds = F.interpolate(
            label_volume,
            size=indices.shape[1:],
            mode="nearest",
        ).long().squeeze(1)

        lines = [
            f"[Epoch {self.current_epoch}] Token stats by class "
            f"(grid={tuple(indices.shape[1:])})"
        ]

        for class_id in range(num_classes):
            mask = label_ds == class_id
            class_ids = indices[mask]

            if class_ids.numel() == 0:
                lines.append(f"  class {class_id}: n_tokens=0 (not present in batch)")
                continue

            unique_ids, counts = torch.unique(class_ids, return_counts=True)
            k = min(5, counts.numel())
            top_counts, top_positions = torch.topk(counts, k=k)

            lines.append(
                f"  class {class_id}: n_tokens={class_ids.numel()}, "
                f"unique={unique_ids.numel()}"
            )
            for j in range(k):
                pos = top_positions[j].item()
                lines.append(
                    f"    id {int(unique_ids[pos])}  count {int(top_counts[j])}"
                )

        r0("\n".join(lines))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self._model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        )

    def _embedding_weight(self) -> float:
        cap = 1.0
        return min(cap, cap * float(self.current_epoch + 1) / float(self.warmup_epochs))

    def training_step(self, batch, batch_idx: int):
        labels = batch["label"]
        x = self.build_x(labels)

        logits, emb_loss, indices = self.forward(x, return_indices=True)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.token_stats_by_class(
                indices=indices,
                labels=labels,
                num_classes=self.hparams.num_classes,
            )

        usage_fraction, perplexity, hist = codebook_metrics(
            indices,
            self.n_embed,
            dist_reduce=True,
        )
        self.log(
            "codebook/usage_frac",
            usage_fraction,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "codebook/perplexity",
            perplexity,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        if self.trainer.is_global_zero and (batch_idx == 0 or batch_idx % 20 == 0):
            k = min(10, hist.numel())
            top_ids = torch.topk(hist, k=k).indices
            top_counts = hist[top_ids].to(torch.int64)
            pairs = list(zip(top_ids.tolist(), top_counts.tolist()))
            r0(
                f"[Train][Epoch {self.current_epoch}][b{batch_idx}] "
                f"usage={usage_fraction:.3f} ppl={perplexity:.1f} top={pairs}"
            )

        seg_loss = self.loss_function(logits, labels)
        emb_weight = self._embedding_weight()
        loss = seg_loss + emb_weight * emb_loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_seg_loss",
            seg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_emb_loss",
            emb_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_emb_w",
            emb_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            gt = labels.squeeze(1)
            eps = 1e-6
            dices = []
            for class_id in range(1, self.hparams.num_classes):
                pred_mask = (pred == class_id).float()
                gt_mask = (gt == class_id).float()
                intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
                denom = pred_mask.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3)) + eps
                dices.append(2.0 * intersection / denom)
            dice_scalar = torch.stack(dices, dim=1).mean().item()

        self.log(
            "train_dice",
            dice_scalar,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.current_epoch == 0 and batch_idx == 0:
            unique_labels = torch.unique(labels)
            r0(
                f"[DEBUG] unique labels: {unique_labels.detach().cpu().tolist()} "
                f"max={int(labels.max())}"
            )

        return loss

    def validation_step(self, batch, batch_idx: int):
        labels = batch["label"]
        x = self.build_x(labels)

        logits, emb_loss, indices = self.forward(x, return_indices=True)

        seg_loss = self.loss_function(logits, labels)
        emb_weight = self._embedding_weight()
        loss = seg_loss + emb_weight * emb_loss

        if batch_idx == 0:
            usage_fraction, perplexity, _ = codebook_metrics(
                indices,
                self.hparams.n_embed,
            )
            self.log("codebook_usage", usage_fraction, prog_bar=True, logger=True)
            self.log("codebook_perplexity", perplexity, prog_bar=False, logger=True)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_seg_loss", seg_loss, prog_bar=False, logger=True)
        self.log("val_emb_loss", emb_loss, prog_bar=False, logger=True)
        self.log("val_emb_w", emb_weight, prog_bar=False, logger=True)

        preds = [self.post_pred(item) for item in self._iter_batch(logits)]
        labels_onehot = [self.post_label(item) for item in self._iter_batch(labels)]
        preds_stack = torch.stack(preds, dim=0)
        labels_stack = torch.stack(labels_onehot, dim=0)

        self.dice_metric(y_pred=preds, y=labels_onehot)

        hd, asd = self._compute_surface_metrics(preds_stack, labels_stack)

        output = {
            "val_loss": loss.detach(),
            "hd": hd.detach(),
            "asd": asd.detach(),
            "val_number": logits.shape[0],
        }
        self._val_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        outputs = self._val_outputs
        if not outputs:
            return

        val_loss = 0.0
        hd = 0.0
        asd = 0.0
        num_items = 0

        for output in outputs:
            batch_size = int(output["val_number"])
            val_loss += float(output["val_loss"].item()) * batch_size
            hd += float(output["hd"].item()) * batch_size
            asd += float(output["asd"].item()) * batch_size
            num_items += batch_size

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
            self.w_d * mean_val_dice
            + self.w_hd * hd_score
            + self.w_asd * asd_score
        )

        self.log("val_loss", mean_val_loss, prog_bar=True, logger=True)
        self.log("val_dice", mean_val_dice, prog_bar=True, logger=True)
        self.log("val_hd", mean_hd, prog_bar=False, logger=True)
        self.log("val_asd", mean_asd, prog_bar=False, logger=True)
        self.log("val_multi", multi_score, prog_bar=True, logger=True)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        if multi_score > self.best_multi:
            self.best_multi = multi_score
            self.best_multi_epoch = self.current_epoch

        r0(
            f"epoch {self.current_epoch}: "
            f"Dice={mean_val_dice:.4f}, HD={mean_hd:.4f}, ASD={mean_asd:.4f}, "
            f"Multi={multi_score:.4f}\n"
            f"best Dice={self.best_val_dice:.4f} @ epoch {self.best_val_epoch}, "
            f"best Multi={self.best_multi:.4f} @ epoch {self.best_multi_epoch}\n"
        )

        self.metric_values.append(mean_val_dice)
        self._val_outputs = []

    def test_step(self, batch, batch_idx: int):
        input_masks = batch["image"]
        labels = batch["label"]
        x = self.build_x(labels)

        logits = self.forward_logits(x)
        loss = self.loss_function(logits, labels.long())
        r0(f"test loss: {loss:.3f}")

        outputs = [self.post_pred(item) for item in self._iter_batch(logits)]
        labels_onehot = [self.post_label(item) for item in self._iter_batch(labels)]

        outputs_stack = torch.stack(outputs, dim=0)
        labels_stack = torch.stack(labels_onehot, dim=0)

        dice = torch.mean(
            compute_dice(
                y_pred=outputs_stack,
                y=labels_stack,
                include_background=False,
            )
        )
        hd, asd = self._compute_surface_metrics(outputs_stack, labels_stack)

        r0(f"dice: {dice:.4f} hd: {hd:.4f} asd: {asd:.4f}")

        self.log("test_dice", dice)
        self.log("test_hd", hd)
        self.log("test_asd", asd)

        self.dictresults["dice"].append(float(dice.detach().cpu()))
        self.dictresults["asd"].append(float(asd.detach().cpu()))
        self.dictresults["hd"].append(float(hd.detach().cpu()))

        os.makedirs(self.test_output_dir, exist_ok=True)
        for i in range(len(outputs)):
            output = torch.squeeze(outputs[i])
            output = torch.argmax(output, dim=0).cpu().numpy().astype(np.float32)

            label = torch.squeeze(labels_onehot[i])
            label = torch.argmax(label, dim=0).cpu().numpy().astype(np.float32)

            input_mask = torch.squeeze(input_masks[i]).cpu().numpy().astype(np.float32)

            affine = np.eye(4)
            nib.save(
                nib.Nifti1Image(output, affine),
                os.path.join(self.test_output_dir, f"seg{i}_batch{batch_idx}.nii"),
            )
            nib.save(
                nib.Nifti1Image(label, affine),
                os.path.join(self.test_output_dir, f"label{i}_batch{batch_idx}.nii"),
            )
            nib.save(
                nib.Nifti1Image(input_mask, affine),
                os.path.join(self.test_output_dir, f"input{i}_batch{batch_idx}.nii"),
            )

        return loss

    def on_test_epoch_end(self) -> None:
        results_dir = os.path.dirname(self.test_results_csv)
        os.makedirs(results_dir, exist_ok=True)
        pd.DataFrame(self.dictresults).to_csv(self.test_results_csv, index=False)

    def _iter_batch(self, x: torch.Tensor):
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        return torch.unbind(x, dim=0)

    def build_x(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Convert integer segmentation labels into the model input tensor.

        Input format:
            labels: [B, 1, H, W, D]
        Output format:
            one-hot labels optionally concatenated with a boundary channel:
            [B, C, H, W, D] or [B, C + 1, H, W, D]
        """
        labels = labels.long()
        onehot = F.one_hot(
            labels.squeeze(1),
            num_classes=self.hparams.num_classes,
        ).permute(0, 4, 1, 2, 3).float()

        if self.hparams.use_boundary_channel:
            boundary = boundary_from_onehot(onehot, k=self.hparams.boundary_k)
            x = torch.cat([onehot, boundary], dim=1)
        else:
            x = onehot

        expected_channels = self.hparams.inputchannels
        if x.shape[1] != expected_channels:
            raise RuntimeError(
                f"Channel mismatch: x has {x.shape[1]} channels but model expects "
                f"{expected_channels}. "
                f"(num_classes={self.hparams.num_classes}, "
                f"use_boundary_channel={self.hparams.use_boundary_channel})"
            )
        return x


def _extract_epoch_from_name(path: str) -> int:
    """Extract epoch number from a checkpoint filename."""
    name = os.path.basename(path)

    match = re.search(r"epoch=(\d+)", name)
    if match:
        return int(match.group(1))

    match = re.search(r"epoch(\d+)", name)
    if match:
        return int(match.group(1))

    return -1


def get_highest_epoch_ckpt(
    root_dir: str,
    patterns: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Return the checkpoint with the highest epoch among the provided glob patterns.
    """
    if patterns is None:
        patterns = ["best_*.ckpt"]

    ckpts: list[str] = []
    for pattern in patterns:
        ckpts.extend(glob.glob(os.path.join(root_dir, pattern)))

    if not ckpts:
        return None

    parsed = [(ckpt, _extract_epoch_from_name(ckpt)) for ckpt in ckpts]
    parsed_ok = [(ckpt, epoch) for ckpt, epoch in parsed if epoch >= 0]

    if parsed_ok:
        parsed_ok.sort(key=lambda item: item[1])
        return parsed_ok[-1][0]

    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]


def export_token_indices(model: Net, dataloader, out_path: str) -> None:
    """Export token indices for an entire dataloader split."""
    was_training = model.training
    model.eval()

    all_tokens = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"]
            if hasattr(labels, "as_tensor"):
                labels = labels.as_tensor()
            labels = labels.to(model.device)
            x = model.build_x(labels)
            indices = model.encode_x(x, add_noise=False)
            all_tokens.append(indices.cpu())

    tokens = torch.cat(all_tokens, dim=0)
    torch.save(tokens, out_path)
    r0(f"Saved tokens: {tokens.shape} -> {out_path}")

    if was_training:
        model.train()


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    resume_training = False
    resume_mode = "highest_epoch"
    root_dir = "./data/Prostate/outputprostatefinal"

    resume_ckpt = None
    if resume_training and resume_mode == "highest_epoch":
        resume_ckpt = get_highest_epoch_ckpt(root_dir, patterns=["best_*.ckpt"])
        if resume_ckpt is None:
            r0(
                f"[WARN] resume_training=True but no best_*.ckpt found in {root_dir}. "
                "Training from scratch."
            )
        else:
            r0(f"[INFO] Resuming training from checkpoint: {resume_ckpt}")

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
        max_epochs=2,
        check_val=1,
        output_root=root_dir,
    )

    data = ProstateDataModule(
        batch_size=1,
        num_workers=2,
        csv_train_img="./train.csv",
        csv_val_img="./validation.csv",
        csv_test_img="./test.csv",
    )

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

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=net.max_epochs,
        check_val_every_n_epoch=net.check_val,
        callbacks=[checkpoint_dice, checkpoint_multi],
        default_root_dir=root_dir,
        enable_progress_bar=False,
        log_every_n_steps=2,
    )

    try:
        if resume_ckpt is not None:
            trainer.fit(net, data, ckpt_path=resume_ckpt)
        else:
            trainer.fit(net, data)
    except TypeError:
        if resume_ckpt is not None:
            trainer = pl.Trainer(
                gpus=[0],
                max_epochs=net.max_epochs,
                check_val_every_n_epoch=net.check_val,
                callbacks=[checkpoint_dice, checkpoint_multi],
                default_root_dir=root_dir,
                resume_from_checkpoint=resume_ckpt,
            )
        trainer.fit(net, data)

    net.eval()
    batch = next(iter(data.val_dataloader()))
    labels = batch["label"].to(net.device)

    x = net.build_x(labels)
    logits_forward, _, indices = net.forward(x, return_indices=True)
    logits_decoded = net.decode_indices(indices)
    r0(f"max abs diff: {(logits_forward - logits_decoded).abs().max().item()}")

    best_dice_path = get_highest_epoch_ckpt(root_dir, patterns=["best_dice_epoch*.ckpt"])
    best_multi_path = get_highest_epoch_ckpt(root_dir, patterns=["best_multi_epoch*.ckpt"])

    if best_dice_path is None:
        raise FileNotFoundError(
            f"No best_dice_epoch*.ckpt file was found in {root_dir}"
        )
    if best_multi_path is None:
        raise FileNotFoundError(
            f"No best_multi_epoch*.ckpt file was found in {root_dir}"
        )

    r0(f"Best DICE checkpoint: {best_dice_path}")
    r0(f"Best MULTI checkpoint: {best_multi_path}")

    output_root_dice = os.path.join(root_dir, "test_best_dice")
    model_dice = Net.load_from_checkpoint(
        best_dice_path,
        output_root=output_root_dice,
        strict=False,
    )
    os.makedirs(model_dice.test_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_dice.test_results_csv), exist_ok=True)

    r0(f"=== Test model best_dice: results in {output_root_dice} ===")
    trainer.test(model_dice, dataloaders=data.test_dataloader())

    output_root_multi = os.path.join(root_dir, "test_best_multi")
    model_multi = Net.load_from_checkpoint(
        best_multi_path,
        output_root=output_root_multi,
        strict=False,
    )
    os.makedirs(model_multi.test_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_multi.test_results_csv), exist_ok=True)

    r0(f"=== Test model best_multi: results in {output_root_multi} ===")
    trainer.test(model_multi, dataloaders=data.test_dataloader())

    export_token_indices(net, data.train_dataloader(), os.path.join(root_dir, "tokens_train.pt"))
    export_token_indices(net, data.val_dataloader(), os.path.join(root_dir, "tokens_val.pt"))
    export_token_indices(net, data.test_dataloader(), os.path.join(root_dir, "tokens_test.pt"))
