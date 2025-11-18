import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import glob
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl
from transformers import ViTForImageClassification
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
import nibabel as nib
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

from monai.config import print_config
from monai.metrics import DiceMetric, compute_hausdorff_distance, compute_average_surface_distance,  compute_meandice
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from transformers.optimization import AdamW
from transformers import ViTForImageClassification, BeitFeatureExtractor, BeitForImageClassification, BeitForMaskedImageModeling, BeitModel
import os
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DataParallel
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchvision
from convnet3D_utils import  UNet3Dv2,  VQUNet3Dposv3, GumbelUNet3Dpos
import torchvision.transforms as T
from transformers.optimization import get_cosine_schedule_with_warmup
from einops import rearrange
from pytorch_lightning.callbacks import LearningRateMonitor
from functools import partial

from torchvision import models
import pytorch_lightning as pl
import pytorch_lightning

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
import math
from argparse import ArgumentParser


image_size = (224, 224)
num_classes = 14
batch_size = 100
epochs = 300
num_workers = 4


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
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
               # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(192, 192, 64),
                ),

                CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(192, 192, 64),
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandRotated(
                    keys=["image", "label"],
                    range_x = 0.2,
                    prob=0.10,
                ),

                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.4),
                RandGaussianNoised(keys = "image", std = 0.05, prob = 0.15),
                RandAdjustContrastd(keys="image",prob = 0.2),
                RandBiasFieldd(keys="image",  prob=0.2),

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
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                # CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(
                    keys=["image", "label"],
                    spatial_size=(192, 192, 64),
                ),
                CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(192, 192, 64),
                ),
                #ToTensord(keys=["image", "label"]),
            ]
        )

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = self.data.loc[idx, 'images']
            img_label = self.data.loc[idx, 'labels']

            sample = {'image': img_path, 'label': img_label}

            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image'])
        label = torch.from_numpy(sample['label'])
        #label = torch.where(label > 0, 1, 0)
        return {'image': image, 'label': label}

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
        inputchannels=1,
        num_classes=3,
        channels=16,
        dropout=0.0,
        n_embed=1024,
        embed_dim=256,
        w_d = 0.8,
        w_hd = 0.1,
        w_asd = 0.1,
        max_epochs=50,
        check_val=1,
        output_root=".",
    ):
        super().__init__()
        self.save_hyperparameters()
#        self.automatic_optimization = False
       ### data = pd.read_csv(train_csv)
        #self.trainlen = len(data)
        self._model = VQUNet3Dposv3(
        #self._model = GumbelUNet2Dpos(
            inputchannels=inputchannels,
            num_classes=num_classes,
            channels=channels,
            dropout=dropout,
            n_embed=n_embed,
            embed_dim=embed_dim,
        )
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=3)
        self.post_label = AsDiscrete(to_onehot=3)
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
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
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []

        self.output_root = output_root
        self.test_output_dir = os.path.join(self.output_root, "testimagesoutput")
        self.test_results_csv = os.path.join(self.output_root, "testresults", "result.csv")

        #self.bn = int(math.ceil(self.trainlen/batch_size))

    def forward(self, input):
        quant, loss, latents = self._model(input)
        return quant, loss

    def forward1(self, input):
        quant, loss, latents = self._model(input)
        return quant

    def get_input(self, batch, k):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        output, embloss = self.forward(images)
        loss = self.loss_function(output, labels) + embloss

        print('train Loss: %.3f' % (loss))

        # Training loss log
        self.log(
            "train_loss",
            loss,
            on_step=False,      
            on_epoch=True,      
            prog_bar=True,
            logger=True,
        )

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
        images, labels = batch['image'], batch['label']
        outputs = self.forward1(images)

        loss = self.loss_function(outputs, labels)
        print('val Loss: %.3f' % (loss))
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels1 = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels1)
        hd = compute_hausdorff_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels1, dim=0), include_background=False)
        asd = compute_average_surface_distance(y_pred=torch.stack(outputs, dim=0), y=torch.stack(labels1, dim=0), include_background=False)
        return {"val_loss": loss, "hd": hd, "asd": asd, "val_number": len(outputs)}

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
        images, labels = batch['image'], batch['label']
        outputs = self.forward1(images)

        loss = self.loss_function(outputs, labels)
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

if __name__ == '__main__':
        pl.seed_everything(42, workers=True)
        root_dir = "./data/Prostate/outputprostatefinal"
        net = Net(
            inputchannels=1,
            num_classes=3,
            channels=16,
            dropout=0.0,
            n_embed=1024,
            embed_dim=256,
            w_d = 0.8,
            w_hd = 0.1,
            w_asd = 0.1,
            max_epochs=5,
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
            gpus=[0],
            max_epochs=net.max_epochs,
            check_val_every_n_epoch=net.check_val,
            #callbacks=[checkpoint_dice, checkpoint_multi, early_stopping],
            callbacks=[checkpoint_dice,checkpoint_multi],
            default_root_dir=root_dir,
        )

        os.makedirs(net.test_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(net.test_results_csv), exist_ok=True)
        trainer.fit(net, data)

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

        output_root_dice = os.path.join(root_dir, "test_best_dice")
        model_dice = Net.load_from_checkpoint(
            best_dice_path,
            output_root=output_root_dice,
        )
        os.makedirs(model_dice.test_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_dice.test_results_csv), exist_ok=True)

        print(f"=== Test model best_dice: results in {output_root_dice} ===")
        trainer.test(model_dice, data.test_dataloader())


        output_root_multi = os.path.join(root_dir, "test_best_multi")
        model_multi = Net.load_from_checkpoint(
            best_multi_path,
            output_root=output_root_multi,
        )
        os.makedirs(model_multi.test_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_multi.test_results_csv), exist_ok=True)

        print(f"=== Test model best_multi: results in {output_root_multi} ===")
        trainer.test(model_multi, data.test_dataloader())
