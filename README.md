# Vector-Quantisation-for-Robust-Segmentation (Fork)

This repository is a **fork and extension** of the original work by **Ainkaran Santhirasekaram** and **Avinash Kori** on *Vector-Quantisation-for-Robust-Segmentation*.  
It preserves the original training framework for VQ-UNet and VQ-TransUNet, and additionally includes:

- A complete **balanced split generator** based on TZ/PZ prostate volumes  
- A fully pre-configured **conda environment** (`environment.yaml`)  
- Updated training scripts and improved checkpoint naming  
- Clear instructions for running prostate segmentation experiments

---

## ✨ Original Authors (Credit)

This repository is based on the original work:

**Ainkaran Santhirasekaram**  
(a.santhirasekaram19@imperial.ac.uk)

**Avinash Kori**  
(a.kori21@imperial.ac.uk)

Original repository: *Vector-Quantisation-for-Robust-Segmentation*

---

# 📘 Description

This project implements autoencoder-based segmentation models that integrate **Vector Quantisation (VQ) blocks** at the bottleneck.  
The codebase includes models and training pipelines for:

- **VQ-UNet**  
- **VQ-TransUNet**

Both architectures enable discrete representations that improve robustness and anatomical consistency.

---

# 🔧 Environment Setup

You can create the complete environment (Python, PyTorch Lightning, MONAI, TorchIO, etc.) by running:

```bash
conda env create -f environment.yaml
```

This installs all required libraries automatically.

# 📂 Datasets

You must prepare 3 CSV files:

```
train.csv  
validation.csv  
test.csv
```

Each must contain three columns:

```
item, images, labels
```

Pointing to ```.nii.gz``` or ```.png``` files.

Example:

```
,images,labels
0,/path/CaseXX.nii.gz,/path/CaseXX_Segmentation.nii.gz
```

# 🟦 Balanced Volume-Based Split

**Script:** ```make_balanced_split_by_volume.py```

This script creates **train/val/test** splits balanced by TZ and PZ volumes, removing center-specific biases (BCM vs RUNMC).

▶️ Usage Example

```
python make_balanced_split_by_volume.py \
  --csv ./data/Prostate/all_cases.csv \
  --out_dir . \
  --tz_label 1 --pz_label 2 \
  --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15
```

**Explanation of arguments**

- ```--csv```: CSV containing all **79 prostate cases** from BCM and RUNMC

- ```--out_dir```: Directory where the new split CSVs will be saved

- ```--tz_label``` / ```--pz_label```: Class labels for TZ and PZ in the segmentation masks

Ratios: Train/Validation/Test proportions (here 70% / 15% / 15%)

All generated CSVs will appear in the **repository root directory**, where the training script expects them.

# 🚀 Training

Once the CSVs are created, start training with:

```
python trainprostVQ.py
```

The script automatically loads:

```
./train.csv
./validation.csv
./test.csv
```

# ⚙️ Training Arguments (Where to Modify)

At the end of ```trainprostVQ.py```, the following block contains all the relevant hyperparameters:

**🔹 Model hyperparameters**

```
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
    max_epochs=100,
    check_val=5,
    output_root=root_dir,
)
```

You can modify:

- Image channels
- Number of classes
- Channels in the UNet
- Dropout
- VQ embedding size
- Loss weighting
- Number of training epochs
- Validation frequency

**🔹 Data loading**

```
data = ProstateDataModule(
    batch_size=1,
    num_workers=4,
    csv_train_img="./train.csv",
    csv_val_img="./validation.csv",
    csv_test_img="./test.csv",
)
```

**🔹 Checkpoints**

```
checkpoint_dice = ModelCheckpoint(
    dirpath=root_dir,
    filename="best_dice_epoch{epoch:02d}_dice{val_dice:.4f}",
    monitor="val_dice",
    mode="max",
    save_top_k=1,
)
```

Results are saved as:

```
best_dice_epochXX_dice0.XXXX.ckpt
best_multi_epochXX_multi0.XXXX.ckpt
```

**🔹 Lightning Trainer**

```
trainer = pytorch_lightning.Trainer(
    gpus=[0],
    max_epochs=net.max_epochs,
    check_val_every_n_epoch=net.check_val,
    callbacks=[checkpoint_dice, checkpoint_multi],
    default_root_dir=root_dir,
)
```

# 📊 Output Files

All outputs are saved in:

```
./data/Prostate/outputprostatefinal/
```

You will find:

**✔ TensorBoard logs**

Run with:

```
tensorboard --logdir ./data/Prostate/outputprostatefinal/lightning_logs
```

## 📈 Log Conversion to EPS/PNG

In the folder `Log_to_eps_png/` you will find example `.json` files extracted from **TensorBoard logs**.  

These files contain metric histories (e.g., Dice, loss, HD, multi-metric score).

The script `Logs_to_eps_png.py` allows you to generate **EPS** and **PNG** plots directly from these JSON logs.

# 🧑‍💻 Full Execution Workflow

```
conda env create -f environment.yaml

conda activate vqrs_env

python make_balanced_split_by_volume.py \
  --csv ./data/Prostate/all_cases.csv \
  --out_dir . \
  --tz_label 1 --pz_label 2 \
  --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15

python trainprostVQ.py

tensorboard --logdir ./data/Prostate/outputprostatefinal/
```

# 📚 References

1. B Nicolas Bloch, Ashali Jain, and C. Carl Jae. Data From  ROSTATE-DIAGNOSIS. 2015. doi: 10.7937/ K9 / TCIA . 2015 . FOQEUJVT. url: https://www.cancerimagingarchive.net/collection/prostate-diagnosis/.

2. Geert Litjens, Jurgen Futterer, and Henkjan Huisman. Data From Prostate-3T. 2015. doi: 10.7937/K9/TCIA.2015.QJTV5IL5. url: https://www.cancerimagingarchive.net/collection/prostate-3t/.

3.  Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural Discrete Representation Learning. In: CoRR abs/1711.00937 (2017). arXiv: 1711.00937. url: http://arxiv.org/abs/1711.00937.

4. Minghui Hu et al. Global Context with Discrete Diffusion in Vector Quantised Modelling for Image Generation. In: CoRR abs/2112.01799 (2021). arXiv: 2112.01799. url: https://arxiv.org/abs/2112.01799.

5. AinkaranSanthi. Vector-Quantisation-for-Robust-Segmentation. https://github.com/AinkaranSanthi/Vector-Quantisation-for-Robust-Segmentation. June 2022.

6. Slicer Wiki. [Online; accessed 17-November-2025]. 2019. url: https://www.slicer.org/w/index.php?title=Main_Page&oldid=62645%7D.

7. B. Nicholas Bloch et al. NCI-ISBI 2013 Challenge: Automated Segmentation of Prostate Structures (ISBI-MR-Prostate-2013). 2015. doi: 10.7937/K9/TCIA.2015.ZF0VLOPV. url: https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/
