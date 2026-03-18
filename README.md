# PRIM – Discrete Diffusion for Anatomical Shape Generation (Prostate Version)

This repository contains the **final PRIM pipeline** used to learn a discrete anatomical shape model from **prostate segmentation masks**.

The current implementation is centered on a **3D VQ tokenizer + discrete diffusion prior**:
1. create balanced train / validation / test splits,
2. train the tokenizer on prostate segmentation volumes,
3. export discrete token grids,
4. train a diffusion-style token prior,
5. decode generated tokens back into segmentation masks.

This version is focused on **prostate anatomical shape generation**. The active code works on **segmentation masks**, not directly on raw MRI intensities.

---

## Recommended Way to Run the Project

The recommended entry point is:

```text
PRIM_pipeline.ipynb
```

This notebook already executes the **full pipeline end-to-end**, so the project does **not need to be run stage by stage from the console** unless you explicitly want to do that.

The notebook is designed to work in both settings:

- **Google Colab**
- **Local computer** (for example with Jupyter Notebook, JupyterLab, or VS Code)

### In Google Colab

The notebook automatically checks whether it is running in Colab. In that case it:

- mounts Google Drive,
- installs the required missing packages used in the notebook,
- changes the working directory to the project folder.

The current Colab setup in the notebook assumes the repository is located at:

```text
/content/drive/MyDrive/PRIM
```

If your repository is stored somewhere else in Drive, only that path cell needs to be adjusted.

### On a local computer

Open `PRIM_pipeline.ipynb` from the repository root and run the cells in order.

For local execution, you only need:

- the project dependencies installed in your Python environment,
- the dataset available in the expected paths,
- the notebook launched from the repository root so relative paths resolve correctly.

---

## What the Notebook Executes

`PRIM_pipeline.ipynb` is the practical orchestrator of the project. It runs the following stages:

### Stage 0 — Balanced dataset split
Runs `make_balanced_split_by_volume.py` to create:
- `mask_volumes.csv`
- `train.csv`
- `validation.csv`
- `test.csv`

### Stage 1 — Train the VQ tokenizer
Runs `trainprostVQ.py` to train the 3D tokenizer and save checkpoints, test outputs, and token files.

### Stage 2 — Token extraction / token file preparation
Uses the exported token tensors and makes sure they are saved in a clean integer format for the diffusion prior.

### Stage 3 — Train the token prior
Runs `train_token_prior.py` to train the discrete diffusion-style prior over token grids.

### Stage 4 — Decode generated samples
Runs `decode_samples.py` to reconstruct generated token grids into NIfTI segmentation volumes.

### Additional notebook diagnostics
The notebook also includes extra analysis cells for:
- decoded-sample sanity checks,
- reconstruction comparison against a real validation case,
- token usage statistics,
- label distribution checks,
- diversity checks between generated samples,
- nearest-training-token similarity checks.

---

## Optional Alternative: Run Each Script Manually

Even though the notebook is the main workflow, the scripts still exist and can be executed separately when needed.

Typical manual order:

```bash
python make_balanced_split_by_volume.py
python trainprostVQ.py
python tokens.py
python train_token_prior.py
python decode_samples.py
```

This manual execution path is optional. The final project workflow is primarily intended to be run through `PRIM_pipeline.ipynb`.

---

## Repository Components

### `PRIM_pipeline.ipynb`
Main end-to-end notebook. This is the **recommended execution interface** for the project, especially for reproducible runs in Colab or on a local machine.

### `make_balanced_split_by_volume.py`
Creates balanced train / validation / test splits from the prostate dataset using TZ / PZ mask volumes.

### `trainprostVQ.py`
Trains, validates, tests, and exports tokens for the 3D VQ tokenizer.

### `convnet3D_utils.py`
Contains the core 3D tokenizer architecture used by `trainprostVQ.py`.

### `tokens.py`
Loads exported token tensors and generates basic token statistics and plots. This file is mainly for **inspection and visualization**, not for core training.

### `train_token_prior.py`
Trains the discrete diffusion-style prior on token grids.

### `token_diffusion.py`
Contains the token corruption process, denoiser utilities, diffusion schedule, and sampling helpers.

### `decode_samples.py`
Decodes sampled token grids back into segmentation volumes.

---

## Current Modeling Choice

The tokenizer is trained on **segmentation structure**, not on raw MR appearance.

In practice, `trainprostVQ.py` converts the segmentation label into:
- one-hot label channels,
- optionally one extra boundary channel.

So the model learns a discrete latent representation of **anatomical segmentation volumes**.

---

## Dataset Format

The pipeline expects CSV files with the columns:

```text
images,labels
```

Supported mask formats include:

- `.nii`
- `.nii.gz`
- `.nrrd`

In the current tokenizer pipeline, the dataloader intentionally uses the **mask path for both `image` and `label`**, because the training target is the segmentation structure itself.

---

## Output Root Directory

The main output root used by the current code is:

```text
./data/Prostate/outputprostatefinal/
```

Important generated content includes:

- Lightning logs,
- tokenizer checkpoints,
- tokenizer test outputs,
- exported token tensors,
- token plots,
- diffusion checkpoints,
- generated token samples,
- decoded segmentation samples.

---

## Important Output Files

### Split files
```text
train.csv
validation.csv
test.csv
mask_volumes.csv
```

### Tokenizer outputs
```text
./data/Prostate/outputprostatefinal/best_dice_epoch*.ckpt
./data/Prostate/outputprostatefinal/best_multi_epoch*.ckpt
./data/Prostate/outputprostatefinal/tokens_train.pt
./data/Prostate/outputprostatefinal/tokens_val.pt
./data/Prostate/outputprostatefinal/tokens_test.pt
```

### Diffusion / prior outputs
```text
./data/Prostate/outputprostatefinal/token_prior_best.pt
./data/Prostate/outputprostatefinal/token_prior_last.pt
./data/Prostate/outputprostatefinal/samples_tokens_ep*.pt
./data/Prostate/outputprostatefinal/samples_tokens_best.pt
```

### Decoded generated samples
```text
./data/Prostate/outputprostatefinal/decoded_samples/sample_0.nii.gz
```

---

## Token Inspection

`tokens.py` is not the main execution entry point of the project. It is only used to inspect already saved token files.

It loads:
- `tokens_train.pt`
- `tokens_val.pt`
- `tokens_test.pt`

and saves visualization outputs under:

```text
./data/Prostate/outputprostatefinal/tokens_plots/
```

---

## TensorBoard

Tokenizer training logs are written inside the Lightning output directory. To inspect them with TensorBoard:

```bash
tensorboard --logdir ./data/Prostate/outputprostatefinal/lightning_logs
```

---

## Credit to the Original Repository

This repository is adapted from the original work:

- **Ainkaran Santhirasekaram**
- **Avinash Kori**

Original repository: *Vector-Quantisation-for-Robust-Segmentation*

The PRIM version keeps the general idea of vector-quantized segmentation modeling, but the current repository has been adapted into a **prostate mask tokenizer + discrete diffusion generation pipeline** with a notebook-driven workflow.

---

## Notes About the Current Final Version

Compared with older repository descriptions, the current final version should be understood as follows:

- the main practical entry point is **`PRIM_pipeline.ipynb`**,
- the full workflow can be run from **Colab or a local computer**,
- manual console execution is **optional**, not the primary interface,
- the split generator is a fixed script without CLI arguments,
- tokenizer training is performed on segmentation masks encoded as one-hot channels plus an optional boundary channel,
- token export is part of the tokenizer workflow,
- the diffusion stage is separated into `train_token_prior.py` and `token_diffusion.py`,
- generated samples are decoded by `decode_samples.py`.

---

## References

1. Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. *Neural Discrete Representation Learning*. NeurIPS, 2017.
2. Minghui Hu et al. *Global Context with Discrete Diffusion in Vector Quantised Modelling for Image Generation*. CVPR, 2022.
3. The repository is adapted from the original *Vector-Quantisation-for-Robust-Segmentation* codebase.
