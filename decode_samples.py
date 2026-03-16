"""Decode sampled token volumes back into segmentation volumes.

This script loads:
1. The latest valid VQ-VAE checkpoint from PRIM training.
2. A saved token sample file from the token prior.

It then decodes token indices into segmentation masks and exports them as
NIfTI volumes.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from trainprostVQ import Net


_EPOCH_PATTERN = re.compile(r"epoch(\d+)")


def extract_epoch(path_like: str | Path) -> int:
    """Extract the epoch number from a checkpoint or sample filename.

    Files without an epoch pattern receive -1 so they sort first.
    """
    match = _EPOCH_PATTERN.search(Path(path_like).name)
    return int(match.group(1)) if match else -1


def get_latest_checkpoint(root: Path) -> Path:
    """Return the highest-epoch best_dice checkpoint."""
    candidates = [
        root / name
        for name in os.listdir(root)
        if name.startswith("best_dice_epoch") and name.endswith(".ckpt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No best_dice_epoch*.ckpt found in {root}")

    return max(candidates, key=extract_epoch)


def get_latest_token_samples(root: Path) -> Path:
    """Return the preferred token sample file.

    Priority:
    1. samples_tokens_best.pt
    2. Highest-epoch file matching samples_tokens_ep*.pt
    """
    best_tokens = root / "samples_tokens_best.pt"
    if best_tokens.exists():
        return best_tokens

    sample_files = [Path(p) for p in glob.glob(str(root / "samples_tokens_ep*.pt"))]
    if not sample_files:
        raise FileNotFoundError(f"No token sample files found in {root}")

    return max(sample_files, key=extract_epoch)


def load_token_tensor(path: Path, device: str) -> torch.Tensor:
    """Load sampled tokens safely on CPU, then move them to the target device."""
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(loaded, dict) and "tokens" in loaded:
        loaded = loaded["tokens"]

    if not isinstance(loaded, torch.Tensor):
        raise TypeError(f"Unsupported token file format in {path}. Expected a tensor.")

    return loaded.long().to(device)


def main() -> None:
    """Decode sampled token grids and save NIfTI segmentation volumes."""
    root = Path("./data/Prostate/outputprostatefinal")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = get_latest_checkpoint(root)
    tokens_path = get_latest_token_samples(root)

    print("[decode] root:", root)
    print("[decode] ckpt:", checkpoint_path)
    print("[decode] tokens:", tokens_path)
    print("[decode] device:", device)

    tokens = load_token_tensor(tokens_path, device=device)
    if tokens.dim() != 4:
        raise ValueError(
            f"Expected sampled tokens with shape [B, H, W, D], but got {tuple(tokens.shape)}."
        )

    model = Net.load_from_checkpoint(str(checkpoint_path), strict=False).to(device)
    model.eval()

    print(
        "[decode] tokens shape:",
        tuple(tokens.shape),
        "min/max:",
        int(tokens.min().item()),
        int(tokens.max().item()),
    )

    decoded = model.decode_indices_to_seg(tokens)  # [B, 1, H, W, D]
    segmentation = decoded.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    print("[decode] seg shape:", segmentation.shape)

    out_dir = root / "decoded_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for index in range(segmentation.shape[0]):
        out_path = out_dir / f"sample_{index}.nii.gz"
        nib.save(nib.Nifti1Image(segmentation[index], np.eye(4)), str(out_path))
        print("[decode] saved:", out_path)


if __name__ == "__main__":
    main()
