import os, glob
import torch
import nibabel as nib
import numpy as np
from trainprostVQ import Net

def main():
    root = "./data/Prostate/outputprostatefinal"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = sorted([p for p in os.listdir(root) if p.startswith("best_dice_epoch") and p.endswith(".ckpt")])
    if not ckpts:
        raise FileNotFoundError(f"No best_dice_epoch*.ckpt found in {root}")
    ckpt_path = os.path.join(root, ckpts[-1])

    best_tok = os.path.join(root, "samples_tokens_best.pt")
    if os.path.exists(best_tok):
        tokens_path = best_tok
    else:
        sample_files = sorted(glob.glob(os.path.join(root, "samples_tokens_ep*.pt")))
        if not sample_files:
            raise FileNotFoundError(f"No samples_tokens_ep*.pt found in {root}")
        tokens_path = sample_files[-1]

    print("[decode] root:", root)
    print("[decode] ckpt:", ckpt_path)
    print("[decode] tokens:", tokens_path)
    print("[decode] device:", device)

    # load tokens safely on CPU then move to device
    tokens = torch.load(tokens_path, map_location="cpu").long().to(device)

    net = Net.load_from_checkpoint(ckpt_path, strict=False).to(device)
    net.eval()

    print("[decode] tokens shape:", tuple(tokens.shape), "min/max:", int(tokens.min()), int(tokens.max()))

    seg = net.decode_indices_to_seg(tokens)  # [B,1,H,W,D]
    seg = seg.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    print("[decode] seg shape:", seg.shape)

    out_dir = os.path.join(root, "decoded_samples")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(seg.shape[0]):
        out_path = os.path.join(out_dir, f"sample_{i}.nii.gz")
        nib.save(nib.Nifti1Image(seg[i], np.eye(4)), out_path)
        print("[decode] saved:", out_path)


if __name__ == "__main__":
    main()