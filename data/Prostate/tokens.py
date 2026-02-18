import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/content/drive/MyDrive/PRIM/data/Prostate/outputprostatefinal"
FILES = ["tokens_train.pt", "tokens_val.pt", "tokens_test.pt"]

OUT_DIR = os.path.join(BASE_DIR, "tokens_plots")
os.makedirs(OUT_DIR, exist_ok=True)

def load_tokens(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)

def summarize_tokens(name: str, tokens: torch.Tensor):
    print(f"\n=== {name} ===")
    print("type:", type(tokens))
    if isinstance(tokens, torch.Tensor):
        print("shape:", tuple(tokens.shape))
        print("dtype:", tokens.dtype)
        print("min/max:", tokens.min().item(), tokens.max().item())
        print("#unique:", torch.unique(tokens).numel())
    else:
        print("Non-tensor object keys/len:", getattr(tokens, "keys", lambda: None)(), getattr(tokens, "__len__", lambda: None)())

def plot_slice(name: str, tokens: torch.Tensor):
    # tokens: (N,H,W,D) o (H,W,D) o (H,W)
    if tokens.dim() == 4:
        t = tokens[0]
    else:
        t = tokens

    if t.dim() == 3:
        z = t.shape[-1] // 2
        img = t[:, :, z].numpy()
        title = f"{name} | slice z={z}"
    elif t.dim() == 2:
        img = t.numpy()
        title = f"{name} | 2D"
    else:
        print(f"[WARN] {name}: I can't slice with dim={t.dim()}")
        return

    out_path = os.path.join(OUT_DIR, f"{name}_slice.png")
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)

def plot_hist(name: str, tokens: torch.Tensor):
    flat = tokens.reshape(-1)

    vals, counts = torch.unique(flat, return_counts=True)
    order = torch.argsort(vals)
    vals = vals[order]
    counts = counts[order]

    out_path = os.path.join(OUT_DIR, f"{name}_hist.png")
    plt.figure(figsize=(12, 4))
    plt.bar(vals.numpy(), counts.numpy())
    plt.title(f"{name} | token counts")
    plt.xlabel("code id")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)

for fname in FILES:
    path = os.path.join(BASE_DIR, fname)
    name = os.path.splitext(fname)[0]  # tokens_train, tokens_val, tokens_test

    if not os.path.exists(path):
        print(f"[SKIP] Does not exist: {path}")
        continue

    tokens = load_tokens(path)

    if not isinstance(tokens, torch.Tensor):
        if isinstance(tokens, dict) and "tokens" in tokens and isinstance(tokens["tokens"], torch.Tensor):
            tokens = tokens["tokens"]
        else:
            print(f"[WARN] {name}: non-tensor object; adjusts parsing based on content.")
            continue

    summarize_tokens(name, tokens)
    plot_slice(name, tokens)
    plot_hist(name, tokens)

print(f"\nReady. Figures in: {OUT_DIR}")