import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from token_diffusion import (
    cosine_alpha_bar,
    q_sample_uniform_replace,
    TokenDenoiser3D,
    sample_tokens,
)

class TokenGridDataset(Dataset):
    def __init__(self, path: str):
        self.tokens = torch.load(path).long()  # [N,H,W,D]

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return self.tokens[i]

@torch.no_grad()
def eval_loss(model, dl, T, K, alpha_bar, device):
    model.eval()
    tot, n = 0.0, 0
    for x0 in dl:
        x0 = x0.to(device)          # [B,H,W,D]
        B = x0.shape[0]
        t = torch.randint(1, T + 1, (B,), device=device)
        x_t = q_sample_uniform_replace(x0, t, alpha_bar, K)
        logits = model(x_t, t)       # [B,K,H,W,D]
        loss = F.cross_entropy(
            logits.permute(0, 2, 3, 4, 1).contiguous().view(-1, K),
            x0.view(-1),
        )
        tot += float(loss.item())
        n += 1
    return tot / max(n, 1)

def main():
    root = "./data/Prostate/outputprostatefinal"
    train_path = os.path.join(root, "tokens_train.pt")
    val_path   = os.path.join(root, "tokens_val.pt")

    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(val_path),   f"Missing {val_path}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Must match your VQ codebook size
    K = 32
    # Diffusion steps
    T = 200

    bs = 8
    lr = 2e-4
    epochs = 200
    patience = 10
    min_delta = 1e-4

    # --- token stats (train) ---
    train_full = torch.load(train_path).long()               # [N,H,W,D]
    counts = torch.bincount(train_full.view(-1), minlength=K)
    bg_tok = int(torch.argmax(counts).item())
    bg_freq = float(counts[bg_tok].float() / counts.sum().float())
    allowed = torch.nonzero(counts > 0, as_tuple=False).squeeze(1).to(device)

    print(f"TRAIN: unique tokens = {int((counts>0).sum())} / {K}")
    print(f"bg_tok = {bg_tok} freq = {bg_freq:.6f}")

    # Datasets / loaders
    ds_tr = TokenGridDataset(train_path)
    ds_va = TokenGridDataset(val_path)

    # If Colab gives issues with workers, set num_workers=0
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    H, W, D = ds_tr[0].shape

    model = TokenDenoiser3D(K=K, T=T, d_model=256, n_blocks=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    alpha_bar = cosine_alpha_bar(T, device=device)

    best_val = float("inf")
    bad = 0
    best_path = os.path.join(root, "token_prior_best.pt")

    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0

        for x0 in dl_tr:
            x0 = x0.to(device)      # [B,H,W,D]
            B = x0.shape[0]
            t = torch.randint(1, T + 1, (B,), device=device)
            x_t = q_sample_uniform_replace(x0, t, alpha_bar, K)
            logits = model(x_t, t)  # [B,K,H,W,D]

            loss = F.cross_entropy(
                logits.permute(0, 2, 3, 4, 1).contiguous().view(-1, K),
                x0.view(-1),
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss.item())
            n += 1

        tr_loss = tot / max(n, 1)
        va_loss = eval_loss(model, dl_va, T=T, K=K, alpha_bar=alpha_bar, device=device)

        print(f"[ep {ep:03d}] loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

        # --- save BEST + early stop ---
        if va_loss < best_val - min_delta:
            best_val = va_loss
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop: no mejora en {patience} epochs. best_val={best_val:.4f}")
                break

        # --- optional: periodic samples from CURRENT model ---
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                samp = sample_tokens(
                    model,
                    T=T, K=K,
                    shape=(2, H, W, D),
                    device=device,
                    alpha_bar=alpha_bar,
                    allowed=allowed,
                    logit_bias=None,
                    temp_hi=1.25,
                    temp_lo=1.05,
                    temp_split=0.6,
                )
                out = os.path.join(root, f"samples_tokens_ep{ep:03d}.pt")
                torch.save(samp.detach().cpu(), out)
                print("saved samples tokens")

    # --- After training: sample using BEST checkpoint ---
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    with torch.no_grad():
        samp = sample_tokens(
            model,
            T=T, K=K,
            shape=(8, H, W, D),
            device=device,
            alpha_bar=alpha_bar,
            allowed=allowed,
            logit_bias=None,
            temp_hi=1.25,
            temp_lo=1.05,
            temp_split=0.6,
        )
        out = os.path.join(root, "samples_tokens_best.pt")
        torch.save(samp.detach().cpu(), out)
        print("Saved best samples tokens:", out)

    torch.save(model.state_dict(), os.path.join(root, "token_prior_last.pt"))

if __name__ == "__main__":
    main()
