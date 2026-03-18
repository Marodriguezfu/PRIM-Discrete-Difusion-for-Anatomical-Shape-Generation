"""Train a discrete diffusion prior over token grids.

The model learns to denoise corrupted token volumes produced by the PRIM
tokenizer. Training uses cross-entropy between predicted logits over the
codebook and the original token ids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from token_diffusion import (
    TokenDenoiser3D,
    cosine_alpha_bar,
    q_sample_uniform_replace,
    sample_tokens,
)


class TokenGridDataset(Dataset):
    """Dataset that loads a token tensor of shape [N, H, W, D]."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        loaded = torch.load(self.path, map_location="cpu", weights_only=False)
        self.tokens = self._extract_tensor(loaded).long()

        if self.tokens.dim() != 4:
            raise ValueError(
                f"Expected a 4D token tensor [N, H, W, D], but got shape {tuple(self.tokens.shape)} "
                f"from {self.path}."
            )

    @staticmethod
    def _extract_tensor(obj: Any) -> torch.Tensor:
        """Support either a raw tensor or a dict with a 'tokens' entry."""
        if isinstance(obj, torch.Tensor):
            return obj

        if isinstance(obj, dict):
            tokens = obj.get("tokens")
            if isinstance(tokens, torch.Tensor):
                return tokens

        raise TypeError(
            "Unsupported token file format. Expected a tensor or a dict containing a tensor under key 'tokens'."
        )

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tokens[index]


def compute_denoising_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    alpha_bar: torch.Tensor,
    num_codes: int,
) -> torch.Tensor:
    """Corrupt token volumes at time t and compute the denoising cross-entropy loss."""
    x_t = q_sample_uniform_replace(x0, t, alpha_bar, num_codes)
    logits = model(x_t, t)  # [B, K, H, W, D]

    return F.cross_entropy(
        logits.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_codes),
        x0.view(-1),
    )


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    num_steps: int,
    num_codes: int,
    alpha_bar: torch.Tensor,
    device: str,
) -> float:
    """Estimate the average validation loss over one dataloader pass."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x0 in dataloader:
        x0 = x0.to(device, non_blocking=True)  # [B, H, W, D]
        batch_size = x0.shape[0]
        t = torch.randint(1, num_steps + 1, (batch_size,), device=device)

        loss = compute_denoising_loss(
            model,
            x0,
            t,
            alpha_bar=alpha_bar,
            num_codes=num_codes,
        )
        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_sample_tokens(
    model: torch.nn.Module,
    *,
    out_path: Path,
    num_steps: int,
    num_codes: int,
    shape: tuple[int, int, int, int],
    device: str,
    alpha_bar: torch.Tensor,
    allowed: torch.Tensor,
) -> None:
    """Sample token volumes from the current model state and save them to disk."""
    model.eval()
    with torch.no_grad():
        samples = sample_tokens(
            model,
            T=num_steps,
            K=num_codes,
            shape=shape,
            device=device,
            alpha_bar=alpha_bar,
            allowed=allowed,
            logit_bias=None,
            temp_hi=1.25,
            temp_lo=1.05,
            temp_split=0.6,
        )
        torch.save(samples.detach().cpu(), out_path)


def main() -> None:
    """Train the token prior and periodically export sampled token volumes."""
    root = Path("./data/Prostate/outputprostatefinal")
    train_path = root / "tokens_train.pt"
    val_path = root / "tokens_val.pt"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training tokens: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation tokens: {val_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # This must match the tokenizer codebook size.
    num_codes = 32
    # Number of diffusion steps used during training and sampling.
    num_steps = 200

    batch_size = 8
    learning_rate = 2e-4
    epochs = 30
    patience = 10
    min_delta = 1e-4
    num_workers = 2
    pin_memory = device == "cuda"

    train_dataset = TokenGridDataset(train_path)
    val_dataset = TokenGridDataset(val_path)

    train_full = train_dataset.tokens
    if train_full.min().item() < 0 or train_full.max().item() >= num_codes:
        raise ValueError(
            f"Token ids must lie in [0, {num_codes - 1}], "
            f"but found min={int(train_full.min())}, max={int(train_full.max())}."
        )

    counts = torch.bincount(train_full.view(-1), minlength=num_codes)
    background_token = int(torch.argmax(counts).item())
    background_frequency = float(counts[background_token].float() / counts.sum().float())
    allowed_tokens = torch.nonzero(counts > 0, as_tuple=False).squeeze(1).to(device)

    print(f"TRAIN: unique tokens = {int((counts > 0).sum())} / {num_codes}")
    print(f"bg_tok = {background_token} freq = {background_frequency:.6f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    height, width, depth = (int(v) for v in train_dataset[0].shape)

    model = TokenDenoiser3D(K=num_codes, T=num_steps, d_model=256, n_blocks=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    alpha_bar = cosine_alpha_bar(num_steps, device=device)

    best_val = float("inf")
    epochs_without_improvement = 0
    best_path = root / "token_prior_best.pt"
    last_path = root / "token_prior_last.pt"

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for x0 in train_loader:
            x0 = x0.to(device, non_blocking=True)  # [B, H, W, D]
            batch_size_curr = x0.shape[0]
            t = torch.randint(1, num_steps + 1, (batch_size_curr,), device=device)

            loss = compute_denoising_loss(
                model,
                x0,
                t,
                alpha_bar=alpha_bar,
                num_codes=num_codes,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += float(loss.item())
            num_batches += 1

        train_loss = total_train_loss / max(num_batches, 1)
        val_loss = evaluate_loss(
            model,
            val_loader,
            num_steps=num_steps,
            num_codes=num_codes,
            alpha_bar=alpha_bar,
            device=device,
        )

        print(f"[ep {epoch:03d}] loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping: no improvement for {patience} epochs. "
                    f"best_val={best_val:.4f}"
                )
                break

        if (epoch + 1) % 10 == 0:
            out_path = root / f"samples_tokens_ep{epoch:03d}.pt"
            save_sample_tokens(
                model,
                out_path=out_path,
                num_steps=num_steps,
                num_codes=num_codes,
                shape=(2, height, width, depth),
                device=device,
                alpha_bar=alpha_bar,
                allowed=allowed_tokens,
            )
            print("Saved sample tokens:", out_path)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    else:
        print("[WARN] Best checkpoint was not created. Sampling from the current model state.")

    save_sample_tokens(
        model,
        out_path=root / "samples_tokens_best.pt",
        num_steps=num_steps,
        num_codes=num_codes,
        shape=(8, height, width, depth),
        device=device,
        alpha_bar=alpha_bar,
        allowed=allowed_tokens,
    )
    print("Saved best sample tokens:", root / "samples_tokens_best.pt")

    torch.save(model.state_dict(), last_path)
    print("Saved last model checkpoint:", last_path)


if __name__ == "__main__":
    main()
