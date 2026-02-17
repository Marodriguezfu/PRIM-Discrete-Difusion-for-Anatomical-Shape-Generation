import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = "./outputprostatefinal/tokens_test.pt"   # or tokens_train.pt, etc.
tokens = torch.load(path, map_location="cpu")

print(type(tokens))
print(tokens.shape)            # expected: (B, Ht, Wt, Dt) or (Ht, Wt, Dt)
print(tokens.dtype)
print(tokens.min().item(), tokens.max().item())
print(torch.unique(tokens).numel())  # how many codes appear in this file




tokens = torch.load(path, map_location="cpu")  # (N,H,W,D) o (H,W,D)

# Si es (N,H,W,D), toma un ejemplo
if tokens.dim() == 4:
    t = tokens[0]           # (H,W,D)
else:
    t = tokens

z = t.shape[-1] // 2        # slice central
plt.figure()
plt.imshow(t[:, :, z].numpy())
plt.title(f"Tokens slice z={z}")
plt.axis("off")
plt.tight_layout()
plt.savefig("tokens_slice.png", dpi=200)
plt.close()
print("Saved: tokens_slice.png")
 
flat = tokens.reshape(-1)

vals, counts = torch.unique(flat, return_counts=True)
plt.figure()
plt.bar(vals.numpy(), counts.numpy())
plt.title("Token counts")
plt.xlabel("code id")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("tokens_hist.png", dpi=200)
plt.close()
print("Saved: tokens_hist.png")