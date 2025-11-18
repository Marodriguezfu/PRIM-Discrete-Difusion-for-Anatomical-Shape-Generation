import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Utility to load TensorBoard JSON ----------
def load_tb_json(path):
    """
    Load a JSON file exported from TensorBoard.
    Each entry has format [timestamp, step, value].
    Returns:
        steps: np.array
        values: np.array (float, with 'Infinity' as np.nan)
    """
    with open(path, "r") as f:
        data = json.load(f)

    steps = []
    values = []
    for ts, step, val in data:
        steps.append(step)
        if isinstance(val, str) and val == "Infinity":
            values.append(np.nan)
        else:
            values.append(float(val))

    return np.array(steps), np.array(values)


# ---------- File paths ----------
# Adjust this if the files are in another folder
base_dir = Path(".")
train_loss_file = base_dir / "train_loss.json"
val_loss_file   = base_dir / "val_loss.json"
val_dice_file   = base_dir / "val_dice.json"
val_hd_file     = base_dir / "val_hd.json"
val_asd_file    = base_dir / "val_asd.json"
val_multi_file  = base_dir / "val_multi.json"

# ---------- Load data ----------
steps_train, train_loss = load_tb_json(train_loss_file)
steps_val,   val_loss   = load_tb_json(val_loss_file)
_, val_dice  = load_tb_json(val_dice_file)
_, val_hd    = load_tb_json(val_hd_file)
_, val_asd   = load_tb_json(val_asd_file)
_, val_multi = load_tb_json(val_multi_file)


x=np.linspace(1,100,20)
# ---------- 1. Train vs Val Loss ----------
plt.figure(figsize=(7, 4))
plt.plot(train_loss[:-1], label="Train loss")
plt.plot(val_loss[:-1], label="Train loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_train_val.eps", format="eps")
plt.savefig("loss_train_val.png", format="png")
plt.close()

# ---------- 2. Dice metric ----------
plt.figure(figsize=(7, 4))
plt.plot(val_dice[:-1])
plt.xlabel("Epoch")
plt.ylabel("Dice score")
plt.title("Validation Dice")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("val_dice.eps", format="eps")
plt.savefig("val_dice.png", format="png")
plt.close()

# ---------- 3. Hausdorff Distance (HD) ----------
plt.figure(figsize=(7, 4))
plt.plot(val_hd[:-1])
plt.xlabel("Epoch")
plt.ylabel("HD (mm)")
plt.title("Validation Hausdorff Distance (HD)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("val_hd.eps", format="eps")
plt.savefig("val_hd.png", format="png")
plt.close()

# ---------- 4. ASD ----------
plt.figure(figsize=(7, 4))
plt.plot(val_asd[:-1])
plt.xlabel("Epoch")
plt.ylabel("ASD (mm)")
plt.title("Validation ASD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("val_asd.eps", format="eps")
plt.savefig("val_asd.png", format="png")
plt.close()

# ---------- 5. Combined metric (multi-score) ----------
plt.figure(figsize=(7, 4))
plt.plot(val_multi[:-1])
plt.xlabel("Epoch")
plt.ylabel("Multi-score")
plt.title("Validation Multi-score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("val_multi.eps", format="eps")
plt.savefig("val_multi.png", format="png")
plt.close()

# ---------- 6. All validation metrics together ----------
fig, ax1 = plt.subplots(figsize=(8, 5))

# Axis 1: overlap metrics (Dice, multi)
lns1 = ax1.plot(val_dice[:-1], label="Dice")
lns2 = ax1.plot(val_multi[:-1], label="Multi-score")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Dice / Multi-score")

# Axis 2: distances (HD, ASD)
ax2 = ax1.twinx()
lns3 = ax2.plot(val_hd[:-1], linestyle="--", label="HD")
lns4 = ax2.plot(val_asd[:-1], linestyle="--", label="ASD")
ax2.set_ylabel("HD / ASD (mm)")

# Combine legends
lns = lns1 + lns2 + lns3 + lns4
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc="best")

ax1.set_title("Validation metrics evolution")
ax1.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("val_all_metrics.eps", format="eps")
plt.savefig("val_all_metrics.png", format="png")
plt.close()

print("Done eps figures saved in the current directory:")
print("  - loss_train_val.eps")
print("  - val_dice.eps")
print("  - val_hd.eps")
print("  - val_asd.eps")
print("  - val_multi.eps")
print("  - val_all_metrics.eps")

print("  - loss_train_val.png")
print("  - val_dice.png")
print("  - val_hd.png")
print("  - val_asd.png")
print("  - val_multi.png")
print("  - val_all_metrics.png")
