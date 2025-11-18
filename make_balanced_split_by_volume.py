#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create balanced splits (70/15/15) stratified by TZ and PZ volume.
- Reads one or several CSVs (existing train/val/test or a master file);
- Computes TZ(=1) and PZ(=2) volumes;
- Generates mask_volumes.csv (per-row report);
- Splits by patient while balancing TZ/PZ bins;
- Writes new train.csv / validation.csv / test.csv.
"""

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd

# Try to support both NIfTI and NRRD
_HAS_NIB = True
try:
    import nibabel as nib
except Exception:
    _HAS_NIB = False

_HAS_SITK = True
try:
    import SimpleITK as sitk
except Exception:
    _HAS_SITK = False


def infer_patient_id(path: str) -> str:
    """
    Extract a stable patient_id from the file path.
    Adjust the regex to your folder structure.
    Tries to capture /<center>/<CaseXX>.nii.gz

    Examples:
      .../Prostate/BMC/Case12.nii.gz  --> BMC_Case12
      .../Prostate/RUNMC/Case03.nii   --> RUNMC_Case03
    Fallback: <parentdir>_<stem>
    """
    m = re.search(r"/(BMC|RUNMC)/(?P<case>Case\d+)\.", path, flags=re.IGNORECASE)
    if m:
        center = m.group(1)
        case = m.group("case")
        return f"{center}_{case}"
    # Generic fallback
    parent = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    # Handle .nii.gz
    if stem.endswith(".nii"):
        stem = stem[:-4]
    return f"{parent}_{stem}"


def load_mask(path: str) -> (np.ndarray, tuple):
    """
    Load the mask and return (array, spacing_mm).
    spacing_mm = (sx, sy, sz) or (1,1,1) if not available.
    Supports NIfTI (nibabel) and NRRD (SimpleITK).
    """
    ext = path.lower()
    if (ext.endswith(".nii") or ext.endswith(".nii.gz")) and _HAS_NIB:
        img = nib.load(path)
        data = img.get_fdata()
        # Ensure 3D
        if data.ndim == 4:
            # If last dim is 1, squeeze; otherwise, take channel 0
            data = data[..., 0]
        zooms = img.header.get_zooms()
        if len(zooms) >= 3:
            spacing = tuple(float(z) for z in zooms[:3])
        else:
            spacing = (1.0, 1.0, 1.0)
        return np.asarray(data), spacing

    if ext.endswith(".nrrd") and _HAS_SITK:
        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img)  # (z,y,x)
        # Convert to (y,x,z)
        data = np.moveaxis(data, 0, -1)
        spacing = img.GetSpacing()  # (sx, sy, sz) in SimpleITK == (x,y,z)
        # SimpleITK spacing is (x,y,z); our final array is (y,x,z) -> reorder
        spacing = (spacing[1], spacing[0], spacing[2])
        return np.asarray(data), spacing

    raise RuntimeError(
        f"Could not load mask (unsupported extension or missing nibabel/SimpleITK): {path}"
    )


def compute_volumes(mask: np.ndarray, spacing: tuple, tz_label=1, pz_label=2):
    """
    Returns:
    - voxel counts for TZ and PZ
    - volumes in mm3 if spacing>0; otherwise, returns mm3=voxel_count
    """
    mask = np.asarray(mask)
    # Ensure integer dtype to compare labels
    if not np.issubdtype(mask.dtype, np.integer):
        mask = np.rint(mask).astype(np.int32)

    tz_vox = int(np.sum(mask == tz_label))
    pz_vox = int(np.sum(mask == pz_label))

    sx, sy, sz = spacing if spacing and len(spacing) == 3 else (1.0, 1.0, 1.0)
    voxel_mm3 = float(sx) * float(sy) * float(sz)
    if voxel_mm3 <= 0:
        voxel_mm3 = 1.0  # fallback

    tz_mm3 = tz_vox * voxel_mm3
    pz_mm3 = pz_vox * voxel_mm3
    return tz_vox, pz_vox, tz_mm3, pz_mm3, voxel_mm3


def read_inputs(csv_paths):
    dfs = []
    for p in csv_paths:
        if not os.path.isfile(p):
            warnings.warn(f"[WARN] CSV not found: {p}")
            continue
        df = pd.read_csv(p)
        if not {"images", "labels"}.issubset(df.columns):
            raise ValueError(f"CSV {p} must have columns: images,labels")
        dfs.append(df)
    if not dfs:
        raise ValueError("Could not read any input CSV.")
    all_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return all_df


def make_bins(series, n_bins=3):
    """
    Returns bins with labels {0,1,2} (low/medium/high) using quantiles.
    Handles constant series: if all equal, returns all 1 (medium).
    """
    s = series.fillna(0).values
    if np.all(s == s[0]):
        return pd.Series(np.zeros_like(s, dtype=int) + 1, index=series.index)  # all medium
    qs = np.quantile(s, [0.33, 0.66])
    bins = np.zeros_like(s, dtype=int)
    bins[s > qs[0]] = 1
    bins[s > qs[1]] = 2
    return pd.Series(bins, index=series.index)


def balanced_group_split(df_stats, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Balanced split by TZ/PZ bins and grouped by patient_id.

    Strategy:
      - stratum = (tz_bin, pz_bin)
      - group by patient_id and take the mode of the stratum per patient
      - distribute patients per stratum in round-robin across train/val/test
    """
    rng = np.random.default_rng(random_state)

    # stratum label per row (study)
    df_stats["tz_bin"] = make_bins(df_stats["tz_mm3"])
    df_stats["pz_bin"] = make_bins(df_stats["pz_mm3"])
    df_stats["stratum"] = df_stats["tz_bin"].astype(str) + "-" + df_stats["pz_bin"].astype(str)

    # stratum per patient = mode
    patient_groups = df_stats.groupby("patient_id")
    patient_stratum = patient_groups["stratum"].agg(
        lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]
    )

    patients = patient_stratum.index.to_numpy()
    strata = patient_stratum.values

    uniq = np.unique(strata)
    # >>> use lists (not arrays) <<<
    buckets = {u: list(patients[strata == u]) for u in uniq}
    for u in uniq:
        rng.shuffle(buckets[u])

    # target sizes
    n_pat = len(patients)
    n_train = int(round(n_pat * train_ratio))
    n_val = int(round(n_pat * val_ratio))
    n_test = n_pat - n_train - n_val  # ensure exact sum

    train_p, val_p, test_p = [], [], []
    order = ["train", "val", "test"]
    idx_order = 0

    # round-robin assignment per stratum
    def add_patient(dest, pid):
        if dest == "train" and len(train_p) < n_train:
            train_p.append(pid)
            return True
        if dest == "val" and len(val_p) < n_val:
            val_p.append(pid)
            return True
        if dest == "test" and len(test_p) < n_test:
            test_p.append(pid)
            return True
        return False

    while any(len(buckets[u]) > 0 for u in uniq):
        for u in uniq:
            if not buckets[u]:
                continue
            pid = buckets[u].pop()

            # try current destination; if full, try the others
            placed = False
            for k in range(3):
                dest = order[(idx_order + k) % 3]
                if add_patient(dest, pid):
                    placed = True
                    break
            if not placed:
                # if all full due to rounding, put in train
                train_p.append(pid)

            idx_order += 1

        if len(train_p) >= n_train and len(val_p) >= n_val and len(test_p) >= n_test:
            break

    # if there are still patients left, send them to train
    for u in uniq:
        while buckets[u]:
            train_p.append(buckets[u].pop())

    return set(train_p), set(val_p), set(test_p)


def main():
    ap = argparse.ArgumentParser(description="Create balanced splits by TZ/PZ volume.")
    ap.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or several CSVs (e.g., train.csv validation.csv test.csv or a master file).",
    )
    ap.add_argument(
        "--out_dir",
        default=".",
        help="Output folder for mask_volumes.csv and new CSVs.",
    )
    ap.add_argument(
        "--tz_label", type=int, default=1, help="TZ label in the mask (default=1)."
    )
    ap.add_argument(
        "--pz_label", type=int, default=2, help="PZ label in the mask (default=2)."
    )
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = read_inputs(args.csv)
    # Compute per-row stats
    rows = []
    for i, r in df.iterrows():
        img_p = r["images"]
        msk_p = r["labels"]
        try:
            mask, spacing = load_mask(msk_p)
        except Exception as e:
            warnings.warn(f"[WARN] Could not read mask: {msk_p} ({e})")
            continue

        tz_vox, pz_vox, tz_mm3, pz_mm3, voxel_mm3 = compute_volumes(
            mask, spacing, tz_label=args.tz_label, pz_label=args.pz_label
        )
        pid = infer_patient_id(msk_p)
        rows.append(
            {
                "patient_id": pid,
                "images": img_p,
                "labels": msk_p,
                "tz_vox": tz_vox,
                "pz_vox": pz_vox,
                "tz_mm3": tz_mm3,
                "pz_mm3": pz_mm3,
                "voxel_mm3": voxel_mm3,
            }
        )

    if not rows:
        raise RuntimeError("No rows were generated. Are paths/formats correct?")

    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(args.out_dir, "mask_volumes.csv"), index=False)

    # Balanced split by TZ/PZ volume bins and grouped by patient
    train_p, val_p, test_p = balanced_group_split(
        stats.copy(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    # Build new CSVs (same columns expected by your dataloaders)
    train_df = stats[stats["patient_id"].isin(train_p)][["images", "labels"]].drop_duplicates().reset_index(drop=True)
    val_df   = stats[stats["patient_id"].isin(val_p)][["images", "labels"]].drop_duplicates().reset_index(drop=True)
    test_df  = stats[stats["patient_id"].isin(test_p)][["images", "labels"]].drop_duplicates().reset_index(drop=True)

    # Save with index so that the first column is the row number (unnamed)
    train_csv = os.path.join(args.out_dir, "train.csv")
    val_csv   = os.path.join(args.out_dir, "validation.csv")
    test_csv  = os.path.join(args.out_dir, "test.csv")

    train_df.to_csv(train_csv, index=True)
    val_df.to_csv(val_csv, index=True)
    test_df.to_csv(test_csv, index=True)


    # Summary
    print(">>> Saved volume report to:", os.path.join(args.out_dir, "mask_volumes.csv"))
    print(f">>> Split created in {args.out_dir}/")
    print(f"    train.csv: {len(train_df)} rows, patients: {len(train_p)}")
    print(f"    validation.csv: {len(val_df)} rows, patients: {len(val_p)}")
    print(f"    test.csv: {len(test_df)} rows, patients: {len(test_p)}")


if __name__ == "__main__":
    main()
