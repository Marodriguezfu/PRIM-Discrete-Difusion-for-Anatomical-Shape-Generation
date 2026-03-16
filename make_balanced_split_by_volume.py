#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create balanced prostate dataset splits with a fixed 70/15/15 ratio.

This script has no command-line arguments. It reads the master case list from
`./data/Prostate/all_cases.csv`, computes TZ and PZ mask volumes, creates a
balanced patient-level split, and saves:

- `mask_volumes.csv`
- `train.csv`
- `validation.csv`
- `test.csv`

The output CSV format matches the existing dataloader format:
an unnamed index column plus `images` and `labels`.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Optional backends for mask loading.
_HAS_NIBABEL = True
try:
    import nibabel as nib
except Exception:
    _HAS_NIBABEL = False

_HAS_SIMPLEITK = True
try:
    import SimpleITK as sitk
except Exception:
    _HAS_SIMPLEITK = False


# -----------------------------------------------------------------------------
# Fixed configuration
# -----------------------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
TZ_LABEL = 1
PZ_LABEL = 2

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "Prostate"
INPUT_CSV = DATA_DIR / "all_cases.csv"
MASK_VOLUMES_CSV = "mask_volumes.csv"
TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"
TEST_CSV = "test.csv"


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve an image or mask path relative to the project root or CSV folder."""
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path

    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate

    csv_candidate = base_dir / path
    if csv_candidate.exists():
        return csv_candidate

    # Return the project-relative candidate for clearer error messages.
    return project_candidate



def infer_patient_id(path_str: str) -> str:
    """
    Extract a stable patient identifier from a mask path.

    Supported examples:
    - data/Prostate/masks/Case00_Segmentation.nii.gz -> Case00
    - .../BMC/Case12.nii.gz -> BMC_Case12
    - .../RUNMC/Case03.nrrd -> RUNMC_Case03
    """
    normalized = path_str.replace("\\", "/")

    center_case_match = re.search(
        r"/(BMC|RUNMC)/(?P<case>Case\d+)(?:_Segmentation)?(?:\.nii(?:\.gz)?|\.nrrd)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if center_case_match:
        center = center_case_match.group(1)
        case = center_case_match.group("case")
        return f"{center}_{case}"

    case_match = re.search(
        r"(?P<case>Case\d+)(?:_Segmentation)?(?:\.nii(?:\.gz)?|\.nrrd)$",
        normalized,
    )
    if case_match:
        return case_match.group("case")

    path = Path(path_str)
    stem = path.name
    for suffix in (".nii.gz", ".nii", ".nrrd"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = stem.replace("_Segmentation", "")
    return stem



def load_mask(mask_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Load a segmentation mask and return ``(array, spacing_mm)``.

    NIfTI files are loaded with nibabel when available.
    NRRD files are loaded with SimpleITK when available.
    """
    path_str = str(mask_path).lower()

    if path_str.endswith((".nii", ".nii.gz")) and _HAS_NIBABEL:
        image = nib.load(str(mask_path))
        data = image.get_fdata()
        if data.ndim == 4:
            data = data[..., 0]
        zooms = image.header.get_zooms()
        spacing = tuple(float(v) for v in zooms[:3]) if len(zooms) >= 3 else (1.0, 1.0, 1.0)
        return np.asarray(data), spacing

    if path_str.endswith(".nrrd") and _HAS_SIMPLEITK:
        image = sitk.ReadImage(str(mask_path))
        data = sitk.GetArrayFromImage(image)  # (z, y, x)
        data = np.moveaxis(data, 0, -1)       # (y, x, z)
        spacing_xyz = image.GetSpacing()      # (x, y, z)
        spacing = (float(spacing_xyz[1]), float(spacing_xyz[0]), float(spacing_xyz[2]))
        return np.asarray(data), spacing

    raise RuntimeError(
        f"Unable to load mask: {mask_path}. Install nibabel for NIfTI or SimpleITK for NRRD."
    )



def compute_volumes(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    tz_label: int = TZ_LABEL,
    pz_label: int = PZ_LABEL,
) -> tuple[int, int, float, float, float]:
    """Compute TZ/PZ voxel counts and physical volumes in mm³."""
    mask_array = np.asarray(mask)
    if not np.issubdtype(mask_array.dtype, np.integer):
        mask_array = np.rint(mask_array).astype(np.int32)

    tz_voxels = int(np.sum(mask_array == tz_label))
    pz_voxels = int(np.sum(mask_array == pz_label))

    sx, sy, sz = spacing if len(spacing) == 3 else (1.0, 1.0, 1.0)
    voxel_volume_mm3 = max(float(sx) * float(sy) * float(sz), 1.0)

    tz_volume_mm3 = tz_voxels * voxel_volume_mm3
    pz_volume_mm3 = pz_voxels * voxel_volume_mm3
    return tz_voxels, pz_voxels, tz_volume_mm3, pz_volume_mm3, voxel_volume_mm3



def read_master_csv(csv_path: Path) -> pd.DataFrame:
    """Read the master CSV and keep only the columns required by the pipeline."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    required_columns = {"images", "labels"}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(f"{csv_path} must contain the columns: images, labels")

    return dataframe[["images", "labels"]].drop_duplicates().reset_index(drop=True)



def make_bins(series: pd.Series) -> pd.Series:
    """Create three quantile-based bins: 0=low, 1=medium, 2=high."""
    values = series.fillna(0).to_numpy()
    if values.size == 0:
        return pd.Series(dtype=int, index=series.index)
    if np.all(values == values[0]):
        return pd.Series(np.full_like(values, 1, dtype=int), index=series.index)

    q33, q66 = np.quantile(values, [0.33, 0.66])
    bins = np.zeros_like(values, dtype=int)
    bins[values > q33] = 1
    bins[values > q66] = 2
    return pd.Series(bins, index=series.index)



def balanced_group_split(
    stats_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
) -> tuple[set[str], set[str], set[str]]:
    """
    Split patients into train/validation/test while balancing TZ/PZ volume bins.

    The split is patient-level. Each patient is assigned a stratum defined by the
    mode of its ``(tz_bin, pz_bin)`` combination. Patients are then distributed
    across train/validation/test in a round-robin fashion inside each stratum.
    """
    stats_df = stats_df.copy()
    rng = np.random.default_rng(random_state)

    stats_df["tz_bin"] = make_bins(stats_df["tz_mm3"])
    stats_df["pz_bin"] = make_bins(stats_df["pz_mm3"])
    stats_df["stratum"] = stats_df["tz_bin"].astype(str) + "-" + stats_df["pz_bin"].astype(str)

    patient_stratum = stats_df.groupby("patient_id")["stratum"].agg(
        lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]
    )

    patients = patient_stratum.index.to_numpy()
    strata = patient_stratum.to_numpy()
    unique_strata = np.unique(strata)

    buckets = {stratum: list(patients[strata == stratum]) for stratum in unique_strata}
    for stratum in unique_strata:
        rng.shuffle(buckets[stratum])

    n_patients = len(patients)
    n_train = int(round(n_patients * train_ratio))
    n_val = int(round(n_patients * val_ratio))
    n_test = n_patients - n_train - n_val

    train_patients: list[str] = []
    val_patients: list[str] = []
    test_patients: list[str] = []

    split_order = ["train", "val", "test"]
    split_cursor = 0

    def add_patient(split_name: str, patient_id: str) -> bool:
        if split_name == "train" and len(train_patients) < n_train:
            train_patients.append(patient_id)
            return True
        if split_name == "val" and len(val_patients) < n_val:
            val_patients.append(patient_id)
            return True
        if split_name == "test" and len(test_patients) < n_test:
            test_patients.append(patient_id)
            return True
        return False

    while any(buckets[stratum] for stratum in unique_strata):
        for stratum in unique_strata:
            if not buckets[stratum]:
                continue

            patient_id = buckets[stratum].pop()
            placed = False
            for offset in range(3):
                destination = split_order[(split_cursor + offset) % 3]
                if add_patient(destination, patient_id):
                    placed = True
                    break

            if not placed:
                train_patients.append(patient_id)

            split_cursor += 1

        if len(train_patients) >= n_train and len(val_patients) >= n_val and len(test_patients) >= n_test:
            break

    for stratum in unique_strata:
        while buckets[stratum]:
            train_patients.append(buckets[stratum].pop())

    return set(train_patients), set(val_patients), set(test_patients)



def build_volume_table(master_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Compute per-case mask statistics from the master CSV."""
    rows: list[dict[str, object]] = []
    csv_dir = csv_path.parent

    for row in master_df.itertuples(index=False):
        image_rel_path = row.images
        label_rel_path = row.labels
        label_abs_path = resolve_path(label_rel_path, csv_dir)

        try:
            mask, spacing = load_mask(label_abs_path)
        except Exception as exc:
            warnings.warn(f"Could not read mask: {label_rel_path} ({exc})")
            continue

        tz_voxels, pz_voxels, tz_volume_mm3, pz_volume_mm3, voxel_volume_mm3 = compute_volumes(
            mask,
            spacing,
            tz_label=TZ_LABEL,
            pz_label=PZ_LABEL,
        )

        rows.append(
            {
                "patient_id": infer_patient_id(label_rel_path),
                "images": image_rel_path,
                "labels": label_rel_path,
                "tz_vox": tz_voxels,
                "pz_vox": pz_voxels,
                "tz_mm3": tz_volume_mm3,
                "pz_mm3": pz_volume_mm3,
                "voxel_mm3": voxel_volume_mm3,
            }
        )

    if not rows:
        raise RuntimeError("No valid rows were generated. Check the CSV paths and mask files.")

    return pd.DataFrame(rows)



def save_split_csv(split_df: pd.DataFrame, output_path: Path) -> None:
    """Save a split CSV with the same format as the existing pipeline files."""
    output_df = split_df[["images", "labels"]].drop_duplicates().reset_index(drop=True)
    output_df.to_csv(output_path, index=True)



def main() -> None:
    """Run the full split generation pipeline with fixed settings."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    master_df = read_master_csv(INPUT_CSV)
    stats_df = build_volume_table(master_df, INPUT_CSV)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(MASK_VOLUMES_CSV, index=False)

    train_patients, val_patients, test_patients = balanced_group_split(stats_df)

    train_df = stats_df[stats_df["patient_id"].isin(train_patients)]
    val_df = stats_df[stats_df["patient_id"].isin(val_patients)]
    test_df = stats_df[stats_df["patient_id"].isin(test_patients)]

    save_split_csv(train_df, TRAIN_CSV)
    save_split_csv(val_df, VALIDATION_CSV)
    save_split_csv(test_df, TEST_CSV)

    print(f"Saved volume report: {MASK_VOLUMES_CSV}")
    print(f"Saved train split: {TRAIN_CSV} ({len(train_df)} rows, {len(train_patients)} patients)")
    print(f"Saved validation split: {VALIDATION_CSV} ({len(val_df)} rows, {len(val_patients)} patients)")
    print(f"Saved test split: {TEST_CSV} ({len(test_df)} rows, {len(test_patients)} patients)")


if __name__ == "__main__":
    main()
