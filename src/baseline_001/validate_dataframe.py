from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("/workspace")

TRAIN_CSV = PROJECT_ROOT / "data/raw/train.csv"
ROUTING_CSV = PROJECT_ROOT / "data/reports/baseline_001_series_selection.csv"
SPLIT_CSV = PROJECT_ROOT / "data/reports/baseline_001_split.csv"

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
routing = pd.read_csv(ROUTING_CSV, dtype=str)
split = pd.read_csv(SPLIT_CSV, dtype=str)

# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------
required_train = {"StudyInstanceUID", *TARGETS}
required_routing = {
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "zarr_path",
    "Anatomical_Plane",
}
required_split = {"StudyInstanceUID", "split"}

missing = required_train - set(train.columns)
if missing:
    raise RuntimeError(f"train.csv missing columns: {sorted(missing)}")

missing = required_routing - set(routing.columns)
if missing:
    raise RuntimeError(
        f"routing artifact missing columns: {sorted(missing)}"
    )

missing = required_split - set(split.columns)
if missing:
    raise RuntimeError(
        f"split artifact missing columns: {sorted(missing)}"
    )

# ------------------------------------------------------------
# Uniqueness
# ------------------------------------------------------------
if train["StudyInstanceUID"].duplicated().any():
    raise RuntimeError("Duplicate StudyInstanceUID in train.csv")

if routing["StudyInstanceUID"].duplicated().any():
    raise RuntimeError("Duplicate StudyInstanceUID in routing artifact")

if split["StudyInstanceUID"].duplicated().any():
    raise RuntimeError("Duplicate StudyInstanceUID in split artifact")

# ------------------------------------------------------------
# Merge labels + routing
# ------------------------------------------------------------
df = train[["StudyInstanceUID", *TARGETS]].merge(
    routing[
        [
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "zarr_path",
            "Anatomical_Plane",
            "selection_rule",
            "zarr_source",
        ]
    ],
    on="StudyInstanceUID",
    how="inner",
    validate="one_to_one",
)

if len(df) != len(train):
    raise RuntimeError(
        f"Routing merge lost studies: {len(df)} / {len(train)}"
    )

# ------------------------------------------------------------
# Merge deterministic split
# ------------------------------------------------------------
df = df.merge(
    split[["StudyInstanceUID", "split"]],
    on="StudyInstanceUID",
    how="inner",
    validate="one_to_one",
)

if len(df) != len(train):
    raise RuntimeError(
        f"Split merge lost studies: {len(df)} / {len(train)}"
    )

# ------------------------------------------------------------
# Study coverage
# ------------------------------------------------------------
if len(df) != 4407:
    raise RuntimeError(f"Expected 4407 studies, got {len(df)}")

if df["StudyInstanceUID"].nunique() != 4407:
    raise RuntimeError("StudyInstanceUID is not unique")

# ------------------------------------------------------------
# Routing checks
# ------------------------------------------------------------
if not (df["Anatomical_Plane"] == "Sagittal").all():
    raise RuntimeError("Non-sagittal series found")

if df["zarr_path"].isna().any():
    raise RuntimeError("Missing zarr_path")

if (df["zarr_path"].str.len() == 0).any():
    raise RuntimeError("Empty zarr_path")

# ------------------------------------------------------------
# Split checks
# ------------------------------------------------------------
split_values = set(df["split"])

if split_values != {"train", "val"}:
    raise RuntimeError(f"Unexpected split values: {split_values}")

train_uids = set(df.loc[df["split"] == "train", "StudyInstanceUID"])
val_uids = set(df.loc[df["split"] == "val", "StudyInstanceUID"])

if train_uids & val_uids:
    raise RuntimeError("TRAIN/VAL STUDY LEAKAGE DETECTED")

if len(train_uids) != 3525:
    raise RuntimeError(f"Expected 3525 train studies, got {len(train_uids)}")

if len(val_uids) != 882:
    raise RuntimeError(f"Expected 882 val studies, got {len(val_uids)}")

# ------------------------------------------------------------
# Target checks
# ------------------------------------------------------------
for target in TARGETS:
    values = pd.to_numeric(df[target], errors="coerce")

    if values.isna().any():
        raise RuntimeError(f"{target}: NaN/non-numeric target found")

    if not np.isfinite(values.to_numpy()).all():
        raise RuntimeError(f"{target}: non-finite target found")

    unique = set(values.unique())

    if not unique.issubset({0.0, 1.0}):
        raise RuntimeError(
            f"{target}: expected binary labels, got {sorted(unique)}"
        )

# ------------------------------------------------------------
# Save final merged artifact
# ------------------------------------------------------------
output = PROJECT_ROOT / "data/reports/baseline_001_dataframe.csv"

df = df.sort_values("StudyInstanceUID").reset_index(drop=True)
df.to_csv(output, index=False)

# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------
print("=" * 70)
print("BASELINE-001 DATAFRAME VALIDATION")
print("=" * 70)

print(f"Studies                 : {len(df):,}")
print(f"Unique studies          : {df['StudyInstanceUID'].nunique():,}")
print(f"Train studies           : {(df['split'] == 'train').sum():,}")
print(f"Validation studies      : {(df['split'] == 'val').sum():,}")
print(f"Sagittal volumes        : {(df['Anatomical_Plane'] == 'Sagittal').sum():,}")
print(f"Missing Zarr paths      : {df['zarr_path'].isna().sum():,}")
print(f"Targets                 : {len(TARGETS)}")

print()
print("ZARR SOURCE")
print(df["zarr_source"].value_counts().to_string())

print()
print("TARGET POSITIVE COUNTS")
for target in TARGETS:
    print(f"{target:20s}: {int(df[target].sum()):5d}")

print()
print("CHECKS")
print("Study uniqueness       : PASS")
print("Routing coverage       : PASS")
print("Sagittal-only routing  : PASS")
print("Zarr path coverage     : PASS")
print("Train/val separation   : PASS")
print("Target validity        : PASS")

print()
print(f"ARTIFACT WRITTEN: {output}")
print()
print("STATUS: PASS")
