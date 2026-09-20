from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path("/workspace")
TRAIN_CSV = PROJECT_ROOT / "data/raw/train.csv"
OUTPUT = PROJECT_ROOT / "data/reports/baseline_001_split.csv"

SEED = 42
VAL_FRACTION = 0.20

df = pd.read_csv(TRAIN_CSV)

if "StudyInstanceUID" not in df.columns:
    raise ValueError("train.csv missing StudyInstanceUID")

study_uids = df["StudyInstanceUID"].astype(str)

if study_uids.isna().any():
    raise ValueError("StudyInstanceUID contains NaN")

if study_uids.duplicated().any():
    raise ValueError(
        f"train.csv contains duplicated StudyInstanceUID rows: "
        f"{study_uids.duplicated().sum()}"
    )

study_uids = study_uids.sort_values().to_numpy()

train_uids, val_uids = train_test_split(
    study_uids,
    test_size=VAL_FRACTION,
    random_state=SEED,
    shuffle=True,
)

split = pd.DataFrame({
    "StudyInstanceUID": list(train_uids) + list(val_uids),
    "split": (
        ["train"] * len(train_uids)
        + ["val"] * len(val_uids)
    ),
})

split["StudyInstanceUID"] = split["StudyInstanceUID"].astype(str)

# Deterministic output ordering.
split = split.sort_values("StudyInstanceUID").reset_index(drop=True)

# -------------------------
# Validation
# -------------------------
if len(split) != len(study_uids):
    raise RuntimeError("Split row count mismatch")

if split["StudyInstanceUID"].nunique() != len(study_uids):
    raise RuntimeError("Duplicate StudyInstanceUID in split")

if set(split.loc[split["split"] == "train", "StudyInstanceUID"]) & \
   set(split.loc[split["split"] == "val", "StudyInstanceUID"]):
    raise RuntimeError("Train/validation overlap detected")

if set(split["StudyInstanceUID"]) != set(study_uids):
    raise RuntimeError("Split does not cover all studies")

counts = split["split"].value_counts()

expected = {"train", "val"}
if set(counts.index) != expected:
    raise RuntimeError(f"Unexpected split labels: {counts.index.tolist()}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
split.to_csv(OUTPUT, index=False)

# Re-read to verify artifact.
check = pd.read_csv(OUTPUT, dtype={"StudyInstanceUID": str})

if len(check) != len(split):
    raise RuntimeError("Written artifact row count mismatch")

if check["StudyInstanceUID"].nunique() != len(check):
    raise RuntimeError("Written artifact contains duplicate studies")

print("=" * 70)
print("BASELINE-001 DETERMINISTIC STUDY SPLIT")
print("=" * 70)
print(f"Seed                    : {SEED}")
print(f"Validation fraction     : {VAL_FRACTION}")
print(f"Total studies           : {len(check):,}")
print(f"Train studies           : {(check['split'] == 'train').sum():,}")
print(f"Validation studies      : {(check['split'] == 'val').sum():,}")
print()
print("OVERLAP CHECK            : PASS")
print("FULL STUDY COVERAGE      : PASS")
print("UNIQUE STUDIES           : PASS")
print()
print(f"ARTIFACT WRITTEN: {OUTPUT}")
print()
print("STATUS: PASS")
