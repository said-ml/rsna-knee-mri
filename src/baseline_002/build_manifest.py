from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================

# PATHS

# ============================================================

PROJECT_ROOT = Path("/workspace")

REPORT_LABELS = (
PROJECT_ROOT
/ "data/reports/report_001_frozen/report_labels.csv"
)

SERIES_SELECTION = (
PROJECT_ROOT
/ "data/reports/baseline_001_series_selection.csv"
)

OUTPUT_MANIFEST = (
PROJECT_ROOT
/ "data/reports/baseline_002_manifest.csv"
)

OUTPUT_AUDIT = (
PROJECT_ROOT
/ "data/reports/baseline_002_manifest_audit.txt"
)

# ============================================================

# TARGETS

# ============================================================

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

# ============================================================

# BASELINE-001 SPLIT DEFINITION

# ============================================================

# This exactly reproduces the split used by

# src/baseline_001/train.py:

#

# train_test_split(

# df,

# test_size=0.20,

# random_state=SEED,

# shuffle=True,

# )

#

# config.py:

# SEED = 42

#

# IMPORTANT:

# baseline_001_split.csv is NOT used to determine the

# final BASELINE-001 gold train/validation split.

#

# baseline_001_split.csv is a separate global 4407-study

# split and contains 48 gold train / 10 gold validation

# studies. BASELINE-001 itself re-splits the 58 gold studies

# into 46 train / 12 validation studies.

BASELINE_001_SEED = 42
BASELINE_001_TEST_SIZE = 0.20

# ============================================================

# HELPERS

# ============================================================

def require_columns(df, columns, name):
        missing = [c for c in columns if c not in df.columns]

        if missing:
            raise RuntimeError(
                f"{name} is missing required columns: {missing}"
            )


def assert_unique(df, column, name):
    duplicated = df[column].duplicated().sum()


    if duplicated:
            dup_values = (
                df.loc[df[column].duplicated(keep=False), column]
                .drop_duplicates()
                .tolist()
            )

            raise RuntimeError(
                f"{name}: {duplicated} duplicated {column} values. "
                f"Examples: {dup_values[:10]}"
            )


def validate_binary_or_nan(df, targets, name):
    for target in targets:
         values = df[target].dropna().unique()

         invalid = [
           x for x in values
               if x not in (0, 1)
             ]

         if invalid:
          raise RuntimeError(
            f"{name}: target {target!r} contains invalid "
            f"values: {invalid}"
        )


# ============================================================

# LOAD

# ============================================================

print("=" * 70)
print("BASELINE-002 MANIFEST BUILDER")
print("=" * 70)

print("\nLoading report labels...")
report = pd.read_csv(REPORT_LABELS)

print("Loading series selection...")
series = pd.read_csv(SERIES_SELECTION)

print("Loading original structured-label population...")
TRAIN_CSV = PROJECT_ROOT / "data/raw/train.csv"
train_full = pd.read_csv(TRAIN_CSV)

# ============================================================

# BASIC VALIDATION

# ============================================================

require_columns(
report,
["StudyInstanceUID"] + TARGETS,
"report_labels.csv",
)

require_columns(
series,
["StudyInstanceUID", "SeriesInstanceUID", "zarr_path"],
"baseline_001_series_selection.csv",
)

require_columns(
train_full,
["StudyInstanceUID"] + TARGETS,
"data/raw/train.csv",
)

assert_unique(
report,
"StudyInstanceUID",
"report_labels.csv",
)

assert_unique(
series,
"StudyInstanceUID",
"baseline_001_series_selection.csv",
)

assert_unique(
train_full,
"StudyInstanceUID",
"data/raw/train.csv",
)

validate_binary_or_nan(
report,
TARGETS,
"report_labels.csv",
)

# ============================================================

# DETERMINE BASELINE-001 GOLD TRAIN / VALIDATION

# ============================================================

print("\nDetermining BASELINE-001 gold split...")

# BASELINE-001 defines the gold population directly from

# train.csv: all 12 structured targets must be present.

gold_df = train_full.dropna(
subset=TARGETS
).copy()

if len(gold_df) != 58:
            raise RuntimeError(
            f"Expected exactly 58 complete-label studies, "
            f"found {len(gold_df)}"
            )

print()
print("BASELINE-001 split definition:")
print("  complete-label studies:", len(gold_df))
print("  test_size:", BASELINE_001_TEST_SIZE)
print("  random_state:", BASELINE_001_SEED)
print("  shuffle:", True)

# IMPORTANT:

# This exactly reproduces BASELINE-001 train.py.

#

# Do NOT use baseline_001_split.csv here.

#

# The order of gold_df is intentionally preserved from

# train.csv because sklearn's train_test_split operates

# on this ordered dataframe.

gold_train_df, gold_val_df = train_test_split(
gold_df,
test_size=BASELINE_001_TEST_SIZE,
random_state=BASELINE_001_SEED,
shuffle=True,
)

gold_train_df = gold_train_df.reset_index(drop=True)
gold_val_df = gold_val_df.reset_index(drop=True)

gold_train_ids = set(
gold_train_df["StudyInstanceUID"]
)

gold_val_ids = set(
gold_val_df["StudyInstanceUID"]
)

gold_ids = set(
gold_df["StudyInstanceUID"]
)

print()
print("Gold population:")
print("  complete-label:", len(gold_ids))
print("  gold train:", len(gold_train_ids))
print("  gold val:", len(gold_val_ids))

if len(gold_train_ids) != 46:
        raise RuntimeError(
        f"Expected exactly 46 gold training studies, "
        f"found {len(gold_train_ids)}"
        )

if len(gold_val_ids) != 12:
            raise RuntimeError(
            f"Expected exactly 12 gold validation studies, "
            f"found {len(gold_val_ids)}"
            )

if gold_train_ids & gold_val_ids:
            raise RuntimeError(
            "Gold train/validation overlap detected."
            )

if gold_train_ids | gold_val_ids != gold_ids:
    raise RuntimeError(
    "Gold train/validation split does not cover "
    "the complete-label population exactly."
    )

print("\nBASELINE-001 gold split: PASS")

# ============================================================

# GOLD POPULATION

# ============================================================

# The gold population and gold split are both derived directly

# from train.csv, exactly as in BASELINE-001.

gold_series = series[
series["StudyInstanceUID"].isin(gold_train_ids)
].copy()

if len(gold_series) != 46:
        raise RuntimeError(
        "Expected exactly 46 gold training studies in "
        "series selection, "
        f"found {len(gold_series)}"
        )

# ============================================================

# REPORT-ONLY POPULATION

# ============================================================

report_ids = set(
report["StudyInstanceUID"]
)

# REPORT-001 contains exactly the 4,349 report-only studies.

if len(report_ids) != 4349:
        raise RuntimeError(
        f"Expected 4,349 report-only studies, "
        f"found {len(report_ids)}"
        )

# Make absolutely sure report-only and gold populations

# are disjoint.

gold_report_overlap = report_ids & gold_ids

if gold_report_overlap:
        raise RuntimeError(
        "REPORT-001 report-only population overlaps "
        f"gold population: {len(gold_report_overlap)} studies"
        )

# ============================================================

# KEEP ONLY REPORT STUDIES WITH >=1 USABLE LABEL

# ============================================================

report_label_count = (
report[TARGETS]
.notna()
.sum(axis=1)
)

report["num_report_labels"] = report_label_count

report_usable = report[
report["num_report_labels"] > 0
].copy()

print()
print("Report-only studies:", len(report))
print(
"Report-only studies with >=1 usable label:",
len(report_usable),
)

if len(report_usable) != 2419:
        raise RuntimeError(
        "Expected exactly 2,419 report-supervised studies, "
        f"found {len(report_usable)}"
        )

# ============================================================

# INTERSECT WITH SELECTED MRI SERIES

# ============================================================

report_usable = report_usable.merge(
series[
[
"StudyInstanceUID",
"SeriesInstanceUID",
"zarr_path",
]
],
on="StudyInstanceUID",
how="left",
validate="one_to_one",
)

missing_series = report_usable[
report_usable["SeriesInstanceUID"].isna()
]

if len(missing_series):
    raise RuntimeError(
    "Report-supervised studies without selected MRI series: "
    f"{len(missing_series)}"
    )

print(
"Report-supervised studies with selected MRI series:",
len(report_usable),
)

# ============================================================

# BUILD GOLD TRAIN

# ============================================================

gold_train = gold_series.copy()

# Gold labels come from the original structured labels.

# This makes label provenance explicit and independent of

# REPORT-001.

gold_labels = train_full[
train_full["StudyInstanceUID"].isin(gold_train_ids)
][
["StudyInstanceUID"] + TARGETS
].copy()

assert_unique(
gold_labels,
"StudyInstanceUID",
"gold labels",
)

if len(gold_labels) != 46:
        raise RuntimeError(
        f"Expected 46 gold training label rows, "
        f"found {len(gold_labels)}"
        )

validate_binary_or_nan(
gold_labels,
TARGETS,
"gold labels",
)

gold_train = gold_train.merge(
gold_labels,
on="StudyInstanceUID",
how="left",
validate="one_to_one",
)

# Every gold training study must have all 12 labels.

gold_complete = (
gold_train[TARGETS]
.notna()
.all(axis=1)
)

if not gold_complete.all():
        bad = gold_train.loc[
        ~gold_complete,
        "StudyInstanceUID"
        ].tolist()


        raise RuntimeError(
            "Gold training studies missing labels: "
            f"{bad}"
)


gold_train["label_source"] = "gold"
gold_train["num_report_labels"] = 12

# ============================================================

# DROP REPORT EVIDENCE/CONFIDENCE FROM GOLD

# ============================================================

# For the final manifest, keep report evidence/confidence

# only for report-derived labels. Gold is independent.

report_confidence_cols = [
f"{t}__confidence"
for t in TARGETS
]

report_evidence_cols = [
f"{t}__evidence"
for t in TARGETS
]

report_metadata_cols = (
report_confidence_cols
+ report_evidence_cols
)

# ============================================================

# BUILD REPORT-WEAK POPULATION

# ============================================================

report_weak = report_usable.copy()

report_weak["label_source"] = "report_v2.2"

# ============================================================

# SELECT FINAL COLUMNS

# ============================================================

base_columns = [
"StudyInstanceUID",
"SeriesInstanceUID",
"zarr_path",
]

gold_columns = (
base_columns
+ TARGETS
+ ["label_source", "num_report_labels"]
)

report_columns = (
base_columns
+ TARGETS
+ report_confidence_cols
+ report_evidence_cols
+ ["label_source", "num_report_labels"]
)

gold_train = gold_train[gold_columns]
report_weak = report_weak[report_columns]

# ============================================================

# ALIGN GOLD AND REPORT COLUMNS

# ============================================================

for col in report_columns:
  if col not in gold_train.columns:
    gold_train[col] = pd.NA

for col in gold_columns:
  if col not in report_weak.columns:
     report_weak[col] = pd.NA

final_columns = (
base_columns
+ TARGETS
+ report_confidence_cols
+ report_evidence_cols
+ ["label_source", "num_report_labels"]
)

gold_train = gold_train[final_columns]
report_weak = report_weak[final_columns]

# ============================================================

# CONCATENATE

# ============================================================

manifest = pd.concat(
[gold_train, report_weak],
ignore_index=True,
)

# ============================================================

# FINAL VALIDATION

# ============================================================

print("\n" + "=" * 70)
print("FINAL MANIFEST VALIDATION")
print("=" * 70)

assert_unique(
manifest,
"StudyInstanceUID",
"BASELINE-002 manifest",
)

expected_rows = 46 + 2419

if len(manifest) != expected_rows:
    raise RuntimeError(
    f"Expected {expected_rows} rows, "
    f"found {len(manifest)}"
    )

source_counts = (
manifest["label_source"]
.value_counts()
)

print("\nRows by label source:")
print(source_counts.to_string())

if source_counts.get("gold", 0) != 46:
        raise RuntimeError(
        "Expected 46 gold rows"
        )

if source_counts.get("report_v2.2", 0) != 2419:
    raise RuntimeError(
    "Expected 2,419 report_v2.2 rows"
    )

# ============================================================

# GOLD VALIDATION LEAKAGE CHECK

# ============================================================

# Gold validation must NEVER appear in the training manifest.

validation_overlap = (
set(manifest["StudyInstanceUID"])
& gold_val_ids
)

if validation_overlap:
        raise RuntimeError(
        "CRITICAL: gold validation leakage detected. "
        f"{len(validation_overlap)} validation studies "
        "appear in BASELINE-002 training manifest."
        )

print(
"\nGold validation overlap:",
len(validation_overlap),
)

if len(validation_overlap) != 0:
        raise RuntimeError(
        "Expected zero gold validation studies in training."
        )

# ============================================================

# REPORT-DERIVED / GOLD POPULATION CHECK

# ============================================================

report_training_ids = set(
report_weak["StudyInstanceUID"]
)

if report_training_ids & gold_ids:
        raise RuntimeError(
        "Report-derived training overlaps gold population."
        )

print(
"Report-derived / gold overlap:",
len(report_training_ids & gold_ids),
)

# ============================================================

# TARGET-LEVEL COUNTS

# ============================================================

print("\nTarget-level labeled pairs:")

total_pairs = 0
for target in TARGETS:
    n = manifest[target].notna().sum()
    total_pairs += n

    pos = (manifest[target] == 1).sum()
    neg = (manifest[target] == 0).sum()

    print(
        f"{target:18s} "
        f"labeled={n:4d} "
        f"positive={pos:4d} "
        f"negative={neg:4d}"
    )


expected_pairs = 46 * 12 + 7887

if total_pairs != expected_pairs:
    raise RuntimeError(
    f"Expected {expected_pairs} total labeled pairs, "
    f"found {total_pairs}"
    )

print()
print("Total labeled target pairs:", total_pairs)

# ============================================================

# REPORT-SPECIFIC CHECK

# ============================================================

weak_pairs = 0

for target in TARGETS:
   weak_pairs += report_weak[target].notna().sum()

if weak_pairs != 7887:
    raise RuntimeError(
    f"Expected 7,887 weak-label pairs, "
    f"found {weak_pairs}"
     )


print("Weak-label pairs:", weak_pairs)

# ============================================================

# STUDY LABEL COUNTS

# ============================================================

label_count = (
report_weak[TARGETS]
.notna()
.sum(axis=1)
)

print("\nWeak-label count per study:")
print(
label_count
.value_counts()
.sort_index()
.to_string()
)

# ============================================================

# WRITE MANIFEST

# ============================================================

manifest.to_csv(
OUTPUT_MANIFEST,
index=False,
)

print()
print("Wrote:")
print(OUTPUT_MANIFEST)
print("Rows:", len(manifest))
print("Columns:", len(manifest.columns))

# ============================================================

# WRITE AUDIT

# ============================================================

audit_lines = []

audit_lines.append(
"BASELINE-002 MANIFEST AUDIT"
)

audit_lines.append(
"========================================"
)

audit_lines.append(
"BASELINE-001 GOLD SPLIT"
)

audit_lines.append(
"----------------------------------------"
)

audit_lines.append(
f"Gold population: {len(gold_ids)}"
)

audit_lines.append(
f"Gold train studies: {len(gold_train)}"
)

audit_lines.append(
f"Gold validation studies: {len(gold_val_ids)}"
)

audit_lines.append(
f"Split method: sklearn train_test_split"
)

audit_lines.append(
f"Split test_size: {BASELINE_001_TEST_SIZE}"
)

audit_lines.append(
f"Split random_state: {BASELINE_001_SEED}"
)

audit_lines.append(
"Split shuffle: True"
)

audit_lines.append(
""
)

audit_lines.append(
"BASELINE-002 TRAINING POPULATION"
)

audit_lines.append(
"----------------------------------------"
)

audit_lines.append(
f"Gold train studies: {len(gold_train)}"
)

audit_lines.append(
f"Report-only population: {len(report)}"
)

audit_lines.append(
f"Report-supervised studies: {len(report_weak)}"
)

audit_lines.append(
f"Total training studies: {len(manifest)}"
)

audit_lines.append(
f"Validation overlap: {len(validation_overlap)}"
)

audit_lines.append(
f"Weak-label pairs: {weak_pairs}"
)

audit_lines.append(
f"Total labeled target pairs: {total_pairs}"
)

audit_lines.append("")
audit_lines.append("TARGET COUNTS")
audit_lines.append("----------------------------------------")

for target in TARGETS:
    n = manifest[target].notna().sum()
    pos = (manifest[target] == 1).sum()
    neg = (manifest[target] == 0).sum()


    audit_lines.append(
        f"{target}: labeled={n}, "
        f"positive={pos}, negative={neg}"
    )


audit_lines.append("")
audit_lines.append("STATUS: PASS")

OUTPUT_AUDIT.write_text(
"\n".join(audit_lines) + "\n"
)

print("Wrote:")
print(OUTPUT_AUDIT)

print("\n" + "=" * 70)
print("BASELINE-002 MANIFEST BUILD: PASS")
print("=" * 70)
