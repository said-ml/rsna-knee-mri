from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from extractor_v2 import TARGETS, extract_report


EXPECTED_SHA256 = (
    "8b6145a1e5659c410bd629bd7acf0067b3d16fa5d0ab23bea9a9d6c6a3f1269d"
)

PROJECT_ROOT = Path("/workspace")

FROZEN_EXTRACTOR = (
    PROJECT_ROOT / "src" / "report_001" / "extractor_v2.py"
)

FREEZE_DIR = (
    PROJECT_ROOT / "data" / "reports" / "report_001_frozen"
)

OUTPUT_PATH = FREEZE_DIR / "report_labels.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_frozen_extractor() -> None:
    actual = sha256_file(FROZEN_EXTRACTOR)

    print("Frozen extractor verification")
    print(f"Expected SHA256: {EXPECTED_SHA256}")
    print(f"Actual SHA256:   {actual}")

    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            "FATAL: extractor SHA256 mismatch. "
            "Application aborted; frozen extractor was not used."
        )

    print("SHA256: PASS")
    print()


def validate_population(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["StudyInstanceUID", "Report", *TARGETS] if c not in df.columns]

    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}"
        )

    complete_mask = df[TARGETS].notna().all(axis=1)
    report_only_mask = df[TARGETS].isna().all(axis=1)
    mixed_mask = ~(complete_mask | report_only_mask)

    complete = int(complete_mask.sum())
    report_only = int(report_only_mask.sum())
    mixed = int(mixed_mask.sum())

    print("Population verification")
    print(f"Total studies:    {len(df)}")
    print(f"Complete-label:   {complete}")
    print(f"Report-only:      {report_only}")
    print(f"Mixed:             {mixed}")

    if mixed != 0:
        raise RuntimeError(
            f"FATAL: found {mixed} mixed-label studies. "
            "Population is not the expected 58/4349 partition."
        )

    if complete != 58 or report_only != 4349:
        raise RuntimeError(
            "FATAL: unexpected population. "
            f"Expected 58 complete + 4349 report-only, "
            f"got {complete} + {report_only}."
        )

    print("POPULATION CHECK: PASS")
    print()

    return df.loc[report_only_mask].copy()


def apply_extractor(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        result = extract_report(row["Report"])

        result["StudyInstanceUID"] = row["StudyInstanceUID"]
        rows.append(result)

        if i == 1 or i % 250 == 0 or i == total:
            print(f"Processed {i}/{total}")

    predictions = pd.DataFrame(rows)

    columns = (
        ["StudyInstanceUID", "language_hint"]
        + TARGETS
        + [f"{t}__evidence" for t in TARGETS]
        + [f"{t}__confidence" for t in TARGETS]
    )

    return predictions[columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply frozen REPORT-001 v2.2 extractor to report-only population."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV containing all 4,407 studies.",
    )

    args = parser.parse_args()

    FREEZE_DIR.mkdir(parents=True, exist_ok=True)

    verify_frozen_extractor()

    print(f"Input: {args.input}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    df = pd.read_csv(args.input)

    report_only = validate_population(df)

    predictions = apply_extractor(report_only)

    if len(predictions) != 4349:
        raise RuntimeError(
            f"FATAL: expected 4349 predictions, got {len(predictions)}."
        )

    if predictions["StudyInstanceUID"].duplicated().any():
        raise RuntimeError(
            "FATAL: duplicate StudyInstanceUID detected in output."
        )

    predictions.to_csv(OUTPUT_PATH, index=False)

    print()
    print("REPORT-001 v2.2 APPLICATION COMPLETE")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows:   {len(predictions)}")
    print(f"Cols:   {len(predictions.columns)}")
    print()

    print("Coverage:")
    for target in TARGETS:
        mask = predictions[target].notna()

        coverage = mask.mean()
        positive = int((predictions.loc[mask, target] == 1).sum())
        negative = int((predictions.loc[mask, target] == 0).sum())

        print(
            f"{target:18s} "
            f"coverage={coverage:6.1%} "
            f"positive={positive:4d} "
            f"negative={negative:4d}"
        )


if __name__ == "__main__":
    main()
