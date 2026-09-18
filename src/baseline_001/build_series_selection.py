from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/workspace")

TRAIN_CSV = PROJECT_ROOT / "data/raw/train.csv"
SERIES_CSV = PROJECT_ROOT / "data/raw/train_series.csv"
MANIFEST_CSV = (
    PROJECT_ROOT
    / "data/reports/zarr_conversion_manifest_v3.csv"
)

LEGACY_MANIFEST_CSV = (
    PROJECT_ROOT
    / "data/reports/zarr_conversion_manifest_v2.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data/reports/baseline_001_series_selection.csv"
)


def main():
    train = pd.read_csv(TRAIN_CSV)
    series = pd.read_csv(SERIES_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)
    legacy_manifest = pd.read_csv(LEGACY_MANIFEST_CSV)

    # ---------------------------------------------------------
    # Basic integrity
    # ---------------------------------------------------------

    required_train = {"StudyInstanceUID"}
    required_series = {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Fluid_Sensitive",
        "Fat_Suppression",
        "Anatomical_Plane",
    }
    required_manifest = {
        "study_uid",
        "series_uid",
        "output_path",
        "status",
    }

    if not required_train.issubset(train.columns):
        raise RuntimeError("train.csv missing StudyInstanceUID")

    if not required_series.issubset(series.columns):
        raise RuntimeError("train_series.csv missing required columns")

    if not required_manifest.issubset(manifest.columns):
        raise RuntimeError("Zarr manifest missing required columns")

    # ---------------------------------------------------------
    # Dataset study integrity
    # ---------------------------------------------------------

    train_studies = set(train["StudyInstanceUID"].astype(str))

    if len(train_studies) != len(train):
        raise RuntimeError(
            "train.csv contains duplicate StudyInstanceUID values."
        )

    series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
    series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)

    if series["SeriesInstanceUID"].duplicated().any():
        raise RuntimeError(
            "Duplicate SeriesInstanceUID detected."
        )

    # ---------------------------------------------------------
    # Confirm all studies have sagittal series
    # ---------------------------------------------------------

    sagittal = series[
        series["Anatomical_Plane"].eq("Sagittal")
    ].copy()

    sagittal_studies = set(sagittal["StudyInstanceUID"])

    missing_sagittal = train_studies - sagittal_studies

    if missing_sagittal:
        raise RuntimeError(
            f"{len(missing_sagittal)} studies have no sagittal series."
        )

    # ---------------------------------------------------------
    # Deterministic selection
    #
    # Rule:
    #   1. Prefer Fluid_Sensitive=1
    #   2. Within the preferred group choose the smallest
    #      SeriesInstanceUID
    #   3. If no fluid-sensitive sagittal series exists,
    #      choose the smallest sagittal SeriesInstanceUID
    # ---------------------------------------------------------

    sagittal["priority"] = (
        sagittal["Fluid_Sensitive"].eq(1).astype(int)
    )

    selected = (
        sagittal
        .sort_values(
            [
                "StudyInstanceUID",
                "priority",
                "SeriesInstanceUID",
            ],
            ascending=[True, False, True],
        )
        .groupby(
            "StudyInstanceUID",
            as_index=False,
            sort=False,
        )
        .first()
    )

    selected["selection_rule"] = selected["priority"].map(
        {
            1: "sagittal_fluid_sensitive_min_series_uid",
            0: "sagittal_fallback_min_series_uid",
        }
    )

    # ---------------------------------------------------------
    # Exact expected coverage
    # ---------------------------------------------------------

    if len(selected) != len(train):
        raise RuntimeError(
            f"Expected {len(train)} selected studies, "
            f"got {len(selected)}."
        )

    if selected["StudyInstanceUID"].nunique() != len(train):
        raise RuntimeError(
            "Selected routing table does not contain exactly "
            "one row per study."
        )

    # ---------------------------------------------------------
    # Attach Zarr path using BOTH study + series UID.
    # Never join on StudyInstanceUID alone.
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Resolve Zarr paths.
    #
    # V3 is canonical. V2 is a legacy fallback for the 50
    # series converted before the production v3 conversion.
    #
    # Always join on BOTH study + series UID.
    # ---------------------------------------------------------

    manifest = manifest[
        manifest["status"].astype(str).str.upper().eq("SUCCESS")
    ].copy()

    legacy_manifest = legacy_manifest[
        legacy_manifest["status"].astype(str).str.upper().eq("SUCCESS")
    ].copy()

    for m in (manifest, legacy_manifest):
        m["study_uid"] = m["study_uid"].astype(str)
        m["series_uid"] = m["series_uid"].astype(str)

    v3_lookup = manifest[
        [
            "study_uid",
            "series_uid",
            "output_path",
        ]
    ].rename(
        columns={
            "study_uid": "StudyInstanceUID",
            "series_uid": "SeriesInstanceUID",
            "output_path": "zarr_path",
        }
    )

    v2_lookup = legacy_manifest[
        [
            "study_uid",
            "series_uid",
            "output_path",
        ]
    ].rename(
        columns={
            "study_uid": "StudyInstanceUID",
            "series_uid": "SeriesInstanceUID",
            "output_path": "zarr_path_v2",
        }
    )

    selected = selected.merge(
        v3_lookup,
        on=[
            "StudyInstanceUID",
            "SeriesInstanceUID",
        ],
        how="left",
        validate="one_to_one",
    )

    selected = selected.merge(
        v2_lookup,
        on=[
            "StudyInstanceUID",
            "SeriesInstanceUID",
        ],
        how="left",
        validate="one_to_one",
    )

    selected["zarr_path"] = selected["zarr_path"].fillna(
        selected["zarr_path_v2"]
    )

    selected["zarr_source"] = selected["zarr_path_v2"].notna().map(
        {
            False: "v3",
            True: "v2_legacy_fallback",
        }
    )

    selected = selected.drop(columns=["zarr_path_v2"])

    # ---------------------------------------------------------
    # Zarr coverage
    # ---------------------------------------------------------

    if selected["zarr_path"].isna().any():
        missing = selected[
            selected["zarr_path"].isna()
        ][
            ["StudyInstanceUID", "SeriesInstanceUID"]
        ]

        raise RuntimeError(
            f"Missing Zarr path for {len(missing)} selected series."
        )

    # ---------------------------------------------------------
    # Final columns
    # ---------------------------------------------------------

    selected = selected[
        [
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Anatomical_Plane",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "zarr_path",
            "zarr_source",
            "selection_rule",
        ]
    ].copy()

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # Verify artifact was actually written
    # ---------------------------------------------------------

    if not OUTPUT_CSV.exists():
        raise RuntimeError(
            f"Routing artifact was not created: {OUTPUT_CSV}"
        )

    if OUTPUT_CSV.stat().st_size == 0:
        raise RuntimeError(
            f"Routing artifact is empty: {OUTPUT_CSV}"
        )

    # Re-read the artifact and verify row count.
    written = pd.read_csv(OUTPUT_CSV)

    if len(written) != len(selected):
        raise RuntimeError(
            f"Written row count mismatch: "
            f"{len(written)} != {len(selected)}"
        )

    print(
        f"ARTIFACT WRITTEN: {OUTPUT_CSV} "
        f"({OUTPUT_CSV.stat().st_size:,} bytes)"
    )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("=" * 70)
    print("BASELINE-001 SERIES ROUTING")
    print("=" * 70)

    print(f"Training studies          : {len(train):,}")
    print(f"Selected series           : {len(selected):,}")
    print(
        "Unique selected studies   : "
        f"{selected['StudyInstanceUID'].nunique():,}"
    )
    print(
        "Unique selected series    : "
        f"{selected['SeriesInstanceUID'].nunique():,}"
    )

    print("\nSELECTION RULE")
    print(
        selected["selection_rule"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nANATOMICAL PLANE")
    print(
        selected["Anatomical_Plane"]
        .value_counts()
        .to_string()
    )

    print("\nZARR PATH COVERAGE")
    print(
        f"Missing Zarr paths        : "
        f"{selected['zarr_path'].isna().sum():,}"
    )

    print("\nZARR SOURCE")
    print(
        selected["zarr_source"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nOUTPUT")
    print(OUTPUT_CSV)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
