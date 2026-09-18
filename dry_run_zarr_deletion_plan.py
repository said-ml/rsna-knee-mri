from pathlib import Path
import pandas as pd
import os
import shutil

PROJECT_ROOT = Path("/workspace")
RAW_ROOT = PROJECT_ROOT / "data/raw/train_series"
V3_MANIFEST = PROJECT_ROOT / "data/reports/zarr_conversion_manifest_v3.csv"
V2_MANIFEST = PROJECT_ROOT / "data/reports/zarr_conversion_manifest_v2.csv"
OUTPUT = PROJECT_ROOT / "data/reports/zarr_dicom_deletion_plan.csv"


def directory_size_bytes(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def main():
    print("=" * 78)
    print("ZARR → DICOM DRY-RUN DELETION PLAN")
    print("=" * 78)

    # Load v3
    v3 = pd.read_csv(V3_MANIFEST)
    v3 = v3[v3["status"] == "SUCCESS"].copy()

    records = []

    for _, row in v3.iterrows():
        study = str(row["study_uid"])
        series = str(row["series_uid"])

        records.append({
            "study_uid": study,
            "series_uid": series,
            "source": "v3",
            "dicom_dir": str(RAW_ROOT / study / series),
            "zarr_path": str(Path(str(row["output_path"]))),
        })

    # Load v2
    if V2_MANIFEST.exists():
        v2 = pd.read_csv(V2_MANIFEST)

        if "status" in v2.columns:
            v2 = v2[v2["status"] == "SUCCESS"].copy()

        study_col = (
            "study_uid"
            if "study_uid" in v2.columns
            else "StudyInstanceUID"
        )

        series_col = (
            "series_uid"
            if "series_uid" in v2.columns
            else "SeriesInstanceUID"
        )

        path_col = (
            "output_path"
            if "output_path" in v2.columns
            else None
        )

        for _, row in v2.iterrows():
            study = str(row[study_col])
            series = str(row[series_col])

            if path_col is not None:
                zarr_path = Path(str(row[path_col]))
            else:
                zarr_path = (
                    PROJECT_ROOT
                    / "data/zarr"
                    / study
                    / f"{series}.zarr"
                )

            records.append({
                "study_uid": study,
                "series_uid": series,
                "source": "v2",
                "dicom_dir": str(RAW_ROOT / study / series),
                "zarr_path": str(zarr_path),
            })

    plan = pd.DataFrame(records)

    print()
    print("Candidate series:", len(plan))

    # Identity checks
    duplicate_pairs = plan[
        plan.duplicated(
            ["study_uid", "series_uid"],
            keep=False
        )
    ]

    duplicate_series = plan[
        plan.duplicated("series_uid", keep=False)
    ]

    print("Duplicate study/series pairs:", len(duplicate_pairs))
    print("Duplicate SeriesInstanceUIDs:", len(duplicate_series))

    # Existence checks
    plan["zarr_exists"] = plan["zarr_path"].map(
        lambda x: Path(x).exists()
    )

    plan["dicom_exists"] = plan["dicom_dir"].map(
        lambda x: Path(x).is_dir()
    )

    print()
    print("Zarr missing:", (~plan["zarr_exists"]).sum())
    print("DICOM missing:", (~plan["dicom_exists"]).sum())

    # Exact size
    print()
    print("=" * 78)
    print("CALCULATING EXACT DICOM DELETION SIZE")
    print("=" * 78)
    print("READ-ONLY")
    print()

    sizes = []

    for i, row in plan.iterrows():
        dicom_dir = Path(row["dicom_dir"])

        if not dicom_dir.is_dir():
            sizes.append(0)
            continue

        sizes.append(directory_size_bytes(dicom_dir))

        if (i + 1) % 500 == 0:
            print(f"  scanned {i + 1:,} / {len(plan):,}")

    plan["dicom_size_bytes"] = sizes

    # Filesystem state
    usage = shutil.disk_usage(PROJECT_ROOT)

    total_disk = usage.total
    used_disk = usage.used
    free_before = usage.free

    deletion_bytes = int(plan["dicom_size_bytes"].sum())
    free_after = free_before + deletion_bytes

    # Study-level analysis
    candidate_by_study = (
        plan.groupby("study_uid")
        .size()
    )

    study_rows = []

    for study_dir in sorted(RAW_ROOT.iterdir()):
        if not study_dir.is_dir():
            continue

        all_series = [
            p for p in study_dir.iterdir()
            if p.is_dir()
        ]

        total_count = len(all_series)
        candidate_count = int(
            candidate_by_study.get(study_dir.name, 0)
        )

        study_rows.append({
            "study_uid": study_dir.name,
            "total_series": total_count,
            "candidate_series": candidate_count,
            "remaining_series_after_deletion":
                total_count - candidate_count,
        })

    study_df = pd.DataFrame(study_rows)

    fully_deletable_studies = study_df[
        study_df["total_series"]
        == study_df["candidate_series"]
    ]

    partially_deletable_studies = study_df[
        (study_df["candidate_series"] > 0)
        & (
            study_df["candidate_series"]
            < study_df["total_series"]
        )
    ]

    # Save exact deletion plan
    plan = plan.sort_values(
        ["study_uid", "series_uid"]
    )

    plan.to_csv(OUTPUT, index=False)

    # Report
    print()
    print("=" * 78)
    print("DELETION PLAN SUMMARY")
    print("=" * 78)

    print(f"Candidate series              : {len(plan):,}")
    print(
        f"Candidate studies             : "
        f"{plan['study_uid'].nunique():,}"
    )
    print(
        f"Duplicate study/series pairs  : "
        f"{len(duplicate_pairs):,}"
    )
    print(
        f"Duplicate series UIDs         : "
        f"{len(duplicate_series):,}"
    )
    print(
        f"Missing Zarr                  : "
        f"{(~plan['zarr_exists']).sum():,}"
    )
    print(
        f"Missing DICOM directories     : "
        f"{(~plan['dicom_exists']).sum():,}"
    )
    print(
        f"Studies fully deletable       : "
        f"{len(fully_deletable_studies):,}"
    )
    print(
        f"Studies partially deletable   : "
        f"{len(partially_deletable_studies):,}"
    )

    print()
    print(
        f"DICOM bytes to delete        : "
        f"{deletion_bytes:,}"
    )
    print(
        f"DICOM GiB to delete          : "
        f"{deletion_bytes / (1024**3):.2f}"
    )
    print(
        f"DICOM TiB to delete          : "
        f"{deletion_bytes / (1024**4):.3f}"
    )

    print()
    print(
        f"Free space BEFORE             : "
        f"{free_before / (1024**3):.2f} GiB"
    )
    print(
        f"Estimated free space AFTER    : "
        f"{free_after / (1024**3):.2f} GiB"
    )

    print()
    print(
        f"Total filesystem capacity     : "
        f"{total_disk / (1024**3):.2f} GiB"
    )
    print(
        f"Used space BEFORE             : "
        f"{used_disk / (1024**3):.2f} GiB"
    )

    safe = (
        len(plan) == 16484
        and len(duplicate_pairs) == 0
        and len(duplicate_series) == 0
        and plan["zarr_exists"].all()
        and plan["dicom_exists"].all()
        and (plan["dicom_size_bytes"] > 0).all()
    )

    print()
    print("=" * 78)
    print("SAFETY GATE")
    print("=" * 78)
    print(
        "DRY-RUN PLAN:",
        "SAFE" if safe else "DO NOT DELETE"
    )

    print()
    print("NO FILES WERE DELETED.")
    print()
    print("Exact deletion manifest:")
    print(OUTPUT)
    print("=" * 78)


if __name__ == "__main__":
    main()
