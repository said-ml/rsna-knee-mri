#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import shutil
import sys


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path("/workspace").resolve()
RAW_ROOT = (PROJECT_ROOT / "data/raw/train_series").resolve()
PLAN = (
    PROJECT_ROOT
    / "data/reports/zarr_dicom_deletion_plan.csv"
).resolve()

EXPECTED_TOTAL = 24371
EXPECTED_ALREADY_DELETED = 16484
EXPECTED_TO_DELETE = 7887

CONFIRMATION = "DELETE 7887 SERIES"


# ============================================================
# Helpers
# ============================================================

def fail(message):
    print()
    print("=" * 78)
    print("SAFETY FAILURE")
    print("=" * 78)
    print(message)
    print("=" * 78)
    sys.exit(1)


def directory_size_bytes(path):
    total = 0

    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size

    return total


def resolve_under_raw(path):
    path = path.resolve()

    try:
        relative = path.relative_to(RAW_ROOT)
    except ValueError:
        fail(
            f"Path escapes RAW_ROOT:\n"
            f"{path}\n"
            f"RAW_ROOT:\n"
            f"{RAW_ROOT}"
        )

    # Exact structure:
    # train_series / StudyInstanceUID / SeriesInstanceUID
    if len(relative.parts) != 2:
        fail(
            f"Unexpected DICOM directory depth:\n"
            f"{path}\n"
            f"Relative path:\n"
            f"{relative}"
        )

    return path


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)
    print("GUARDED DICOM DELETION — V2")
    print("=" * 78)
    print()

    # --------------------------------------------------------
    # Basic filesystem checks
    # --------------------------------------------------------

    if not PROJECT_ROOT.exists():
        fail(f"Project root does not exist:\n{PROJECT_ROOT}")

    if not RAW_ROOT.exists():
        fail(f"Raw DICOM root does not exist:\n{RAW_ROOT}")

    if not PLAN.exists():
        fail(f"Deletion plan does not exist:\n{PLAN}")

    # --------------------------------------------------------
    # Load plan
    # --------------------------------------------------------

    print("READING DELETION PLAN")
    print(f"PLAN: {PLAN}")
    print()

    plan = pd.read_csv(PLAN)

    required_columns = {
        "study_uid",
        "series_uid",
        "dicom_dir",
        "zarr_path",
        "zarr_exists",
        "dicom_exists",
    }

    missing_columns = required_columns - set(plan.columns)

    if missing_columns:
        fail(
            f"Plan is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Validate total plan
    # --------------------------------------------------------

    if len(plan) != EXPECTED_TOTAL:
        fail(
            f"Expected exactly {EXPECTED_TOTAL:,} total plan rows, "
            f"found {len(plan):,}"
        )

    # --------------------------------------------------------
    # Validate duplicate study/series pairs
    # --------------------------------------------------------

    duplicate_pairs = plan.duplicated(
        subset=["study_uid", "series_uid"],
        keep=False,
    )

    if duplicate_pairs.any():
        dup = plan.loc[
            duplicate_pairs,
            ["study_uid", "series_uid"]
        ]

        fail(
            "Duplicate StudyInstanceUID / SeriesInstanceUID pairs "
            f"found: {len(dup):,}"
        )

    # --------------------------------------------------------
    # Validate duplicate SeriesInstanceUID
    # --------------------------------------------------------

    duplicate_series = plan.duplicated(
        subset=["series_uid"],
        keep=False,
    )

    if duplicate_series.any():
        dup = plan.loc[
            duplicate_series,
            ["study_uid", "series_uid"]
        ]

        fail(
            "Duplicate SeriesInstanceUID values found:\n"
            f"{dup.to_string(index=False)}"
        )

    # --------------------------------------------------------
    # Validate all Zarr outputs
    # --------------------------------------------------------

    print("VALIDATING ALL ZARR OUTPUTS")
    print()

    missing_zarr = []

    for _, row in plan.iterrows():

        zarr_path = Path(
            str(row["zarr_path"])
        ).resolve()

        if not zarr_path.exists():
            missing_zarr.append(
                (
                    str(row["study_uid"]),
                    str(row["series_uid"]),
                    str(zarr_path),
                )
            )

    if missing_zarr:
        fail(
            f"Missing Zarr outputs: {len(missing_zarr):,}\n"
            f"First missing:\n{missing_zarr[:5]}"
        )

    print("Zarr outputs missing: 0")
    print()

    # --------------------------------------------------------
    # Determine actual filesystem state
    #
    # IMPORTANT:
    # We do NOT trust dicom_exists from the old snapshot.
    # --------------------------------------------------------

    print("SCANNING ACTUAL DICOM FILESYSTEM STATE")
    print()

    to_delete = []
    already_deleted = []

    for _, row in plan.iterrows():

        study = str(row["study_uid"])
        series = str(row["series_uid"])

        dicom_dir = resolve_under_raw(
            Path(str(row["dicom_dir"]))
        )

        # Safety: exact expected location
        expected = (
            RAW_ROOT
            / study
            / series
        ).resolve()

        if dicom_dir != expected:
            fail(
                "DICOM path does not match deterministic "
                f"Study/Series path.\n"
                f"Manifest: {dicom_dir}\n"
                f"Expected: {expected}"
            )

        if dicom_dir.exists():

            if dicom_dir.is_symlink():
                fail(
                    f"Refusing to operate on symlink:\n"
                    f"{dicom_dir}"
                )

            if not dicom_dir.is_dir():
                fail(
                    f"DICOM path exists but is not a directory:\n"
                    f"{dicom_dir}"
                )

            to_delete.append(
                {
                    "study_uid": study,
                    "series_uid": series,
                    "dicom_dir": dicom_dir,
                    "zarr_path": Path(
                        str(row["zarr_path"])
                    ).resolve(),
                }
            )

        else:
            already_deleted.append(
                {
                    "study_uid": study,
                    "series_uid": series,
                    "dicom_dir": dicom_dir,
                    "zarr_path": Path(
                        str(row["zarr_path"])
                    ).resolve(),
                }
            )

    # --------------------------------------------------------
    # State-count safety gate
    # --------------------------------------------------------

    print("=" * 78)
    print("CURRENT DATASET STATE")
    print("=" * 78)

    print(
        f"Total plan rows          : {len(plan):,}"
    )

    print(
        f"Already deleted DICOM    : "
        f"{len(already_deleted):,}"
    )

    print(
        f"DICOM still present      : "
        f"{len(to_delete):,}"
    )

    print(
        f"Zarr outputs missing     : "
        f"{len(missing_zarr):,}"
    )

    print()

    if len(already_deleted) != EXPECTED_ALREADY_DELETED:
        fail(
            f"Expected {EXPECTED_ALREADY_DELETED:,} already-deleted "
            f"series, found {len(already_deleted):,}"
        )

    if len(to_delete) != EXPECTED_TO_DELETE:
        fail(
            f"Expected exactly {EXPECTED_TO_DELETE:,} DICOM series "
            f"remaining, found {len(to_delete):,}"
        )

    # --------------------------------------------------------
    # Verify already-deleted set is genuinely absent
    # --------------------------------------------------------

    for item in already_deleted:

        if item["dicom_dir"].exists():
            fail(
                "Filesystem inconsistency: a supposedly deleted "
                f"DICOM directory exists:\n"
                f"{item['dicom_dir']}"
            )

        if not item["zarr_path"].exists():
            fail(
                "Zarr missing for an already-deleted DICOM series:\n"
                f"{item['zarr_path']}"
            )

    # --------------------------------------------------------
    # Verify every deletion candidate
    # --------------------------------------------------------

    print("VALIDATING 7,887 DELETION CANDIDATES")
    print()

    total_bytes = 0

    for i, item in enumerate(to_delete, 1):

        dicom_dir = item["dicom_dir"]
        zarr_path = item["zarr_path"]

        if not dicom_dir.exists():
            fail(
                f"Candidate disappeared during audit:\n"
                f"{dicom_dir}"
            )

        if dicom_dir.is_symlink():
            fail(
                f"Refusing symlink candidate:\n"
                f"{dicom_dir}"
            )

        if not dicom_dir.is_dir():
            fail(
                f"Candidate is not a directory:\n"
                f"{dicom_dir}"
            )

        if not zarr_path.exists():
            fail(
                f"Zarr missing for deletion candidate:\n"
                f"{zarr_path}"
            )

        total_bytes += directory_size_bytes(dicom_dir)

        if i % 500 == 0:
            print(
                f"  validated {i:,}/{len(to_delete):,}"
            )

    print(
        f"  validated {len(to_delete):,}/{len(to_delete):,}"
    )

    # --------------------------------------------------------
    # Disk-space report
    # --------------------------------------------------------

    usage_before = shutil.disk_usage(PROJECT_ROOT)

    gib = 1024 ** 3

    print()
    print("=" * 78)
    print("DELETION PLAN")
    print("=" * 78)

    print(
        f"Series to delete         : "
        f"{len(to_delete):,}"
    )

    print(
        f"Already deleted          : "
        f"{len(already_deleted):,}"
    )

    print(
        f"Exact DICOM size         : "
        f"{total_bytes / gib:.2f} GiB"
    )

    print(
        f"Free space BEFORE        : "
        f"{usage_before.free / gib:.2f} GiB"
    )

    print(
        f"Estimated free AFTER     : "
        f"{(usage_before.free + total_bytes) / gib:.2f} GiB"
    )

    print()

    print("SAFETY CONDITIONS")
    print("  ✓ Total plan rows = 24,371")
    print("  ✓ Already deleted = 16,484")
    print("  ✓ Remaining DICOM = 7,887")
    print("  ✓ Missing Zarr = 0")
    print("  ✓ Duplicate pairs = 0")
    print("  ✓ Duplicate SeriesInstanceUID = 0")
    print("  ✓ Every candidate is an exact Series directory")
    print("  ✓ Every candidate has a surviving Zarr output")
    print("  ✓ No symlinks accepted")
    print("  ✓ Study directories will NOT be deleted")
    print("  ✓ Zarr directories will NOT be deleted")

    print()
    print("=" * 78)
    print("FINAL CONFIRMATION REQUIRED")
    print("=" * 78)
    print()
    print(f"Type exactly: {CONFIRMATION}")
    print()

    confirmation = input("> ").strip()

    if confirmation != CONFIRMATION:
        print()
        print("CONFIRMATION DID NOT MATCH.")
        print("NO FILES WERE DELETED.")
        return

    # --------------------------------------------------------
    # Guarded deletion
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("STARTING GUARDED DELETION")
    print("=" * 78)
    print()

    deleted = 0
    deleted_bytes = 0

    for i, item in enumerate(to_delete):

        dicom_dir = item["dicom_dir"]
        zarr_path = item["zarr_path"]

        # Re-check immediately before deletion.
        if not dicom_dir.exists():
            fail(
                f"Candidate disappeared before deletion:\n"
                f"{dicom_dir}"
            )

        if dicom_dir.is_symlink():
            fail(
                f"Refusing symlink:\n"
                f"{dicom_dir}"
            )

        if not dicom_dir.is_dir():
            fail(
                f"Candidate is not a directory:\n"
                f"{dicom_dir}"
            )

        if not zarr_path.exists():
            fail(
                f"Zarr missing immediately before deletion:\n"
                f"{zarr_path}"
            )

        size_before = directory_size_bytes(dicom_dir)

        print(
            f"[{i + 1:5d}/{len(to_delete):5d}] "
            f"DELETE "
            f"{item['study_uid']}/"
            f"{item['series_uid']}"
        )

        shutil.rmtree(dicom_dir)

        # Immediate DICOM deletion verification.
        if dicom_dir.exists():
            fail(
                f"Directory still exists after deletion:\n"
                f"{dicom_dir}"
            )

        # Immediate Zarr preservation verification.
        if not zarr_path.exists():
            fail(
                f"Zarr disappeared after DICOM deletion:\n"
                f"{zarr_path}"
            )

        deleted += 1
        deleted_bytes += size_before

    # --------------------------------------------------------
    # Independent post-deletion verification
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("POST-DELETION VERIFICATION")
    print("=" * 78)

    remaining_dicom = 0
    missing_zarr_after = 0

    for item in to_delete:

        if item["dicom_dir"].exists():
            remaining_dicom += 1

        if not item["zarr_path"].exists():
            missing_zarr_after += 1

    # Verify ALL 24,371 Zarr outputs, not just the deleted batch.
    all_zarr_missing = 0

    for _, row in plan.iterrows():

        zarr_path = Path(
            str(row["zarr_path"])
        ).resolve()

        if not zarr_path.exists():
            all_zarr_missing += 1

    usage_after = shutil.disk_usage(PROJECT_ROOT)

    print(
        f"Deleted series          : "
        f"{deleted:,}"
    )

    print(
        f"Deleted DICOM GiB       : "
        f"{deleted_bytes / gib:.2f}"
    )

    print(
        f"Remaining target DICOM  : "
        f"{remaining_dicom:,}"
    )

    print(
        f"Missing target Zarr      : "
        f"{missing_zarr_after:,}"
    )

    print(
        f"Missing ALL Zarr         : "
        f"{all_zarr_missing:,}"
    )

    print(
        f"Free space AFTER        : "
        f"{usage_after.free / gib:.2f} GiB"
    )

    print()

    if (
        deleted == EXPECTED_TO_DELETE
        and remaining_dicom == 0
        and missing_zarr_after == 0
        and all_zarr_missing == 0
    ):
        print("=" * 78)
        print("DELETION COMPLETE — POST-CHECK PASS")
        print("=" * 78)
    else:
        print("=" * 78)
        print("POST-CHECK FAILED — INVESTIGATE")
        print("=" * 78)
        sys.exit(1)


if __name__ == "__main__":
    main()
