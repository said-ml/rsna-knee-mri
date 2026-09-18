from pathlib import Path
import pandas as pd
import shutil
import os
import sys

PROJECT_ROOT = Path("/workspace").resolve()
RAW_ROOT = (PROJECT_ROOT / "data/raw/train_series").resolve()
PLAN = (PROJECT_ROOT / "data/reports/zarr_dicom_deletion_plan.csv").resolve()

EXPECTED_CANDIDATES = 16484
CONFIRMATION = "DELETE 16484 SERIES"


def fail(message):
    print()
    print("=" * 78)
    print("ABORTED")
    print("=" * 78)
    print(message)
    print("NO FURTHER DELETIONS WILL BE PERFORMED.")
    sys.exit(1)


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
    print("GUARDED ZARR → DICOM DELETION")
    print("=" * 78)
    print()
    print("THIS OPERATION IS DESTRUCTIVE.")
    print("Only the exact series directories listed in the audited")
    print("deletion manifest will be removed.")
    print()

    # ------------------------------------------------------------
    # Verify plan
    # ------------------------------------------------------------

    if not PLAN.is_file():
        fail(f"Deletion manifest not found:\n{PLAN}")

    if not RAW_ROOT.is_dir():
        fail(f"Raw DICOM root not found:\n{RAW_ROOT}")

    plan = pd.read_csv(PLAN)

    required_columns = {
        "study_uid",
        "series_uid",
        "source",
        "dicom_dir",
        "zarr_path",
        "zarr_exists",
        "dicom_exists",
        "dicom_size_bytes",
    }

    missing_columns = required_columns - set(plan.columns)

    if missing_columns:
        fail(
            "Deletion manifest is missing required columns:\n"
            + "\n".join(sorted(missing_columns))
        )

    # ------------------------------------------------------------
    # Candidate count
    # ------------------------------------------------------------

    if len(plan) != EXPECTED_CANDIDATES:
        fail(
            f"Expected exactly {EXPECTED_CANDIDATES:,} candidates, "
            f"but manifest contains {len(plan):,}."
        )

    # ------------------------------------------------------------
    # Duplicate checks
    # ------------------------------------------------------------

    duplicate_pairs = plan[
        plan.duplicated(
            ["study_uid", "series_uid"],
            keep=False
        )
    ]

    if len(duplicate_pairs):
        fail(
            f"Duplicate study/series pairs detected: "
            f"{len(duplicate_pairs)}"
        )

    duplicate_series = plan[
        plan.duplicated("series_uid", keep=False)
    ]

    if len(duplicate_series):
        fail(
            f"Duplicate SeriesInstanceUIDs detected: "
            f"{len(duplicate_series)}"
        )

    # ------------------------------------------------------------
    # Filesystem validation
    # ------------------------------------------------------------

    print("Validating deletion manifest...")
    print()

    total_bytes = 0

    for i, row in plan.iterrows():

        study = str(row["study_uid"])
        series = str(row["series_uid"])

        expected_dicom = (
            RAW_ROOT / study / series
        ).resolve()

        manifest_dicom = Path(
            str(row["dicom_dir"])
        ).resolve()

        zarr_path = Path(
            str(row["zarr_path"])
        ).resolve()

        # Exact path agreement
        if manifest_dicom != expected_dicom:
            fail(
                "Manifest DICOM path mismatch:\n"
                f"Expected: {expected_dicom}\n"
                f"Manifest: {manifest_dicom}"
            )

        # Prevent path traversal / wrong root
        try:
            expected_dicom.relative_to(RAW_ROOT)
        except ValueError:
            fail(
                f"Candidate path escapes RAW_ROOT:\n"
                f"{expected_dicom}"
            )

        # Candidate must be exactly study/series
        relative_parts = expected_dicom.relative_to(
            RAW_ROOT
        ).parts

        if len(relative_parts) != 2:
            fail(
                f"Unexpected deletion depth:\n"
                f"{expected_dicom}"
            )

        # Study and series must be real directories
        study_dir = expected_dicom.parent

        if not study_dir.is_dir():
            fail(
                f"Study directory missing:\n"
                f"{study_dir}"
            )

        if expected_dicom.is_symlink():
            fail(
                f"Refusing to delete symlink:\n"
                f"{expected_dicom}"
            )

        if not expected_dicom.is_dir():
            fail(
                f"DICOM series directory missing:\n"
                f"{expected_dicom}"
            )

        # Zarr must still exist
        if not zarr_path.exists():
            fail(
                f"Corresponding Zarr does not exist:\n"
                f"{zarr_path}\n"
                f"Refusing deletion of:\n"
                f"{expected_dicom}"
            )

        if not bool(row["zarr_exists"]):
            fail(
                f"Manifest says Zarr does not exist:\n"
                f"{zarr_path}"
            )

        # DICOM must still exist according to manifest
        if not bool(row["dicom_exists"]):
            fail(
                f"Manifest says DICOM does not exist:\n"
                f"{expected_dicom}"
            )

        size = directory_size_bytes(expected_dicom)

        if size <= 0:
            fail(
                f"Candidate DICOM directory is empty:\n"
                f"{expected_dicom}"
            )

        total_bytes += size

        if (i + 1) % 1000 == 0:
            print(
                f"  validated {i + 1:,} / "
                f"{len(plan):,}"
            )

    # ------------------------------------------------------------
    # Current disk state
    # ------------------------------------------------------------

    usage = shutil.disk_usage(PROJECT_ROOT)

    free_before = usage.free
    projected_free = free_before + total_bytes

    # ------------------------------------------------------------
    # Final confirmation screen
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL DELETION SUMMARY")
    print("=" * 78)

    print(
        f"Candidate series        : {len(plan):,}"
    )

    print(
        f"DICOM bytes             : {total_bytes:,}"
    )

    print(
        f"DICOM GiB               : "
        f"{total_bytes / (1024**3):.2f}"
    )

    print(
        f"Free space BEFORE       : "
        f"{free_before / (1024**3):.2f} GiB"
    )

    print(
        f"Projected free space    : "
        f"{projected_free / (1024**3):.2f} GiB"
    )

    print()
    print("RAW ROOT:")
    print(RAW_ROOT)

    print()
    print("DELETION MANIFEST:")
    print(PLAN)

    print()
    print("IMPORTANT:")
    print("  • Only individual SeriesInstanceUID directories will be deleted.")
    print("  • Study directories themselves will NOT be deleted.")
    print("  • Corresponding Zarr directories will NOT be deleted.")
    print("  • The operation cannot be undone from this script.")

    print()
    print("=" * 78)
    print("FINAL CONFIRMATION REQUIRED")
    print("=" * 78)
    print()
    print(
        f"Type exactly: {CONFIRMATION}"
    )
    print()
    confirmation = input("> ")

    if confirmation != CONFIRMATION:
        print()
        print("Confirmation did not match.")
        print("ABORTED. NO FILES WERE DELETED.")
        return

    # ------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("DELETING SERIES DIRECTORIES")
    print("=" * 78)

    deleted = 0
    deleted_bytes = 0

    for i, row in plan.iterrows():

        study = str(row["study_uid"])
        series = str(row["series_uid"])

        dicom_dir = (
            RAW_ROOT / study / series
        ).resolve()

        zarr_path = Path(
            str(row["zarr_path"])
        ).resolve()

        # Re-check immediately before deletion
        if not dicom_dir.is_dir():
            fail(
                f"DICOM directory disappeared before deletion:\n"
                f"{dicom_dir}"
            )

        if dicom_dir.is_symlink():
            fail(
                f"Symlink detected immediately before deletion:\n"
                f"{dicom_dir}"
            )

        if not zarr_path.exists():
            fail(
                f"Zarr disappeared immediately before deletion:\n"
                f"{zarr_path}"
            )

        # Ensure exactly two path components below RAW_ROOT
        try:
            relative = dicom_dir.relative_to(RAW_ROOT)
        except ValueError:
            fail(
                f"Deletion path escaped RAW_ROOT:\n"
                f"{dicom_dir}"
            )

        if len(relative.parts) != 2:
            fail(
                f"Unsafe deletion depth:\n"
                f"{dicom_dir}"
            )

        size_before = directory_size_bytes(dicom_dir)

        print(
            f"[{i + 1:5d}/{len(plan):5d}] "
            f"DELETE {study}/{series}"
        )

        shutil.rmtree(dicom_dir)

        # Immediate verification
        if dicom_dir.exists():
            fail(
                f"Directory still exists after deletion:\n"
                f"{dicom_dir}"
            )

        # Zarr must remain
        if not zarr_path.exists():
            fail(
                f"Zarr disappeared after DICOM deletion:\n"
                f"{zarr_path}"
            )

        deleted += 1
        deleted_bytes += size_before

    # ------------------------------------------------------------
    # Post-deletion verification
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("POST-DELETION VERIFICATION")
    print("=" * 78)

    missing_zarr = 0
    remaining_dicom = 0

    for _, row in plan.iterrows():

        dicom_dir = Path(
            str(row["dicom_dir"])
        ).resolve()

        zarr_path = Path(
            str(row["zarr_path"])
        ).resolve()

        if dicom_dir.exists():
            remaining_dicom += 1

        if not zarr_path.exists():
            missing_zarr += 1

    usage_after = shutil.disk_usage(PROJECT_ROOT)

    print(
        f"Deleted series          : {deleted:,}"
    )

    print(
        f"Deleted DICOM GiB       : "
        f"{deleted_bytes / (1024**3):.2f}"
    )

    print(
        f"Candidate DICOM left    : "
        f"{remaining_dicom:,}"
    )

    print(
        f"Zarr outputs missing    : "
        f"{missing_zarr:,}"
    )

    print(
        f"Free space AFTER        : "
        f"{usage_after.free / (1024**3):.2f} GiB"
    )

    print()

    if (
        deleted == EXPECTED_CANDIDATES
        and remaining_dicom == 0
        and missing_zarr == 0
    ):
        print("=" * 78)
        print("DELETION COMPLETE — POST-CHECK PASS")
        print("=" * 78)
    else:
        print("=" * 78)
        print("POST-CHECK FAILED — INVESTIGATE")
        print("=" * 78)


if __name__ == "__main__":
    main()
