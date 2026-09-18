#!/usr/bin/env python3

"""
DATA-001-ZARR-v3

Production DICOM -> Zarr v3 converter.

Properties:
- DICOM remains immutable source of truth.
- SimpleITK/GDCM performs DICOM reconstruction.
- Geometry comes from the reconstructed SimpleITK image.
- Native decoded dtype is preserved.
- Atomic output creation.
- Resumable conversion.
- Incremental manifest writes.
- Existing verified v2 outputs are recognized.
- 40 GiB minimum free-space reserve.
- Production verification avoids a second full-volume RAM copy.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from zarr.codecs import BloscCodec


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/workspace")

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "train_series"
ZARR_ROOT = PROJECT_ROOT / "data" / "zarr"
REPORT_ROOT = PROJECT_ROOT / "data" / "reports"

MANIFEST_PATH = REPORT_ROOT / "zarr_conversion_manifest_v3.csv"
LEGACY_MANIFEST_PATH = REPORT_ROOT / "zarr_conversion_manifest_v2.csv"

CONVERSION_VERSION = "DATA-001-ZARR-v3"

ZARR_CHUNKS = (8, 256, 256)
ZSTD_LEVEL = 3

# Hard production safety reserve.
MIN_FREE_GIB = 40.0

# Number of Z slices sampled during production verification.
VERIFY_Z_SAMPLES = 3

FIELDNAMES = [
    "study_uid",
    "series_uid",
    "status",
    "n_directory_files",
    "n_dicom_files",
    "shape_z",
    "shape_y",
    "shape_x",
    "dtype",
    "spacing_x",
    "spacing_y",
    "spacing_z",
    "origin_x",
    "origin_y",
    "origin_z",
    "conversion_seconds",
    "output_path",
    "error",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_gib(path: Path) -> float:
    stat = os.statvfs(path)
    return (
        stat.f_bavail
        * stat.f_frsize
        / (1024 ** 3)
    )


def regular_files(path: Path):
    return sorted(
        p for p in path.iterdir()
        if p.is_file()
    )


def is_numeric_supported(dtype: np.dtype) -> bool:
    return (
        np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    )


def geometry_is_finite(spacing, origin, direction) -> bool:
    return (
        np.all(np.isfinite(np.asarray(spacing, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(origin, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(direction, dtype=np.float64)))
    )


def direction_is_reasonable(direction, atol=1e-4) -> bool:
    d = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    gram = d @ d.T

    return (
        np.allclose(gram, np.eye(3), atol=atol)
        and abs(abs(np.linalg.det(d)) - 1.0) <= 1e-3
    )


def output_path_for(study_uid: str, series_uid: str) -> Path:
    return (
        ZARR_ROOT
        / study_uid
        / f"{series_uid}.zarr"
    )


def discover_series(limit: int | None):
    studies = sorted(
        p for p in RAW_ROOT.iterdir()
        if p.is_dir()
    )

    results = []

    for study_dir in studies:
        study_uid = study_dir.name

        series_dirs = sorted(
            p for p in study_dir.iterdir()
            if p.is_dir()
        )

        for series_dir in series_dirs:
            results.append(
                (
                    study_uid,
                    series_dir.name,
                    series_dir,
                )
            )

            if limit is not None and len(results) >= limit:
                return results

    return results


# ---------------------------------------------------------------------
# Manifest / resume
# ---------------------------------------------------------------------

def load_existing_state():
    """
    Load successful work from both v3 and the validated v2 manifest.

    v3 SUCCESS:
        already completed by this production converter.

    v2 SUCCESS:
        already fully round-trip verified during validation and therefore
        safe to treat as completed.

    Returns:
        dict[(study_uid, series_uid)] = row
    """

    state = {}

    for manifest_path in (
        LEGACY_MANIFEST_PATH,
        MANIFEST_PATH,
    ):
        if not manifest_path.exists():
            continue

        try:
            with open(manifest_path, newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    if row.get("status") != "SUCCESS":
                        continue

                    study_uid = row.get("study_uid")
                    series_uid = row.get("series_uid")

                    if not study_uid or not series_uid:
                        continue

                    state[(study_uid, series_uid)] = row

        except Exception as exc:
            print(
                f"WARNING: could not read manifest "
                f"{manifest_path}: {exc}",
                file=sys.stderr,
            )

    return state


def append_manifest(row):
    """
    Append exactly one completed result and fsync it.

    This makes the run restart-safe even after a hard interruption.
    """

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = MANIFEST_PATH.exists()

    with open(
        MANIFEST_PATH,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        if not exists or MANIFEST_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------
# DICOM reconstruction
# ---------------------------------------------------------------------

def reconstruct_series(series_dir: Path):
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(series_dir)
    )

    if not files:
        raise RuntimeError(
            "GDCM found zero DICOM files"
        )

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)

    image = reader.Execute()

    return image, files


def validate_reconstruction(
    image,
    files,
    directory_files,
):
    if image.GetDimension() != 3:
        raise RuntimeError(
            f"Expected 3D image, got dimension={image.GetDimension()}"
        )

    size_xyz = tuple(
        int(x)
        for x in image.GetSize()
    )

    if any(x <= 0 for x in size_xyz):
        raise RuntimeError(
            f"Invalid reconstructed size={size_xyz}"
        )

    spacing_xyz = tuple(
        float(x)
        for x in image.GetSpacing()
    )

    origin_xyz = tuple(
        float(x)
        for x in image.GetOrigin()
    )

    direction = tuple(
        float(x)
        for x in image.GetDirection()
    )

    if not geometry_is_finite(
        spacing_xyz,
        origin_xyz,
        direction,
    ):
        raise RuntimeError(
            "Non-finite image geometry"
        )

    if any(x <= 0 for x in spacing_xyz):
        raise RuntimeError(
            f"Invalid spacing={spacing_xyz}"
        )

    if not direction_is_reasonable(direction):
        raise RuntimeError(
            f"Invalid/non-orthonormal direction={direction}"
        )

    if len(directory_files) != len(files):
        raise RuntimeError(
            "Directory/GDCM file-count mismatch: "
            f"directory={len(directory_files)} "
            f"gdcm={len(files)}"
        )

    array = sitk.GetArrayFromImage(image)

    if array.ndim != 3:
        raise RuntimeError(
            f"Expected ndim=3, got {array.ndim}"
        )

    if not is_numeric_supported(array.dtype):
        raise RuntimeError(
            f"Unsupported decoded dtype={array.dtype}"
        )

    if not np.all(np.isfinite(array)):
        raise RuntimeError(
            "Decoded pixel array contains non-finite values"
        )

    return (
        array,
        spacing_xyz,
        origin_xyz,
        direction,
    )


# ---------------------------------------------------------------------
# Existing-output validation
# ---------------------------------------------------------------------

def validate_existing_zarr(
    output_path: Path,
    row: dict,
) -> bool:
    """
    Lightweight validation for an existing SUCCESS output.

    We trust the 50 v2 outputs because they already passed complete
    pixel-for-pixel verification.

    For general resumability, verify:
    - Zarr readable
    - volume exists
    - shape
    - dtype
    - basic metadata
    - representative Z slices
    """

    try:
        root = zarr.open_group(
            str(output_path),
            mode="r",
        )

        if "volume" not in root:
            return False

        volume = root["volume"]

        expected_shape = (
            int(row["shape_z"]),
            int(row["shape_y"]),
            int(row["shape_x"]),
        )

        if tuple(volume.shape) != expected_shape:
            return False

        if np.dtype(volume.dtype) != np.dtype(row["dtype"]):
            return False

        for key in (
            "spacing_xyz",
            "origin_xyz",
            "direction",
        ):
            if key not in root.attrs:
                return False

        # Read representative slices to ensure the data is actually
        # readable, not merely that metadata exists.
        z = expected_shape[0]

        sample_indices = sorted(
            set(
                [
                    0,
                    z // 2,
                    z - 1,
                ]
            )
        )

        for zi in sample_indices:
            sample = np.asarray(
                volume[zi:zi + 1]
            )

            if sample.shape != (
                1,
                expected_shape[1],
                expected_shape[2],
            ):
                return False

        return True

    except Exception:
        return False


# ---------------------------------------------------------------------
# Zarr writing
# ---------------------------------------------------------------------

def write_zarr_atomic(
    output_path: Path,
    array: np.ndarray,
    study_uid: str,
    series_uid: str,
    n_dicom_files: int,
    spacing_xyz,
    origin_xyz,
    direction,
    source_dicom_path: str,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = Path(
        tempfile.mkdtemp(
            prefix=output_path.name + ".tmp.",
            dir=str(output_path.parent),
        )
    )

    try:
        root = zarr.open_group(
            str(tmp_path),
            mode="w",
            zarr_format=3,
        )

        root.attrs.update(
            {
                "conversion_version": CONVERSION_VERSION,
                "created_utc": utc_now(),

                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,

                "source_dicom_path": source_dicom_path,
                "number_of_dicom_files": int(
                    n_dicom_files
                ),

                "array_layout": "ZYX",
                "shape": [
                    int(x)
                    for x in array.shape
                ],
                "dtype": str(array.dtype),

                "spacing_xyz": [
                    float(x)
                    for x in spacing_xyz
                ],
                "origin_xyz": [
                    float(x)
                    for x in origin_xyz
                ],
                "direction": [
                    float(x)
                    for x in direction
                ],

                "zarr_format": 3,
                "zarr_chunks": list(
                    ZARR_CHUNKS
                ),

                "compression": {
                    "codec": "blosc",
                    "algorithm": "zstd",
                    "level": ZSTD_LEVEL,
                    "shuffle": "shuffle",
                },
            }
        )

        compressor = BloscCodec(
            cname="zstd",
            clevel=ZSTD_LEVEL,
            shuffle="shuffle",
        )

        root.create_array(
            "volume",
            shape=array.shape,
            dtype=array.dtype,
            chunks=ZARR_CHUNKS,
            compressors=[compressor],
            overwrite=False,
        )

        root["volume"][:] = array

        del root

        if output_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing output: "
                f"{output_path}"
            )

        tmp_path.rename(output_path)

    except Exception:
        shutil.rmtree(
            tmp_path,
            ignore_errors=True,
        )
        raise


# ---------------------------------------------------------------------
# Production verification
# ---------------------------------------------------------------------

def verify_zarr_production(
    output_path: Path,
    reference_array: np.ndarray,
    reference_spacing,
    reference_origin,
    reference_direction,
):
    root = zarr.open_group(
        str(output_path),
        mode="r",
    )

    if "volume" not in root:
        raise RuntimeError(
            "Missing volume array"
        )

    volume = root["volume"]

    if tuple(volume.shape) != tuple(
        reference_array.shape
    ):
        raise RuntimeError(
            f"Shape mismatch: "
            f"{volume.shape} vs "
            f"{reference_array.shape}"
        )

    if np.dtype(volume.dtype) != reference_array.dtype:
        raise RuntimeError(
            f"Dtype mismatch: "
            f"{volume.dtype} vs "
            f"{reference_array.dtype}"
        )

    z_spacing = np.asarray(
        root.attrs["spacing_xyz"],
        dtype=np.float64,
    )

    z_origin = np.asarray(
        root.attrs["origin_xyz"],
        dtype=np.float64,
    )

    z_direction = np.asarray(
        root.attrs["direction"],
        dtype=np.float64,
    )

    if not np.allclose(
        z_spacing,
        np.asarray(reference_spacing),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Spacing mismatch"
        )

    if not np.allclose(
        z_origin,
        np.asarray(reference_origin),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Origin mismatch"
        )

    if not np.allclose(
        z_direction,
        np.asarray(reference_direction),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Direction mismatch"
        )

    # Production integrity check:
    # compare several complete Z slices against the source array.
    z = reference_array.shape[0]

    sample_indices = sorted(
        set(
            [
                0,
                z // 2,
                z - 1,
            ]
        )
    )

    for zi in sample_indices:
        actual = np.asarray(
            volume[zi:zi + 1]
        )

        expected = reference_array[
            zi:zi + 1
        ]

        if not np.array_equal(
            actual,
            expected,
        ):
            raise RuntimeError(
                f"Pixel mismatch at Z slice {zi}"
            )


# ---------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------

def convert_one(
    study_uid: str,
    series_uid: str,
    series_dir: Path,
):
    start = time.perf_counter()

    output_path = output_path_for(
        study_uid,
        series_uid,
    )

    if output_path.exists():
        raise RuntimeError(
            f"Output already exists: {output_path}"
        )

    directory_files = regular_files(
        series_dir
    )

    if not directory_files:
        raise RuntimeError(
            "Series directory contains no regular files"
        )

    image, gdcm_files = reconstruct_series(
        series_dir
    )

    (
        array,
        spacing,
        origin,
        direction,
    ) = validate_reconstruction(
        image,
        gdcm_files,
        directory_files,
    )

    write_zarr_atomic(
        output_path=output_path,
        array=array,
        study_uid=study_uid,
        series_uid=series_uid,
        n_dicom_files=len(gdcm_files),
        spacing_xyz=spacing,
        origin_xyz=origin,
        direction=direction,
        source_dicom_path=str(series_dir),
    )

    verify_zarr_production(
        output_path=output_path,
        reference_array=array,
        reference_spacing=spacing,
        reference_origin=origin,
        reference_direction=direction,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "status": "SUCCESS",

        "n_directory_files": len(
            directory_files
        ),
        "n_dicom_files": len(
            gdcm_files
        ),

        "shape_z": int(array.shape[0]),
        "shape_y": int(array.shape[1]),
        "shape_x": int(array.shape[2]),

        "dtype": str(array.dtype),

        "spacing_x": float(spacing[0]),
        "spacing_y": float(spacing[1]),
        "spacing_z": float(spacing[2]),

        "origin_x": float(origin[0]),
        "origin_y": float(origin[1]),
        "origin_z": float(origin[2]),

        "conversion_seconds": elapsed,

        "output_path": str(
            output_path
        ),

        "error": "",
    }


def failed_row(
    study_uid,
    series_uid,
    error,
):
    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "status": "FAILED",

        "n_directory_files": "",
        "n_dicom_files": "",

        "shape_z": "",
        "shape_y": "",
        "shape_x": "",

        "dtype": "",

        "spacing_x": "",
        "spacing_y": "",
        "spacing_z": "",

        "origin_x": "",
        "origin_y": "",
        "origin_z": "",

        "conversion_seconds": "",

        "output_path": "",

        "error": error,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of discovered series. "
            "Default: all series."
        ),
    )

    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=MIN_FREE_GIB,
        help=(
            "Hard minimum free-space reserve. "
            f"Default: {MIN_FREE_GIB} GiB."
        ),
    )

    args = parser.parse_args()

    if not RAW_ROOT.exists():
        raise RuntimeError(
            f"RAW_ROOT does not exist: "
            f"{RAW_ROOT}"
        )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ZARR_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(CONVERSION_VERSION)
    print("=" * 72)
    print(f"RAW_ROOT:       {RAW_ROOT}")
    print(f"ZARR_ROOT:      {ZARR_ROOT}")
    print(f"MANIFEST:       {MANIFEST_PATH}")
    print(f"LEGACY:         {LEGACY_MANIFEST_PATH}")
    print(f"MIN FREE GiB:   {args.min_free_gib:.2f}")
    print(
        f"FREE SPACE GiB: "
        f"{free_gib(PROJECT_ROOT):.2f}"
    )
    print("=" * 72)

    if free_gib(PROJECT_ROOT) <= args.min_free_gib:
        raise RuntimeError(
            "Filesystem is already at or below "
            "the configured safety reserve."
        )

    series = discover_series(
        args.limit
    )

    print(
        f"Discovered {len(series):,} series"
    )

    existing = load_existing_state()

    print(
        f"Previously successful: "
        f"{len(existing):,}"
    )

    converted = 0
    skipped = 0
    failed = 0
    stopped = False

    for i, (
        study_uid,
        series_uid,
        series_dir,
    ) in enumerate(
        series,
        start=1,
    ):

        key = (
            study_uid,
            series_uid,
        )

        output_path = output_path_for(
            study_uid,
            series_uid,
        )

        # -------------------------------------------------------------
        # Resume existing successful work
        # -------------------------------------------------------------

        if key in existing and output_path.exists():

            print(
                f"[{i:,}/{len(series):,}] "
                f"SKIP existing verified "
                f"{series_uid[:24]}...",
                flush=True,
            )

            skipped += 1
            continue

        # -------------------------------------------------------------
        # Safety check before every new series
        # -------------------------------------------------------------

        current_free = free_gib(
            PROJECT_ROOT
        )

        if current_free <= args.min_free_gib:

            print()
            print("=" * 72)
            print("SAFE STOP: FREE SPACE RESERVE REACHED")
            print("=" * 72)
            print(
                f"FREE SPACE: "
                f"{current_free:.2f} GiB"
            )
            print(
                f"RESERVE:    "
                f"{args.min_free_gib:.2f} GiB"
            )
            print(
                "No further series will be converted."
            )
            print("=" * 72)

            stopped = True
            break

        print(
            f"[{i:,}/{len(series):,}] "
            f"FREE={current_free:.2f} GiB "
            f"{study_uid[:16]}.../"
            f"{series_uid[:16]}...",
            flush=True,
        )

        try:
            result = convert_one(
                study_uid,
                series_uid,
                series_dir,
            )

            append_manifest(result)

            converted += 1

            print(
                f"    SUCCESS "
                f"shape=("
                f"{result['shape_z']},"
                f"{result['shape_y']},"
                f"{result['shape_x']}) "
                f"dtype={result['dtype']} "
                f"time="
                f"{result['conversion_seconds']:.2f}s "
                f"free="
                f"{free_gib(PROJECT_ROOT):.2f} GiB",
                flush=True,
            )

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            row = failed_row(
                study_uid,
                series_uid,
                error,
            )

            append_manifest(row)

            failed += 1

            print(
                f"    FAILED: {error}",
                file=sys.stderr,
                flush=True,
            )

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    total = (
        converted
        + skipped
        + failed
    )

    print()
    print("=" * 72)
    print("PRODUCTION RUN COMPLETE")
    print("=" * 72)
    print(f"DISCOVERED:     {len(series):,}")
    print(f"CONVERTED:      {converted:,}")
    print(f"SKIPPED:        {skipped:,}")
    print(f"FAILED:         {failed:,}")
    print(
        f"FREE SPACE:     "
        f"{free_gib(PROJECT_ROOT):.2f} GiB"
    )
    print(
        f"RESERVE:        "
        f"{args.min_free_gib:.2f} GiB"
    )
    print(
        f"SAFE STOP:      "
        f"{stopped}"
    )
    print(
        f"MANIFEST:       "
        f"{MANIFEST_PATH}"
    )
    print("=" * 72)

    if stopped:
        print(
            "Run can be resumed safely."
        )

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
