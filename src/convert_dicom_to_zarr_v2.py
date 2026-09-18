#!/usr/bin/env python3

"""
DATA-001-ZARR-v2

Conservative DICOM -> Zarr v3 converter.

Design principles:
- DICOM remains immutable source of truth.
- SimpleITK/GDCM performs DICOM series reconstruction.
- Geometry comes from the reconstructed SimpleITK image.
- Decoded pixel dtype is preserved.
- Every output is verified after writing.
- Failed conversions never become SUCCESS.
- Output is first written to a temporary directory and atomically renamed.
- No multiprocessing in v2 initial validation phase.
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

MANIFEST_PATH = REPORT_ROOT / "zarr_conversion_manifest_v2.csv"

CONVERSION_VERSION = "DATA-001-ZARR-v2"

# Conservative initial chunking.
ZARR_CHUNKS = (8, 256, 256)

ZSTD_LEVEL = 3

# Never allow the filesystem to approach full capacity.
MIN_FREE_GIB = 50.0


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


def is_numeric_supported(dtype: np.dtype) -> bool:
    return (
        np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    )


def geometry_is_finite(
    spacing,
    origin,
    direction,
) -> bool:
    return (
        np.all(np.isfinite(np.asarray(spacing, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(origin, dtype=np.float64)))
        and np.all(np.isfinite(np.asarray(direction, dtype=np.float64)))
    )


def direction_is_reasonable(direction, atol=1e-4) -> bool:
    """
    Direction is a 3x3 row-major matrix.
    Check approximately orthonormal.
    """
    d = np.asarray(direction, dtype=np.float64).reshape(3, 3)

    gram = d @ d.T

    return (
        np.allclose(gram, np.eye(3), atol=atol)
        and abs(abs(np.linalg.det(d)) - 1.0) <= 1e-3
    )


def discover_series(limit: int | None):
    """
    Deterministic study/series discovery.

    Returns:
        [(study_uid, series_uid, series_dir), ...]
    """
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


def regular_files(path: Path):
    return sorted(
        p for p in path.iterdir()
        if p.is_file()
    )


def reconstruct_series(series_dir: Path):
    """
    Reconstruct a DICOM series through GDCM/SimpleITK.

    Returns:
        image, files
    """
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(series_dir)
    )

    if not files:
        raise RuntimeError("GDCM found zero DICOM files")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)

    image = reader.Execute()

    return image, files


def validate_reconstruction(
    image,
    files,
    directory_files,
):
    """
    Hard validation before writing Zarr.
    """
    if image.GetDimension() != 3:
        raise RuntimeError(
            f"Expected 3D image, got dimension={image.GetDimension()}"
        )

    size_xyz = tuple(int(x) for x in image.GetSize())

    if any(x <= 0 for x in size_xyz):
        raise RuntimeError(
            f"Invalid reconstructed size={size_xyz}"
        )

    spacing_xyz = tuple(float(x) for x in image.GetSpacing())
    origin_xyz = tuple(float(x) for x in image.GetOrigin())
    direction = tuple(float(x) for x in image.GetDirection())

    if not geometry_is_finite(
        spacing_xyz,
        origin_xyz,
        direction,
    ):
        raise RuntimeError("Non-finite image geometry")

    if any(x <= 0 for x in spacing_xyz):
        raise RuntimeError(
            f"Invalid spacing={spacing_xyz}"
        )

    if not direction_is_reasonable(direction):
        raise RuntimeError(
            f"Invalid/non-orthonormal direction={direction}"
        )

    # GDCM-selected files should normally account for all files in a
    # clean SeriesInstanceUID directory. We reject unexpected extras
    # rather than silently ignoring them.
    if len(directory_files) != len(files):
        raise RuntimeError(
            "Directory/GDCM file-count mismatch: "
            f"directory={len(directory_files)} "
            f"gdcm={len(files)}"
        )

    array = sitk.GetArrayFromImage(image)

    if array.ndim != 3:
        raise RuntimeError(
            f"Expected NumPy array ndim=3, got {array.ndim}"
        )

    if not is_numeric_supported(array.dtype):
        raise RuntimeError(
            f"Unsupported decoded dtype={array.dtype}"
        )

    if not np.all(np.isfinite(array)):
        raise RuntimeError(
            "Decoded pixel array contains non-finite values"
        )

    return array, spacing_xyz, origin_xyz, direction


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
    """
    Write to a temporary sibling directory and rename only after
    successful completion.
    """

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
                "number_of_dicom_files": int(n_dicom_files),

                "array_layout": "ZYX",
                "shape": [int(x) for x in array.shape],
                "dtype": str(array.dtype),

                "spacing_xyz": [
                    float(x) for x in spacing_xyz
                ],
                "origin_xyz": [
                    float(x) for x in origin_xyz
                ],
                "direction": [
                    float(x) for x in direction
                ],

                "zarr_format": 3,
                "zarr_chunks": list(ZARR_CHUNKS),

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

        # Explicitly flush/close through object lifetime before rename.
        del root

        if output_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing output: {output_path}"
            )

        tmp_path.rename(output_path)

    except Exception:
        shutil.rmtree(
            tmp_path,
            ignore_errors=True,
        )
        raise


def verify_zarr(
    output_path: Path,
    reference_array: np.ndarray,
    reference_spacing,
    reference_origin,
    reference_direction,
):
    """
    Full verification for the validation phase.

    We intentionally compare the entire array for the first 50-series
    acceptance test.
    """

    root = zarr.open_group(
        str(output_path),
        mode="r",
    )

    if "volume" not in root:
        raise RuntimeError(
            "Zarr verification failed: missing volume"
        )

    volume = root["volume"]

    # Metadata checks.
    expected_shape = tuple(
        int(x) for x in reference_array.shape
    )

    if tuple(volume.shape) != expected_shape:
        raise RuntimeError(
            f"Shape mismatch: "
            f"zarr={volume.shape}, "
            f"reference={expected_shape}"
        )

    if np.dtype(volume.dtype) != reference_array.dtype:
        raise RuntimeError(
            f"Dtype mismatch: "
            f"zarr={volume.dtype}, "
            f"reference={reference_array.dtype}"
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
            f"Spacing mismatch: "
            f"zarr={z_spacing}, "
            f"reference={reference_spacing}"
        )

    if not np.allclose(
        z_origin,
        np.asarray(reference_origin),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"Origin mismatch: "
            f"zarr={z_origin}, "
            f"reference={reference_origin}"
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

    # Full pixel equality.
    zarr_array = np.asarray(volume[:])

    if not np.array_equal(
        zarr_array,
        reference_array,
    ):
        raise RuntimeError(
            "Pixel data mismatch after Zarr round-trip"
        )


# ---------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------

def convert_one(
    study_uid: str,
    series_uid: str,
    series_dir: Path,
):
    start = time.perf_counter()

    output_path = (
        ZARR_ROOT
        / study_uid
        / f"{series_uid}.zarr"
    )

    directory_files = regular_files(series_dir)

    if not directory_files:
        raise RuntimeError(
            "Series directory contains no regular files"
        )

    image, gdcm_files = reconstruct_series(
        series_dir
    )

    array, spacing, origin, direction = (
        validate_reconstruction(
            image,
            gdcm_files,
            directory_files,
        )
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

    # Full verification is intentional for the 50-series validation run.
    verify_zarr(
        output_path=output_path,
        reference_array=array,
        reference_spacing=spacing,
        reference_origin=origin,
        reference_direction=direction,
    )

    elapsed = time.perf_counter() - start

    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "status": "SUCCESS",

        "n_directory_files": len(directory_files),
        "n_dicom_files": len(gdcm_files),

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
        "output_path": str(output_path),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of series to convert. Default: 50",
    )

    args = parser.parse_args()

    if not RAW_ROOT.exists():
        raise RuntimeError(
            f"RAW_ROOT does not exist: {RAW_ROOT}"
        )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ZARR_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(CONVERSION_VERSION)
    print("=" * 70)
    print(f"RAW_ROOT:       {RAW_ROOT}")
    print(f"ZARR_ROOT:      {ZARR_ROOT}")
    print(f"MANIFEST:       {MANIFEST_PATH}")
    print(f"LIMIT:          {args.limit}")
    print(f"FREE SPACE GiB: {free_gib(PROJECT_ROOT):.2f}")
    print("=" * 70)

    if free_gib(PROJECT_ROOT) < MIN_FREE_GIB:
        raise RuntimeError(
            f"Insufficient free space: "
            f"{free_gib(PROJECT_ROOT):.2f} GiB"
        )

    series = discover_series(args.limit)

    print(
        f"Discovered {len(series)} series for this run"
    )

    fieldnames = [
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

    rows = []

    for i, (study_uid, series_uid, series_dir) in enumerate(
        series,
        start=1,
    ):
        print(
            f"[{i:03d}/{len(series):03d}] "
            f"{study_uid[:18]}... "
            f"{series_uid[:18]}...",
            flush=True,
        )

        try:
            result = convert_one(
                study_uid,
                series_uid,
                series_dir,
            )

            rows.append(result)

            print(
                f"    SUCCESS "
                f"shape=({result['shape_z']},"
                f"{result['shape_y']},"
                f"{result['shape_x']}) "
                f"dtype={result['dtype']} "
                f"time={result['conversion_seconds']:.2f}s",
                flush=True,
            )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

            result = {
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

            rows.append(result)

            print(
                f"    FAILED: {error}",
                file=sys.stderr,
                flush=True,
            )

    with open(
        MANIFEST_PATH,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    success = sum(
        r["status"] == "SUCCESS"
        for r in rows
    )

    failed = len(rows) - success

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"SUCCESS:       {success}")
    print(f"FAILED:        {failed}")
    print(f"MANIFEST:      {MANIFEST_PATH}")
    print(
        f"FREE SPACE:    "
        f"{free_gib(PROJECT_ROOT):.2f} GiB"
    )
    print("=" * 70)

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
