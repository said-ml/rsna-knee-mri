#!/usr/bin/env python3

"""
RSNA Knee MRI
DATA-001: Full DICOM -> Zarr conversion

Purpose
-------
Convert every DICOM series in:

    data/raw/train_series/

into chunked Zarr v3 volumes:

    data/zarr/

The original DICOM dataset is NEVER modified.

Output structure
---------------
data/
├── raw/
│   └── train_series/
│       └── StudyInstanceUID/
│           └── SeriesInstanceUID/
│               └── *.dcm
│
├── zarr/
│   └── StudyInstanceUID/
│       └── SeriesInstanceUID.zarr/
│           ├── zarr.json
│           └── c/
│
└── reports/
    └── zarr_conversion_manifest.csv

Each Zarr volume contains:
    volume: (Z, Y, X)

Metadata contains:
    StudyInstanceUID
    SeriesInstanceUID
    shape
    dtype
    spacing
    origin
    direction
    source_dicom_path
    number_of_dicom_files
    conversion_version
    etc.

Important
---------
This converts ALL series, including series that may not ultimately be
used for model training. Later, the inventory can be used to select
training-relevant series.

Requirements
------------
pydicom
SimpleITK
numpy
pandas
zarr
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import zarr


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path("/workspace")

RAW_ROOT = PROJECT_ROOT / "data/raw/train_series"
ZARR_ROOT = PROJECT_ROOT / "data/zarr"
REPORT_ROOT = PROJECT_ROOT / "data/reports"

MANIFEST_PATH = REPORT_ROOT / "zarr_conversion_manifest.csv"

# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

# Zarr chunk layout for (Z, Y, X).
#
# 8 slices per chunk is a good starting point for 3D MRI crops.
ZARR_CHUNKS = (8, 256, 256)

# Zarr v3 compression.
#
# Blosc + Zstd gives strong compression while remaining fast.
ZSTD_LEVEL = 3

# Number of conversion processes.
#
# Start conservatively. DICOM decoding + filesystem I/O are expensive.
# 4 is appropriate for your 8-core/16-thread CPU.
WORKERS = 1

# Minimum free space required before starting another conversion.
#
# We do NOT want the filesystem getting close to 100%.
MIN_FREE_GB = 100.0

# Conversion version.
CONVERSION_VERSION = "DATA-001-ZARR-v1"

# --------------------------------------------------------------------------
# Behavior
# --------------------------------------------------------------------------

# If True, existing valid Zarr outputs are skipped.
RESUME = True

# If True, failed output directories are removed before retrying.
CLEAN_FAILED_OUTPUT = True

# If True, verify the written Zarr volume by reading shape/dtype.
VERIFY_OUTPUT = True


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SeriesTask:
    study_uid: str
    series_uid: str
    series_dir: str


@dataclass
class ConversionResult:
    study_uid: str
    series_uid: str

    status: str

    n_dicom_files: int

    shape_z: int
    shape_y: int
    shape_x: int

    dtype: str

    spacing_z: float
    spacing_y: float
    spacing_x: float

    origin_x: float
    origin_y: float
    origin_z: float

    conversion_seconds: float

    output_path: str

    error: str = ""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_space_gb(path: Path) -> float:
    stat = os.statvfs(path)
    free_bytes = stat.f_bavail * stat.f_frsize
    return free_bytes / (1024 ** 3)


def ensure_directories() -> None:
    ZARR_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def remove_path(path: Path) -> None:
    """
    Remove a Zarr directory recursively.
    """
    if not path.exists():
        return

    if path.is_dir():
        import shutil
        shutil.rmtree(path)
    else:
        path.unlink()


# ============================================================================
# DICOM DISCOVERY
# ============================================================================

def discover_series() -> list[SeriesTask]:
    """
    Discover:

        StudyInstanceUID / SeriesInstanceUID

    directly from the filesystem.

    We assume the competition structure:

        train_series/
            STUDY/
                SERIES/
                    DICOM files
    """

    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw DICOM directory does not exist:\n{RAW_ROOT}"
        )

    tasks: list[SeriesTask] = []

    study_dirs = sorted(
        p for p in RAW_ROOT.iterdir()
        if p.is_dir()
    )

    print(f"[DISCOVERY] Studies found: {len(study_dirs):,}")

    for study_dir in study_dirs:

        study_uid = study_dir.name

        series_dirs = sorted(
            p for p in study_dir.iterdir()
            if p.is_dir()
        )

        for series_dir in series_dirs:

            series_uid = series_dir.name

            tasks.append(
                SeriesTask(
                    study_uid=study_uid,
                    series_uid=series_uid,
                    series_dir=str(series_dir),
                )
            )

    print(f"[DISCOVERY] Series found: {len(tasks):,}")

    return tasks


# ============================================================================
# DICOM SERIES READING
# ============================================================================

def get_dicom_files(series_dir: Path) -> list[str]:
    """
    Return DICOM files.

    The competition directory should contain files directly inside the
    SeriesInstanceUID directory.
    """

    files = [
        p for p in series_dir.iterdir()
        if p.is_file()
    ]

    # Keep deterministic ordering.
    files.sort()

    return [str(p) for p in files]


def read_series_with_sitk(
    series_dir: Path,
) -> tuple[np.ndarray, sitk.Image, int]:
    """
    Read one DICOM series using SimpleITK.

    Returns
    -------
    volume:
        NumPy array with shape (Z, Y, X)

    image:
        SimpleITK image containing spatial metadata.

    n_files:
        Number of source DICOM files.
    """

    dicom_files = get_dicom_files(series_dir)

    if not dicom_files:
        raise RuntimeError(
            f"No files found in {series_dir}"
        )

    reader = sitk.ImageSeriesReader()

    reader.SetFileNames(dicom_files)

    # Keep metadata from the DICOM series.
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image)

    if volume.ndim != 3:
        raise RuntimeError(
            f"Expected 3D volume, got shape {volume.shape}"
        )

    return volume, image, len(dicom_files)


# ============================================================================
# DICOM METADATA
# ============================================================================

def read_representative_metadata(
    series_dir: Path,
) -> dict[str, Any]:
    """
    Read metadata from one DICOM file without loading pixel data.
    """

    dicom_files = get_dicom_files(series_dir)

    if not dicom_files:
        return {}

    ds = pydicom.dcmread(
        dicom_files[0],
        stop_before_pixels=True,
    )

    metadata: dict[str, Any] = {}

    fields = [
        "PatientSex",
        "Modality",
        "SeriesDescription",
        "ProtocolName",
        "Manufacturer",
        "ManufacturerModelName",
        "MagneticFieldStrength",
        "SliceThickness",
        "SpacingBetweenSlices",
        "Rows",
        "Columns",
        "PixelSpacing",
        "ImageOrientationPatient",
        "StudyInstanceUID",
        "SeriesInstanceUID",
    ]

    for field in fields:

        if hasattr(ds, field):

            value = getattr(ds, field)

            # Convert pydicom values to JSON-friendly objects.
            if isinstance(value, (list, tuple)):
                value = [
                    str(x)
                    for x in value
                ]
            else:
                try:
                    json.dumps(value)
                except TypeError:
                    value = str(value)

            metadata[field] = value

    return metadata


# ============================================================================
# ZARR WRITING
# ============================================================================

def create_zarr_volume(
    output_path: Path,
    volume: np.ndarray,
    image: sitk.Image,
    study_uid: str,
    series_uid: str,
    source_path: Path,
    n_dicom_files: int,
    dicom_metadata: dict[str, Any],
) -> None:
    """
    Create one Zarr v3 volume.

    volume shape:
        (Z, Y, X)
    """

    if output_path.exists():

        if CLEAN_FAILED_OUTPUT:
            remove_path(output_path)
        else:
            raise FileExistsError(
                f"Output already exists: {output_path}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Spatial metadata
    # ----------------------------------------------------------------------

    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    # SimpleITK returns spacing as:
    # (X, Y, Z)
    #
    # NumPy volume is:
    # (Z, Y, X)
    #
    # Store both explicitly to avoid ambiguity.
    spacing_xyz = [
        float(x)
        for x in spacing
    ]

    origin_xyz = [
        float(x)
        for x in origin
    ]

    direction_matrix = [
        float(x)
        for x in direction
    ]

    # ----------------------------------------------------------------------
    # Normalize dtype
    # ----------------------------------------------------------------------

    # Preserve the original pixel dtype where possible.
    #
    # MRI DICOM commonly becomes int16 after SimpleITK conversion.
    if volume.dtype == np.int64:
        volume = volume.astype(np.int32)

    elif volume.dtype == np.uint64:
        volume = volume.astype(np.uint32)

    # ----------------------------------------------------------------------
    # Create Zarr v3
    # ----------------------------------------------------------------------

    compressor = zarr.codecs.BloscCodec(
        cname="zstd",
        clevel=ZSTD_LEVEL,
        shuffle="shuffle",
    )

    root = zarr.open_group(
        str(output_path),
        mode="w",
        zarr_format=3,
    )

    # Root metadata.
    root.attrs.update(
        {
            "conversion_version": CONVERSION_VERSION,
            "created_utc": utc_now(),

            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,

            "source_dicom_path": str(source_path),
            "number_of_dicom_files": int(n_dicom_files),

            "array_layout": "ZYX",
            "shape": [
                int(x)
                for x in volume.shape
            ],

            "dtype": str(volume.dtype),

            "spacing_xyz": spacing_xyz,
            "origin_xyz": origin_xyz,
            "direction": direction_matrix,

            "zarr_format": 3,
            "zarr_chunks": list(ZARR_CHUNKS),

            "compression": {
                "codec": "blosc",
                "algorithm": "zstd",
                "level": ZSTD_LEVEL,
                "shuffle": "shuffle",
            },

            "dicom_metadata": dicom_metadata,
        }
    )

    # ----------------------------------------------------------------------
    # Main volume
    # ----------------------------------------------------------------------

    root.create_array(
        "volume",
        shape=volume.shape,
        dtype=volume.dtype,
        chunks=ZARR_CHUNKS,
        compressors=compressor,
        overwrite=True,
    )

    root["volume"][:] = volume


# ============================================================================
# VERIFY
# ============================================================================

def verify_zarr(
    output_path: Path,
    expected_shape: tuple[int, int, int],
    expected_dtype: str,
) -> None:

    root = zarr.open_group(
        str(output_path),
        mode="r",
    )

    volume = root["volume"]

    if tuple(volume.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"Shape verification failed: "
            f"{volume.shape} != {expected_shape}"
        )

    if str(volume.dtype) != expected_dtype:
        raise RuntimeError(
            f"Dtype verification failed: "
            f"{volume.dtype} != {expected_dtype}"
        )

    # Read a tiny piece to ensure actual chunk data is readable.
    if all(x > 0 for x in volume.shape):
        _ = volume[
            0:1,
            0:min(8, volume.shape[1]),
            0:min(8, volume.shape[2]),
        ]


# ============================================================================
# ONE SERIES CONVERSION
# ============================================================================

def convert_one(task: SeriesTask) -> ConversionResult:

    start = time.perf_counter()

    output_path = (
        ZARR_ROOT
        / task.study_uid
        / f"{task.series_uid}.zarr"
    )

    try:

        # --------------------------------------------------------------
        # Resume
        # --------------------------------------------------------------

        if RESUME and output_path.exists():

            try:

                root = zarr.open_group(
                    str(output_path),
                    mode="r",
                )

                volume = root["volume"]

                shape = tuple(
                    int(x)
                    for x in volume.shape
                )

                dtype = str(volume.dtype)

                elapsed = time.perf_counter() - start

                return ConversionResult(
                    study_uid=task.study_uid,
                    series_uid=task.series_uid,
                    status="SKIPPED_EXISTS",
                    n_dicom_files=int(
                        root.attrs.get(
                            "number_of_dicom_files",
                            0,
                        )
                    ),
                    shape_z=shape[0],
                    shape_y=shape[1],
                    shape_x=shape[2],
                    dtype=dtype,
                    spacing_z=0.0,
                    spacing_y=0.0,
                    spacing_x=0.0,
                    origin_x=0.0,
                    origin_y=0.0,
                    origin_z=0.0,
                    conversion_seconds=elapsed,
                    output_path=str(output_path),
                )

            except Exception:

                if CLEAN_FAILED_OUTPUT:
                    remove_path(output_path)

        # --------------------------------------------------------------
        # Read DICOM
        # --------------------------------------------------------------

        series_dir = Path(task.series_dir)

        volume, image, n_files = read_series_with_sitk(
            series_dir
        )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        dicom_metadata = read_representative_metadata(
            series_dir
        )

        # --------------------------------------------------------------
        # Write Zarr
        # --------------------------------------------------------------

        create_zarr_volume(
            output_path=output_path,
            volume=volume,
            image=image,
            study_uid=task.study_uid,
            series_uid=task.series_uid,
            source_path=series_dir,
            n_dicom_files=n_files,
            dicom_metadata=dicom_metadata,
        )

        # --------------------------------------------------------------
        # Verify
        # --------------------------------------------------------------

        if VERIFY_OUTPUT:

            verify_zarr(
                output_path=output_path,
                expected_shape=tuple(
                    int(x)
                    for x in volume.shape
                ),
                expected_dtype=str(volume.dtype),
            )

        elapsed = time.perf_counter() - start

        spacing = image.GetSpacing()
        origin = image.GetOrigin()

        return ConversionResult(
            study_uid=task.study_uid,
            series_uid=task.series_uid,
            status="SUCCESS",

            n_dicom_files=n_files,

            shape_z=int(volume.shape[0]),
            shape_y=int(volume.shape[1]),
            shape_x=int(volume.shape[2]),

            dtype=str(volume.dtype),

            spacing_z=float(spacing[2]),
            spacing_y=float(spacing[1]),
            spacing_x=float(spacing[0]),

            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            origin_z=float(origin[2]),

            conversion_seconds=elapsed,

            output_path=str(output_path),
        )

    except Exception as exc:

        elapsed = time.perf_counter() - start

        error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            f"[FAILED] "
            f"{task.study_uid} / "
            f"{task.series_uid}\n"
            f"         {error}"
        )

        traceback.print_exc()

        if CLEAN_FAILED_OUTPUT:
            try:
                remove_path(output_path)
            except Exception:
                pass

        return ConversionResult(
            study_uid=task.study_uid,
            series_uid=task.series_uid,
            status="FAILED",

            n_dicom_files=0,

            shape_z=0,
            shape_y=0,
            shape_x=0,

            dtype="",

            spacing_z=0.0,
            spacing_y=0.0,
            spacing_x=0.0,

            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,

            conversion_seconds=elapsed,

            output_path=str(output_path),

            error=error,
        )


# ============================================================================
# MANIFEST
# ============================================================================

MANIFEST_COLUMNS = [
    "study_uid",
    "series_uid",
    "status",
    "n_dicom_files",

    "shape_z",
    "shape_y",
    "shape_x",

    "dtype",

    "spacing_z",
    "spacing_y",
    "spacing_x",

    "origin_x",
    "origin_y",
    "origin_z",

    "conversion_seconds",

    "output_path",
    "error",
]


def append_manifest(result: ConversionResult) -> None:

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = MANIFEST_PATH.exists()

    with MANIFEST_PATH.open(
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=MANIFEST_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            asdict(result)
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 70)
    print("RSNA KNEE MRI")
    print("DATA-001: FULL DICOM -> ZARR")
    print("=" * 70)

    print(f"Raw DICOM:      {RAW_ROOT}")
    print(f"Zarr output:    {ZARR_ROOT}")
    print(f"Manifest:       {MANIFEST_PATH}")
    print(f"Workers:        {WORKERS}")
    print(f"Chunks:         {ZARR_CHUNKS}")
    print(f"Compression:    Blosc/Zstd level {ZSTD_LEVEL}")
    print(f"Resume:         {RESUME}")
    print()

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------

    if not RAW_ROOT.exists():

        raise FileNotFoundError(
            f"Raw DICOM root does not exist:\n{RAW_ROOT}"
        )

    ensure_directories()

    # ------------------------------------------------------------------
    # Disk safety
    # ------------------------------------------------------------------

    free_gb = free_space_gb(PROJECT_ROOT)

    print(
        f"[DISK] Free space: "
        f"{free_gb:.1f} GiB"
    )

    if free_gb < MIN_FREE_GB:

        raise RuntimeError(
            f"Only {free_gb:.1f} GiB free.\n"
            f"Minimum required: {MIN_FREE_GB:.1f} GiB.\n"
            f"Refusing to start conversion."
        )

    # ------------------------------------------------------------------
    # Discover
    # ------------------------------------------------------------------

    tasks = discover_series()

    if not tasks:
        raise RuntimeError(
            "No DICOM series found."
        )

    # ------------------------------------------------------------------
    # Estimate existing outputs
    # ------------------------------------------------------------------

    existing = 0

    for task in tasks:

        output_path = (
            ZARR_ROOT
            / task.study_uid
            / f"{task.series_uid}.zarr"
        )

        if output_path.exists():
            existing += 1

    print(
        f"[DISCOVERY] Existing Zarr outputs: "
        f"{existing:,}"
    )

    remaining = len(tasks) - existing

    print(
        f"[DISCOVERY] Remaining conversions: "
        f"{remaining:,}"
    )

    print()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    completed = 0
    success = 0
    skipped = 0
    failed = 0

    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Important:
    #
    # ProcessPoolExecutor is used because SimpleITK/DICOM decoding is
    # CPU-heavy and we want independent worker processes.
    # ------------------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = {
            executor.submit(
                convert_one,
                task,
            ): task
            for task in tasks
        }

        for future in as_completed(futures):

            task = futures[future]

            try:
                result = future.result()

            except Exception as exc:

                result = ConversionResult(
                    study_uid=task.study_uid,
                    series_uid=task.series_uid,
                    status="FAILED_WORKER",

                    n_dicom_files=0,

                    shape_z=0,
                    shape_y=0,
                    shape_x=0,

                    dtype="",

                    spacing_z=0.0,
                    spacing_y=0.0,
                    spacing_x=0.0,

                    origin_x=0.0,
                    origin_y=0.0,
                    origin_z=0.0,

                    conversion_seconds=0.0,

                    output_path="",

                    error=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

            append_manifest(result)

            completed += 1

            if result.status == "SUCCESS":
                success += 1

            elif result.status == "SKIPPED_EXISTS":
                skipped += 1

            else:
                failed += 1

            if completed % 25 == 0:

                elapsed = (
                    time.perf_counter()
                    - total_start
                )

                rate = (
                    completed / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining_count = (
                    len(tasks) - completed
                )

                eta_hours = (
                    remaining_count / rate / 3600
                    if rate > 0
                    else 0
                )

                free_now = free_space_gb(
                    PROJECT_ROOT
                )

                print()
                print(
                    f"[PROGRESS] "
                    f"{completed:,}/{len(tasks):,}"
                )

                print(
                    f"           "
                    f"success={success:,} "
                    f"skipped={skipped:,} "
                    f"failed={failed:,}"
                )

                print(
                    f"           "
                    f"rate={rate:.2f} series/s"
                )

                print(
                    f"           "
                    f"ETA={eta_hours:.2f} h"
                )

                print(
                    f"           "
                    f"free={free_now:.1f} GiB"
                )

                # Emergency disk protection.
                if free_now < MIN_FREE_GB:

                    print()
                    print(
                        "[STOP] Free disk space below "
                        f"{MIN_FREE_GB:.1f} GiB."
                    )

                    print(
                        "[STOP] Conversion will terminate."
                    )

                    executor.shutdown(
                        wait=False,
                        cancel_futures=True,
                    )

                    raise RuntimeError(
                        "Disk safety limit reached."
                    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    elapsed_total = (
        time.perf_counter()
        - total_start
    )

    free_final = free_space_gb(
        PROJECT_ROOT
    )

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"Total series:      {len(tasks):,}"
    )

    print(
        f"Successful:        {success:,}"
    )

    print(
        f"Already existed:   {skipped:,}"
    )

    print(
        f"Failed:            {failed:,}"
    )

    print(
        f"Total time:        "
        f"{elapsed_total / 3600:.2f} hours"
    )

    print(
        f"Free disk:         "
        f"{free_final:.1f} GiB"
    )

    print()
    print(
        f"Manifest:"
    )

    print(
        MANIFEST_PATH
    )

    print()
    print(
        "Original DICOM was not modified."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()