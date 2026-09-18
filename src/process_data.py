
#!/usr/bin/env python3

"""
DATA-001: DICOM vs NIfTI vs Zarr benchmark

Pipeline:
    DICOM
      |
      +-- select representative studies
      |
      +-- DICOM -> NIfTI
      |
      +-- DICOM -> Zarr
      |
      +-- benchmark
            - storage
            - sequential read
            - random crop read
            - workers: 0, 1, 2, 4, 8
            - preprocessing throughput
            - end-to-end DataLoader throughput

First benchmark:
    50 studies

IMPORTANT:
    This is a controlled benchmark, not the final production pipeline.
    Start with 50 studies before scaling to the full dataset.
"""

from pathlib import Path
import csv
import random
import shutil
import time
import statistics
import os

import numpy as np
import pydicom
import SimpleITK as sitk
import zarr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

RAW_ROOT = Path("/workspace/data/raw/train_series")
BENCHMARK_ROOT = Path("/workspace/data/benchmark")

DICOM_SAMPLE_ROOT = BENCHMARK_ROOT / "sample_dicom"
NIFTI_ROOT = BENCHMARK_ROOT / "nifti"
ZARR_ROOT = BENCHMARK_ROOT / "zarr"

REPORT_ROOT = Path(
    "/workspace/data/reports"
)

NUM_STUDIES = 50

RANDOM_SEED = 42

WORKER_COUNTS = [0, 1, 2, 4, 8]

READ_REPEATS = 5

RANDOM_CROP_REPEATS = 100

MAX_DATALOADER_SAMPLES = 50

# Zarr baseline chunk size.
# Do NOT optimize this yet.
ZARR_CHUNKS = (8, 256, 256)

# Representative preprocessing target.
TARGET_SHAPE = (64, 128, 128)

# Random crop.
CROP_SHAPE = (16, 128, 128)


# ============================================================
# SETUP
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

for directory in [
    BENCHMARK_ROOT,
    DICOM_SAMPLE_ROOT,
    NIFTI_ROOT,
    ZARR_ROOT,
    REPORT_ROOT,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITY
# ============================================================

def directory_size(path):
    """Return total size of all files below path in bytes."""

    total = 0

    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size

    return total


def format_gb(num_bytes):
    return num_bytes / (1024 ** 3)


def format_mb(num_bytes):
    return num_bytes / (1024 ** 2)


# ============================================================
# STEP 1
# SELECT REPRESENTATIVE STUDIES
# ============================================================

def select_studies(num_studies=NUM_STUDIES):
    """
    Select complete StudyInstanceUID directories.

    NOTE:
    This is a random baseline selection.
    DATA-001 should eventually use stratified sampling based
    on acquisition metadata.
    """

    studies = [
        p for p in RAW_ROOT.iterdir()
        if p.is_dir()
    ]

    if not studies:
        raise RuntimeError(
            f"No study directories found in {RAW_ROOT}"
        )

    selected = random.sample(
        studies,
        min(num_studies, len(studies))
    )

    print(
        f"[SELECT] Found {len(studies)} studies."
    )

    print(
        f"[SELECT] Selected {len(selected)} studies."
    )

    return selected


# ============================================================
# STEP 2
# COPY SAMPLE DICOM STUDIES
# ============================================================

def copy_sample_studies(studies):
    """
    Copy selected studies into benchmark/sample_dicom.

    This deliberately creates a small benchmark dataset.
    """

    copied = []

    for study in studies:

        destination = DICOM_SAMPLE_ROOT / study.name

        if destination.exists():
            print(
                f"[SAMPLE] Already exists: {study.name}"
            )
        else:

            print(
                f"[SAMPLE] Copying: {study.name}"
            )

            shutil.copytree(
                study,
                destination
            )

        copied.append(destination)

    print(
        f"[SAMPLE] Completed {len(copied)} studies."
    )

    return copied


# ============================================================
# STEP 3
# DICOM SERIES DISCOVERY
# ============================================================

def discover_series(study_dir):
    """
    Return series directories containing DICOM files.
    """

    series = []

    for path in study_dir.iterdir():

        if not path.is_dir():
            continue

        dcm_files = list(path.glob("*.dcm"))

        if dcm_files:
            series.append(path)

    return series


# ============================================================
# STEP 4
# LOAD DICOM SERIES
# ============================================================

def load_dicom_series(series_dir):
    """
    Load one DICOM series into a [Z, Y, X] NumPy array.

    Slices are sorted by ImagePositionPatient when available.
    """

    files = list(
        Path(series_dir).glob("*.dcm")
    )

    slices = []

    for f in files:

        try:
            ds = pydicom.dcmread(f)

        except Exception as e:

            print(
                f"[DICOM] Failed: {f}"
            )

            continue

        if hasattr(
            ds,
            "ImagePositionPatient"
        ):

            slices.append(ds)

    if not slices:

        raise RuntimeError(
            f"No readable positioned DICOM slices: "
            f"{series_dir}"
        )

    slices.sort(
        key=lambda x: float(
            x.ImagePositionPatient[2]
        )
    )

    volume = np.stack(
        [
            s.pixel_array
            for s in slices
        ],
        axis=0
    )

    return volume, slices


# ============================================================
# STEP 5
# DICOM -> NIFTI
# ============================================================

def dicom_to_nifti(
    series_dir,
    output_path
):
    """
    Convert a DICOM series to compressed NIfTI.
    """

    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(
        str(series_dir)
    )

    if not series_ids:

        raise RuntimeError(
            f"No DICOM series detected: {series_dir}"
        )

    # Benchmark first detected series.
    series_id = series_ids[0]

    files = reader.GetGDCMSeriesFileNames(
        str(series_dir),
        series_id
    )

    reader.SetFileNames(files)

    image = reader.Execute()

    sitk.WriteImage(
        image,
        str(output_path),
        useCompression=True
    )

    return image


# ============================================================
# STEP 6
# DICOM -> ZARR
# ============================================================

def save_zarr(
    volume,
    output_path
):
    """
    Save a NumPy volume as a Zarr array.
    """

    root = zarr.open(
        str(output_path),
        mode="w"
    )

    root.create_array(
        "volume",
        data=volume,
        chunks=ZARR_CHUNKS
    )

    return root


# ============================================================
# STEP 7
# BUILD CONVERSIONS
# ============================================================

def build_representations(studies):
    """
    For every selected study:
        select the first available series
        DICOM -> NIfTI
        DICOM -> Zarr
    """

    nifti_paths = []
    zarr_paths = []

    for study in studies:

        series_list = discover_series(
            study
        )

        if not series_list:

            print(
                f"[CONVERT] No series: {study.name}"
            )

            continue

        # Baseline:
        # first available series.
        series_dir = series_list[0]

        study_id = study.name

        nifti_path = (
            NIFTI_ROOT /
            f"{study_id}.nii.gz"
        )

        zarr_path = (
            ZARR_ROOT /
            f"{study_id}.zarr"
        )

        # -----------------------------
        # NIfTI
        # -----------------------------

        if not nifti_path.exists():

            print(
                f"[NIFTI] {study_id}"
            )

            start = time.perf_counter()

            try:

                image = dicom_to_nifti(
                    series_dir,
                    nifti_path
                )

            except Exception as e:

                print(
                    f"[NIFTI] FAILED: {e}"
                )

                continue

            elapsed = (
                time.perf_counter() -
                start
            )

            print(
                f"        conversion: "
                f"{elapsed:.3f}s"
            )

        # -----------------------------
        # Zarr
        # -----------------------------

        if not zarr_path.exists():

            print(
                f"[ZARR] {study_id}"
            )

            start = time.perf_counter()

            try:

                volume, _ = load_dicom_series(
                    series_dir
                )

                save_zarr(
                    volume,
                    zarr_path
                )

            except Exception as e:

                print(
                    f"[ZARR] FAILED: {e}"
                )

                continue

            elapsed = (
                time.perf_counter() -
                start
            )

            print(
                f"        conversion: "
                f"{elapsed:.3f}s"
            )

        nifti_paths.append(
            nifti_path
        )

        zarr_paths.append(
            zarr_path
        )

    print(
        f"[CONVERT] NIfTI volumes: "
        f"{len(nifti_paths)}"
    )

    print(
        f"[CONVERT] Zarr volumes: "
        f"{len(zarr_paths)}"
    )

    return nifti_paths, zarr_paths


# ============================================================
# STEP 8
# STORAGE BENCHMARK
# ============================================================

def benchmark_storage():

    nifti_bytes = directory_size(
        NIFTI_ROOT
    )

    zarr_bytes = directory_size(
        ZARR_ROOT
    )

    print()
    print("========== STORAGE ==========")

    print(
        f"NIfTI: "
        f"{format_gb(nifti_bytes):.3f} GB"
    )

    print(
        f"Zarr:  "
        f"{format_gb(zarr_bytes):.3f} GB"
    )

    return {
        "nifti_storage_gb":
            format_gb(nifti_bytes),

        "zarr_storage_gb":
            format_gb(zarr_bytes),
    }


# ============================================================
# STEP 9
# NIFTI SEQUENTIAL READ
# ============================================================

def benchmark_nifti_read(path):

    times = []

    for _ in range(READ_REPEATS):

        start = time.perf_counter()

        image = sitk.ReadImage(
            str(path)
        )

        volume = sitk.GetArrayFromImage(
            image
        )

        elapsed = (
            time.perf_counter() -
            start
        )

        times.append(elapsed)

        # Prevent accidental optimization.
        _ = volume.shape

    return {
        "mean_sec":
            statistics.mean(times),

        "median_sec":
            statistics.median(times),

        "shape":
            tuple(volume.shape)
    }


# ============================================================
# STEP 10
# ZARR SEQUENTIAL READ
# ============================================================

def benchmark_zarr_read(path):

    times = []

    for _ in range(READ_REPEATS):

        start = time.perf_counter()

        root = zarr.open(
            str(path),
            mode="r"
        )

        volume = root["volume"][:]

        elapsed = (
            time.perf_counter() -
            start
        )

        times.append(elapsed)

        _ = volume.shape

    return {
        "mean_sec":
            statistics.mean(times),

        "median_sec":
            statistics.median(times),

        "shape":
            tuple(volume.shape)
    }


# ============================================================
# STEP 11
# RANDOM CROP
# ============================================================

def random_crop(volume):

    z, y, x = volume.shape

    cz, cy, cx = CROP_SHAPE

    if z < cz or y < cy or x < cx:

        return volume

    z0 = random.randint(
        0,
        z - cz
    )

    y0 = random.randint(
        0,
        y - cy
    )

    x0 = random.randint(
        0,
        x - cx
    )

    return volume[
        z0:z0 + cz,
        y0:y0 + cy,
        x0:x0 + cx
    ]


def benchmark_zarr_random_crop(
    path
):

    root = zarr.open(
        str(path),
        mode="r"
    )

    volume = root["volume"]

    times = []

    for _ in range(
        RANDOM_CROP_REPEATS
    ):

        z, y, x = volume.shape

        cz, cy, cx = CROP_SHAPE

        z0 = random.randint(
            0,
            max(0, z - cz)
        )

        y0 = random.randint(
            0,
            max(0, y - cy)
        )

        x0 = random.randint(
            0,
            max(0, x - cx)
        )

        start = time.perf_counter()

        crop = volume[
            z0:z0 + cz,
            y0:y0 + cy,
            x0:x0 + cx
        ]

        elapsed = (
            time.perf_counter() -
            start
        )

        times.append(elapsed)

        _ = crop.shape

    return {
        "mean_sec":
            statistics.mean(times),

        "median_sec":
            statistics.median(times)
    }


# ============================================================
# STEP 12
# PREPROCESSING
# ============================================================

def preprocess(volume):

    x = torch.from_numpy(
        np.asarray(volume)
    ).float()

    mean = x.mean()

    std = x.std()

    x = (
        x - mean
    ) / (
        std + 1e-6
    )

    # [Z,Y,X]
    # ->
    # [1,1,Z,Y,X]

    x = (
        x
        .unsqueeze(0)
        .unsqueeze(0)
    )

    x = F.interpolate(
        x,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False
    )

    return x.squeeze(0)


# ============================================================
# STEP 13
# ZARR DATASET
# ============================================================

class ZarrDataset(Dataset):

    def __init__(
        self,
        paths
    ):

        self.paths = paths

    def __len__(self):

        return len(self.paths)

    def __getitem__(
        self,
        idx
    ):

        root = zarr.open(
            str(self.paths[idx]),
            mode="r"
        )

        volume = root["volume"][:]

        volume = preprocess(
            volume
        )

        return volume


# ============================================================
# STEP 14
# NIFTI DATASET
# ============================================================

class NiftiDataset(Dataset):

    def __init__(
        self,
        paths
    ):

        self.paths = paths

    def __len__(self):

        return len(self.paths)

    def __getitem__(
        self,
        idx
    ):

        image = sitk.ReadImage(
            str(self.paths[idx])
        )

        volume = sitk.GetArrayFromImage(
            image
        )

        volume = preprocess(
            volume
        )

        return volume


# ============================================================
# STEP 15
# SIMPLE 3D MODEL
# ============================================================

class Tiny3DModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv3d(
                1,
                8,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv3d(
                8,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# STEP 16
# END-TO-END THROUGHPUT
# ============================================================

def benchmark_dataloader(
    dataset,
    workers
):

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(
            workers > 0
        )
    )

    start = time.perf_counter()

    count = 0

    for batch in loader:

        count += len(batch)

        if (
            count >=
            MAX_DATALOADER_SAMPLES
        ):

            break

    elapsed = (
        time.perf_counter() -
        start
    )

    return {
        "samples": count,

        "elapsed_sec": elapsed,

        "samples_sec":
            count / elapsed
            if elapsed > 0
            else 0
    }


# ============================================================
# STEP 17
# GPU END-TO-END BENCHMARK
# ============================================================

def benchmark_gpu_pipeline(
    dataset,
    workers
):

    if not torch.cuda.is_available():

        print(
            "[GPU] CUDA unavailable."
        )

        return None

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(
            workers > 0
        )
    )

    model = Tiny3DModel().cuda()

    model.eval()

    count = 0

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for batch in loader:

            batch = batch.cuda(
                non_blocking=True
            )

            _ = model(
                batch
            )

            count += len(batch)

            if (
                count >=
                MAX_DATALOADER_SAMPLES
            ):

                break

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter() -
        start
    )

    return {
        "samples": count,

        "elapsed_sec": elapsed,

        "samples_sec":
            count / elapsed
            if elapsed > 0
            else 0
    }


# ============================================================
# STEP 18
# WRITE RESULTS
# ============================================================

def save_results(rows):

    output = (
        REPORT_ROOT /
        "data_format_benchmark.csv"
    )

    if not rows:

        return

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with open(
        output,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        f"[REPORT] Saved: {output}"
    )


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DATA-001: DICOM vs NIfTI vs Zarr")
    print("=" * 70)

    print(
        f"Raw data:       {RAW_ROOT}"
    )

    print(
        f"Benchmark root: {BENCHMARK_ROOT}"
    )

    print(
        f"Studies:        {NUM_STUDIES}"
    )

    print(
        f"Workers:        {WORKER_COUNTS}"
    )

    print()

    # --------------------------------------------------------
    # 1. SELECT
    # --------------------------------------------------------

    studies = select_studies()

    # --------------------------------------------------------
    # 2. COPY SAMPLE
    # --------------------------------------------------------

    sampled_studies = copy_sample_studies(
        studies
    )

    # --------------------------------------------------------
    # 3. CONVERT
    # --------------------------------------------------------

    nifti_paths, zarr_paths = (
        build_representations(
            sampled_studies
        )
    )

    if not nifti_paths:

        raise RuntimeError(
            "No NIfTI representations created."
        )

    if not zarr_paths:

        raise RuntimeError(
            "No Zarr representations created."
        )

    # --------------------------------------------------------
    # 4. STORAGE
    # --------------------------------------------------------

    storage = benchmark_storage()

    rows = []

    # --------------------------------------------------------
    # 5. SEQUENTIAL READ
    # --------------------------------------------------------

    print()
    print(
        "========== SEQUENTIAL READ =========="
    )

    nifti_read = benchmark_nifti_read(
        nifti_paths[0]
    )

    zarr_read = benchmark_zarr_read(
        zarr_paths[0]
    )

    print(
        "NIfTI:"
    )

    print(
        f"  mean:   "
        f"{nifti_read['mean_sec']:.4f}s"
    )

    print(
        f"  median: "
        f"{nifti_read['median_sec']:.4f}s"
    )

    print(
        f"  shape:  "
        f"{nifti_read['shape']}"
    )

    print(
        "Zarr:"
    )

    print(
        f"  mean:   "
        f"{zarr_read['mean_sec']:.4f}s"
    )

    print(
        f"  median: "
        f"{zarr_read['median_sec']:.4f}s"
    )

    print(
        f"  shape:  "
        f"{zarr_read['shape']}"
    )

    # --------------------------------------------------------
    # 6. RANDOM CROP
    # --------------------------------------------------------

    print()
    print(
        "========== RANDOM CROP =========="
    )

    zarr_crop = (
        benchmark_zarr_random_crop(
            zarr_paths[0]
        )
    )

    print(
        f"Zarr random crop mean: "
        f"{zarr_crop['mean_sec']:.6f}s"
    )

    print(
        f"Zarr random crop median: "
        f"{zarr_crop['median_sec']:.6f}s"
    )

    # --------------------------------------------------------
    # 7. CPU DATA LOADING
    # --------------------------------------------------------

    print()
    print(
        "========== CPU DATA LOADING =========="
    )

    zarr_dataset = ZarrDataset(
        zarr_paths
    )

    nifti_dataset = NiftiDataset(
        nifti_paths
    )

    for workers in WORKER_COUNTS:

        print()
        print(
            f"Workers = {workers}"
        )

        zarr_result = (
            benchmark_dataloader(
                zarr_dataset,
                workers
            )
        )

        nifti_result = (
            benchmark_dataloader(
                nifti_dataset,
                workers
            )
        )

        print(
            f"Zarr:  "
            f"{zarr_result['samples_sec']:.3f} "
            f"samples/sec"
        )

        print(
            f"NIfTI: "
            f"{nifti_result['samples_sec']:.3f} "
            f"samples/sec"
        )

        rows.append({
            "format": "zarr",
            "workers": workers,
            "samples_sec":
                zarr_result[
                    "samples_sec"
                ]
        })

        rows.append({
            "format": "nifti",
            "workers": workers,
            "samples_sec":
                nifti_result[
                    "samples_sec"
                ]
        })

    # --------------------------------------------------------
    # 8. GPU END-TO-END
    # --------------------------------------------------------

    print()
    print(
        "========== GPU END-TO-END =========="
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    for workers in WORKER_COUNTS:

        print()
        print(
            f"GPU workers = {workers}"
        )

        zarr_gpu = (
            benchmark_gpu_pipeline(
                zarr_dataset,
                workers
            )
        )

        nifti_gpu = (
            benchmark_gpu_pipeline(
                nifti_dataset,
                workers
            )
        )

        if zarr_gpu:

            print(
                f"Zarr:  "
                f"{zarr_gpu['samples_sec']:.3f} "
                f"samples/sec"
            )

        if nifti_gpu:

            print(
                f"NIfTI: "
                f"{nifti_gpu['samples_sec']:.3f} "
                f"samples/sec"
            )

    # --------------------------------------------------------
    # 9. SAVE
    # --------------------------------------------------------

    save_results(rows)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"NIfTI storage: "
        f"{storage['nifti_storage_gb']:.3f} GB"
    )

    print(
        f"Zarr storage:  "
        f"{storage['zarr_storage_gb']:.3f} GB"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Use the measured results to decide "
        "whether Zarr or NIfTI should become "
        "the canonical training representation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()



