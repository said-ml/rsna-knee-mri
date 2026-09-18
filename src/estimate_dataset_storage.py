#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import math
import pydicom

RAW_ROOT = Path("data/raw/train_series")

def gib(x):
    return x / (1024 ** 3)

def mib(x):
    return x / (1024 ** 2)

def get_int(ds, name, default=None):
    value = getattr(ds, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def main():
    series_dirs = sorted(
        p for p in RAW_ROOT.rglob("*")
        if p.is_dir() and p.parent != RAW_ROOT
    )

    # Better: identify series as immediate children of study directories.
    series_dirs = []
    for study_dir in sorted(RAW_ROOT.iterdir()):
        if not study_dir.is_dir():
            continue
        for series_dir in sorted(study_dir.iterdir()):
            if series_dir.is_dir():
                series_dirs.append(series_dir)

    print(f"Series discovered: {len(series_dirs):,}")

    total_bytes = 0
    total_voxels = 0
    dtype_bytes = Counter()
    series_by_dtype = Counter()
    series_by_shape = Counter()
    failures = []

    for i, series_dir in enumerate(series_dirs, 1):
        files = sorted(p for p in series_dir.iterdir() if p.is_file())

        if not files:
            failures.append((str(series_dir), "no files"))
            continue

        try:
            # Read one header to establish geometry/pixel representation.
            ds0 = pydicom.dcmread(
                str(files[0]),
                stop_before_pixels=True,
                specific_tags=[
                    "Rows",
                    "Columns",
                    "BitsAllocated",
                    "PixelRepresentation",
                    "SamplesPerPixel",
                    "NumberOfFrames",
                    "FloatPixelData",
                    "DoubleFloatPixelData",
                    "PixelData",
                ],
            )

            rows = get_int(ds0, "Rows")
            cols = get_int(ds0, "Columns")
            bits = get_int(ds0, "BitsAllocated")
            samples = get_int(ds0, "SamplesPerPixel", 1) or 1
            frames0 = get_int(ds0, "NumberOfFrames", 1) or 1

            if rows is None or cols is None:
                raise ValueError("missing Rows/Columns")

            # Determine bytes per stored sample.
            if hasattr(ds0, "DoubleFloatPixelData"):
                dtype_name = "float64"
                bytes_per_sample = 8
            elif hasattr(ds0, "FloatPixelData"):
                dtype_name = "float32"
                bytes_per_sample = 4
            elif bits in (8, 16, 32, 64):
                bytes_per_sample = bits // 8

                if bits == 16:
                    signed = int(getattr(ds0, "PixelRepresentation", 0)) == 1
                    dtype_name = "int16" if signed else "uint16"
                elif bits == 8:
                    signed = int(getattr(ds0, "PixelRepresentation", 0)) == 1
                    dtype_name = "int8" if signed else "uint8"
                elif bits == 32:
                    signed = int(getattr(ds0, "PixelRepresentation", 0)) == 1
                    dtype_name = "int32" if signed else "uint32"
                else:
                    dtype_name = f"{bits}-bit"
            else:
                raise ValueError(f"unsupported BitsAllocated={bits}")

            # Most series are one frame per DICOM file.
            # Account for multi-frame files too.
            voxels = 0
            for f in files:
                ds = pydicom.dcmread(
                    str(f),
                    stop_before_pixels=True,
                    specific_tags=[
                        "Rows",
                        "Columns",
                        "BitsAllocated",
                        "PixelRepresentation",
                        "SamplesPerPixel",
                        "NumberOfFrames",
                    ],
                )

                r = get_int(ds, "Rows", rows)
                c = get_int(ds, "Columns", cols)
                frames = get_int(ds, "NumberOfFrames", 1) or 1
                spp = get_int(ds, "SamplesPerPixel", samples) or 1

                voxels += r * c * frames * spp

            nbytes = voxels * bytes_per_sample

            total_voxels += voxels
            total_bytes += nbytes
            dtype_bytes[dtype_name] += nbytes
            series_by_dtype[dtype_name] += 1
            series_by_shape[(rows, cols)] += 1

        except Exception as e:
            failures.append((str(series_dir), str(e)))

        if i % 1000 == 0:
            print(f"Processed {i:,}/{len(series_dirs):,}")

    print("\n" + "=" * 70)
    print("DATASET-WIDE DECODED PIXEL STORAGE ESTIMATE")
    print("=" * 70)

    print(f"Series processed: {len(series_dirs):,}")
    print(f"Successful:       {len(series_dirs) - len(failures):,}")
    print(f"Failures:         {len(failures):,}")

    print(f"\nTotal voxels:     {total_voxels:,}")
    print(f"Estimated bytes:  {total_bytes:,}")
    print(f"Estimated GiB:    {gib(total_bytes):,.2f}")
    print(f"Estimated TiB:    {total_bytes / (1024**4):,.3f}")

    print("\nBY ESTIMATED DTYPE")
    print("-" * 70)

    for dtype, n in sorted(series_by_dtype.items()):
        print(
            f"{dtype:10s} "
            f"series={n:6,} "
            f"bytes={dtype_bytes[dtype]:14,} "
            f"GiB={gib(dtype_bytes[dtype]):10.2f}"
        )

    if failures:
        print("\nFAILURES")
        print("-" * 70)
        for path, error in failures[:20]:
            print(path)
            print(" ", error)

        if len(failures) > 20:
            print(f"... {len(failures) - 20} more")

if __name__ == "__main__":
    main()
