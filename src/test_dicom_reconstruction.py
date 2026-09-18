from pathlib import Path
import csv
import warnings

import numpy as np
import pydicom
import SimpleITK as sitk


RAW_ROOT = Path("/workspace/data/raw/train_series")
REPORT = Path("/workspace/data/reports/dicom_reconstruction_test.csv")
REPORT.parent.mkdir(parents=True, exist_ok=True)

N_SERIES = 20


def dicom_geometry(files):
    records = []

    for f in files:
        ds = pydicom.dcmread(
            str(f),
            stop_before_pixels=True,
            specific_tags=[
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "PixelSpacing",
                "Rows",
                "Columns",
                "InstanceNumber",
            ],
        )

        ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)

        normal = np.cross(iop[:3], iop[3:])
        normal /= np.linalg.norm(normal)

        position = float(np.dot(ipp, normal))

        records.append({
            "file": f,
            "ipp": ipp,
            "iop": iop,
            "position": position,
            "instance": getattr(ds, "InstanceNumber", None),
            "pixel_spacing": np.asarray(
                ds.PixelSpacing,
                dtype=np.float64,
            ),
            "rows": int(ds.Rows),
            "columns": int(ds.Columns),
        })

    records.sort(key=lambda x: x["position"])

    positions = np.asarray(
        [r["position"] for r in records],
        dtype=np.float64,
    )

    spacing = np.diff(positions)

    return records, positions, spacing


def main():

    all_series = []

    for study_dir in sorted(RAW_ROOT.iterdir()):
        if not study_dir.is_dir():
            continue

        for series_dir in sorted(study_dir.iterdir()):
            if series_dir.is_dir():
                all_series.append(
                    (study_dir.name, series_dir.name, series_dir)
                )

            if len(all_series) >= N_SERIES:
                break

        if len(all_series) >= N_SERIES:
            break

    print(f"Testing {len(all_series)} series")

    fields = [
        "study_uid",
        "series_uid",
        "n_dicom",
        "sitk_z",
        "sitk_y",
        "sitk_x",
        "sitk_spacing_x",
        "sitk_spacing_y",
        "sitk_spacing_z",
        "dicom_spacing_median",
        "spacing_error_mm",
        "shape_match",
        "spacing_match",
        "direction_valid",
        "dtype",
        "status",
        "error",
    ]

    with REPORT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for idx, (study_uid, series_uid, series_dir) in enumerate(
            all_series, 1
        ):

            result = {
                "study_uid": study_uid,
                "series_uid": series_uid,
                "n_dicom": "",
                "sitk_z": "",
                "sitk_y": "",
                "sitk_x": "",
                "sitk_spacing_x": "",
                "sitk_spacing_y": "",
                "sitk_spacing_z": "",
                "dicom_spacing_median": "",
                "spacing_error_mm": "",
                "shape_match": "",
                "spacing_match": "",
                "direction_valid": "",
                "dtype": "",
                "status": "VALID",
                "error": "",
            }

            try:
                files = sorted(series_dir.glob("*.dcm"))

                records, positions, diffs = dicom_geometry(files)

                dicom_spacing = float(np.median(np.abs(diffs)))

                # Use SimpleITK's GDCM series discovery.
                sitk_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                    str(series_dir)
                )

                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(sitk_files)

                image = reader.Execute()

                size_x, size_y, size_z = image.GetSize()
                spacing_x, spacing_y, spacing_z = image.GetSpacing()

                result["n_dicom"] = len(files)

                result["sitk_z"] = size_z
                result["sitk_y"] = size_y
                result["sitk_x"] = size_x

                result["sitk_spacing_x"] = f"{spacing_x:.9f}"
                result["sitk_spacing_y"] = f"{spacing_y:.9f}"
                result["sitk_spacing_z"] = f"{spacing_z:.9f}"

                result["dicom_spacing_median"] = (
                    f"{dicom_spacing:.9f}"
                )

                spacing_error = abs(
                    spacing_z - dicom_spacing
                )

                result["spacing_error_mm"] = (
                    f"{spacing_error:.9f}"
                )

                expected_x = records[0]["columns"]
                expected_y = records[0]["rows"]
                expected_z = len(records)

                shape_match = (
                    size_x == expected_x
                    and size_y == expected_y
                    and size_z == expected_z
                )

                spacing_match = (
                    abs(spacing_x - records[0]["pixel_spacing"][1])
                    < 1e-4
                    and
                    abs(spacing_y - records[0]["pixel_spacing"][0])
                    < 1e-4
                    and
                    abs(spacing_z - dicom_spacing)
                    < 1e-3
                )

                direction = np.asarray(
                    image.GetDirection(),
                    dtype=np.float64,
                )

                direction_valid = (
                    direction.size == 9
                    and np.all(np.isfinite(direction))
                )

                result["shape_match"] = shape_match
                result["spacing_match"] = spacing_match
                result["direction_valid"] = direction_valid

                result["dtype"] = str(
                    sitk.GetArrayFromImage(
                        image
                    ).dtype
                )

                if not shape_match:
                    result["status"] = "REVIEW"
                    result["error"] = "SHAPE_MISMATCH"

                elif not spacing_match:
                    result["status"] = "REVIEW"
                    result["error"] = "SPACING_MISMATCH"

                elif not direction_valid:
                    result["status"] = "REVIEW"
                    result["error"] = "INVALID_DIRECTION"

            except Exception as e:
                result["status"] = "INVALID"
                result["error"] = (
                    f"{type(e).__name__}: {e}"
                )

            writer.writerow(result)
            f.flush()

            print(
                f"[{idx:02d}/{len(all_series)}] "
                f"{study_uid[:18]}... "
                f"{series_uid[:18]}... "
                f"{result['status']} "
                f"error={result['error']}",
                flush=True,
            )

    print("\n========== COMPLETE ==========")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
