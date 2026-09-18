import csv
import math
import os
from pathlib import Path

import numpy as np
import pydicom


RAW_ROOT = Path("/workspace/data/raw/train_series")
REPORT_DIR = Path("/workspace/data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = REPORT_DIR / "dicom_geometry_audit.csv"

TOL_ORIENTATION = 1e-4
TOL_PIXEL_SPACING = 1e-4
TOL_SPACING_REL = 0.02          # 2% relative deviation
TOL_DUPLICATE_MM = 0.01


def vec(value):
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.float64)
    except Exception:
        return None


def close_vec(a, b, tol):
    if a is None or b is None:
        return False
    return np.allclose(a, b, atol=tol, rtol=0)


def audit_series(study_uid, series_uid, series_path):

    files = sorted(series_path.glob("*.dcm"))

    result = {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "n_files": len(files),
        "status": "VALID",
        "reason": "",
        "modality": "",
        "sop_class": "",
        "rows": "",
        "columns": "",
        "pixel_spacing": "",
        "orientation": "",
        "n_missing_ipp": 0,
        "n_missing_iop": 0,
        "n_orientation_groups": "",
        "n_pixel_spacing_groups": "",
        "n_duplicate_positions": 0,
        "median_spacing_mm": "",
        "min_spacing_mm": "",
        "max_spacing_mm": "",
        "max_spacing_deviation_pct": "",
        "instance_min": "",
        "instance_max": "",
    }

    if not files:
        result["status"] = "INVALID"
        result["reason"] = "NO_DICOM_FILES"
        return result

    records = []

    try:
        for f in files:
            ds = pydicom.dcmread(
                str(f),
                stop_before_pixels=True,
                specific_tags=[
                    "SOPClassUID",
                    "Modality",
                    "Rows",
                    "Columns",
                    "PixelSpacing",
                    "ImageOrientationPatient",
                    "ImagePositionPatient",
                    "InstanceNumber",
                ],
            )

            modality = str(getattr(ds, "Modality", ""))
            sop_class = str(getattr(ds, "SOPClassUID", ""))

            if not result["modality"]:
                result["modality"] = modality

            if not result["sop_class"]:
                result["sop_class"] = sop_class

            rows = getattr(ds, "Rows", None)
            columns = getattr(ds, "Columns", None)

            if result["rows"] == "":
                result["rows"] = rows if rows is not None else ""
                result["columns"] = columns if columns is not None else ""
            else:
                if rows != result["rows"] or columns != result["columns"]:
                    result["status"] = "REVIEW"
                    result["reason"] = "INCONSISTENT_DIMENSIONS"

            ps = vec(getattr(ds, "PixelSpacing", None))

            if ps is None or len(ps) != 2:
                result["status"] = "REVIEW"
                result["reason"] = result["reason"] or "MISSING_PIXEL_SPACING"
            else:
                if result["pixel_spacing"] == "":
                    result["pixel_spacing"] = (
                        f"{ps[0]:.8f},{ps[1]:.8f}"
                    )
                    reference_ps = ps
                elif not close_vec(ps, reference_ps, TOL_PIXEL_SPACING):
                    result["status"] = "REVIEW"
                    result["reason"] = (
                        result["reason"]
                        or "INCONSISTENT_PIXEL_SPACING"
                    )

            iop = vec(getattr(ds, "ImageOrientationPatient", None))
            ipp = vec(getattr(ds, "ImagePositionPatient", None))

            if iop is None or len(iop) != 6:
                result["n_missing_iop"] += 1
            else:
                if result["orientation"] == "":
                    result["orientation"] = ",".join(
                        f"{x:.8f}" for x in iop
                    )
                    reference_iop = iop
                elif not close_vec(iop, reference_iop, TOL_ORIENTATION):
                    result["status"] = "REVIEW"
                    result["reason"] = (
                        result["reason"]
                        or "INCONSISTENT_ORIENTATION"
                    )

            if ipp is None or len(ipp) != 3:
                result["n_missing_ipp"] += 1

            records.append(
                {
                    "file": str(f),
                    "ipp": ipp,
                    "iop": iop,
                    "instance": getattr(ds, "InstanceNumber", None),
                }
            )

        # Geometry requires IOP and IPP for every image.
        if result["n_missing_iop"] > 0:
            result["status"] = "REVIEW"
            result["reason"] = result["reason"] or "MISSING_IOP"

        if result["n_missing_ipp"] > 0:
            result["status"] = "REVIEW"
            result["reason"] = result["reason"] or "MISSING_IPP"

        valid_geometry = [
            r for r in records
            if r["ipp"] is not None
            and r["iop"] is not None
            and len(r["ipp"]) == 3
            and len(r["iop"]) == 6
        ]

        if len(valid_geometry) < 2:
            result["status"] = "REVIEW"
            result["reason"] = result["reason"] or "INSUFFICIENT_GEOMETRY"
            return result

        # Group orientations approximately.
        orientation_groups = []

        for r in valid_geometry:
            iop = r["iop"]

            found = False
            for ref in orientation_groups:
                if np.allclose(iop, ref, atol=TOL_ORIENTATION, rtol=0):
                    found = True
                    break

            if not found:
                orientation_groups.append(iop)

        result["n_orientation_groups"] = len(orientation_groups)

        if len(orientation_groups) > 1:
            result["status"] = "REVIEW"
            result["reason"] = (
                result["reason"]
                or "MULTIPLE_ORIENTATIONS"
            )

        # Use the first orientation to derive the slice normal.
        iop = valid_geometry[0]["iop"]

        row = iop[:3]
        col = iop[3:]
        normal = np.cross(row, col)

        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            result["status"] = "INVALID"
            result["reason"] = "DEGENERATE_SLICE_NORMAL"
            return result

        normal /= norm

        # Project IPP onto the slice normal.
        positions = []

        for r in valid_geometry:
            position = float(np.dot(r["ipp"], normal))
            positions.append(
                (
                    position,
                    r["instance"],
                    r["file"],
                )
            )

        positions.sort(key=lambda x: x[0])

        coords = np.asarray(
            [x[0] for x in positions],
            dtype=np.float64,
        )

        diffs = np.diff(coords)

        if len(diffs) > 0:
            abs_diffs = np.abs(diffs)

            median_spacing = float(np.median(abs_diffs))
            min_spacing = float(np.min(abs_diffs))
            max_spacing = float(np.max(abs_diffs))

            result["median_spacing_mm"] = f"{median_spacing:.6f}"
            result["min_spacing_mm"] = f"{min_spacing:.6f}"
            result["max_spacing_mm"] = f"{max_spacing:.6f}"

            if median_spacing > 0:
                deviation_pct = (
                    np.max(
                        np.abs(abs_diffs - median_spacing)
                    )
                    / median_spacing
                    * 100.0
                )

                result["max_spacing_deviation_pct"] = (
                    f"{deviation_pct:.4f}"
                )

                if deviation_pct > TOL_SPACING_REL * 100:
                    result["status"] = "REVIEW"
                    result["reason"] = (
                        result["reason"]
                        or "IRREGULAR_SLICE_SPACING"
                    )

            # Detect duplicate slice positions.
            duplicate_count = int(
                np.sum(abs_diffs < TOL_DUPLICATE_MM)
            )

            result["n_duplicate_positions"] = duplicate_count

            if duplicate_count > 0:
                result["status"] = "REVIEW"
                result["reason"] = (
                    result["reason"]
                    or "DUPLICATE_SLICE_POSITIONS"
                )

        instances = [
            x[1]
            for x in positions
            if x[1] is not None
        ]

        if instances:
            try:
                result["instance_min"] = min(instances)
                result["instance_max"] = max(instances)
            except Exception:
                pass

        # Only explicitly image modalities should normally become volumes.
        if result["modality"] != "MR":
            result["status"] = "REVIEW"
            result["reason"] = (
                result["reason"]
                or "NON_MR_MODALITY"
            )

    except Exception as e:
        result["status"] = "INVALID"
        result["reason"] = f"EXCEPTION:{type(e).__name__}:{e}"

    return result


def main():

    studies = sorted(
        p for p in RAW_ROOT.iterdir()
        if p.is_dir()
    )

    print(f"Studies discovered: {len(studies)}")

    fields = [
        "study_uid",
        "series_uid",
        "status",
        "reason",
        "n_files",
        "modality",
        "sop_class",
        "rows",
        "columns",
        "pixel_spacing",
        "orientation",
        "n_missing_ipp",
        "n_missing_iop",
        "n_orientation_groups",
        "n_pixel_spacing_groups",
        "n_duplicate_positions",
        "median_spacing_mm",
        "min_spacing_mm",
        "max_spacing_mm",
        "max_spacing_deviation_pct",
        "instance_min",
        "instance_max",
    ]

    counts = {
        "VALID": 0,
        "REVIEW": 0,
        "INVALID": 0,
    }

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        total_series = 0

        for study_dir in studies:

            series_dirs = sorted(
                p for p in study_dir.iterdir()
                if p.is_dir()
            )

            for series_dir in series_dirs:

                total_series += 1

                result = audit_series(
                    study_dir.name,
                    series_dir.name,
                    series_dir,
                )

                writer.writerow(result)

                status = result["status"]
                counts[status] += 1

                if total_series % 500 == 0:
                    print(
                        f"{total_series:,} series | "
                        f"VALID={counts['VALID']:,} "
                        f"REVIEW={counts['REVIEW']:,} "
                        f"INVALID={counts['INVALID']:,}",
                        flush=True,
                    )

    print("\n========== COMPLETE ==========")
    print(f"Series:   {total_series:,}")
    print(f"VALID:    {counts['VALID']:,}")
    print(f"REVIEW:   {counts['REVIEW']:,}")
    print(f"INVALID:  {counts['INVALID']:,}")
    print(f"\nReport: {OUTPUT}")


if __name__ == "__main__":
    main()