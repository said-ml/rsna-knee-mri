# DATA-001 — DICOM → Zarr Storage Migration

## Status

**COMPLETED — POST-CHECK PASS**

Migration date: 2026-09-18

This document records the completed transition from raw DICOM storage to
validated Zarr-backed storage for the converted RSNA Knee MRI series.

---

## Conversion State

- Total studies: 4,407
- Total series: 24,371
- Production v3 Zarr conversions: 16,434
- Legacy v2 Zarr conversions: 50
- Total Zarr-backed series: 16,484
- Duplicate SeriesInstanceUIDs: 0

Production converter:

`src/convert_dicom_to_zarr_v3.py`

Conversion manifest:

`data/reports/zarr_conversion_manifest_v3.csv`

Legacy manifest:

`data/reports/zarr_conversion_manifest_v2.csv`

---

## Validation Gates

### Zarr integrity

Representative Zarr outputs were inspected and verified.

**PASS**

### DICOM ↔ Zarr equivalence

Representative DICOM/Zarr pairs were compared.

- Full pixel equality: PASS
- Shape equality: PASS
- Spacing: PASS
- Origin: PASS
- Direction: PASS
- Failures: 0

**PASS**

### Zarr → PyTorch → GPU

Representative Zarr volumes were loaded through the PyTorch pipeline and
executed on the RTX 4070 SUPER.

- Zarr loading: PASS
- float32 conversion: PASS
- finite values: PASS
- DataLoader: PASS
- GPU transfer: PASS
- 3D model forward pass: PASS
- output shape: `(2, 12)`
- non-finite outputs: 0

**PASS**

This validates the infrastructure boundary, not the final model or final
training preprocessing.

---

## Deletion Audit

The read-only deletion audit established:

- Candidate series: 16,484
- Candidate studies: 2,975
- Duplicate study/series pairs: 0
- Duplicate SeriesInstanceUIDs: 0
- Missing Zarr outputs: 0
- Missing DICOM directories: 0
- Fully deletable studies: 2,974
- Partially deletable studies: 1

The partially covered study was handled at the individual
SeriesInstanceUID-directory level.

Its four non-Zarr-backed series were preserved.

No study directory was deleted.

Deletion manifest:

`data/reports/zarr_dicom_deletion_plan.csv`

---

## Guarded Deletion

Deletion was performed exclusively through:

`guarded_delete_zarr_backed_dicom.py`

The script required the exact confirmation:

`DELETE 16484 SERIES`

Safety checks included:

- exact audited candidate count
- duplicate detection
- manifest/path consistency
- exact path-depth validation
- symlink rejection
- DICOM directory existence checks
- Zarr existence checks
- per-series pre-deletion revalidation
- post-deletion Zarr existence verification

No unrestricted `rm -rf` operation was used.

---

## Final Post-Deletion State

- Deleted DICOM series: 16,484 / 16,484
- Deleted DICOM storage: 355.56 GiB
- Candidate DICOM directories remaining: 0
- Missing Zarr outputs: 0
- Free space after migration: 396.85 GiB

**DELETION COMPLETE — POST-CHECK PASS**

---

## Data-State Transition

The project has transitioned from:

**Raw DICOM + converted Zarr**

to:

**Validated Zarr-backed series + remaining DICOM-only series**

The remaining DICOM series are intentionally retained because they are not
covered by the validated Zarr conversion/deletion manifest.

They must not be treated as redundant storage without a new conversion and
validation cycle.

---

## Reproducibility

The migration is reproducible/auditable through:

1. `src/audit_dicom_geometry.py`
2. `src/convert_dicom_to_zarr_v3.py`
3. `data/reports/zarr_conversion_manifest_v3.csv`
4. `data/reports/zarr_conversion_manifest_v2.csv`
5. `data/reports/zarr_dicom_deletion_plan.csv`
6. `guarded_delete_zarr_backed_dicom.py`

The deletion manifest records the exact DICOM series removed.

The conversion manifests record the corresponding Zarr conversions.

---

## Important Invariant

For every deleted DICOM series:

**A validated Zarr representation existed before deletion and remained
present after deletion.**

Final migration status:

**DATA-001 COMPLETE**
