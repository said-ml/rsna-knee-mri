# rsna-knee-mri
detect a defined set of clinically important abnormalities on knee MRI examinations
``` python
conversion → manifest audit → representative integrity → DICOM↔Zarr equivalence → PyTorch/GPU check → deletion plan → guarded deletion.
```

BASELINE-001
-------------------------------
Training studies:       58
Train / validation:    46 / 12
Model:                 3D CNN
Input:                 1×32×128×128
Series:                sagittal, fluid-sensitive preferred
Normalization:         1–99 percentile → [-1,1]
Epochs:                20
Checkpoint:            best.pt
Kaggle score:          0.525
Status:                END-TO-END VERIFIED


-------------------------------------------------