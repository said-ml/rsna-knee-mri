import monai
import pydicom
import SimpleITK
import nibabel
import zarr
import numpy
import scipy
import pandas
import sklearn


import torch

print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('pytorch:', torch.__version__)


print('MONAI:', monai.__version__)
print('pydicom:', pydicom.__version__)
print('SimpleITK:', SimpleITK.Version_VersionString())
print('nibabel:', nibabel.__version__)
print('zarr:', zarr.__version__)
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('pandas:', pandas.__version__)
print('sklearn:', sklearn.__version__)
print('=====================ALL MEDICAL STACK OK===================')