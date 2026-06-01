import numpy as np
import pandas as pd
import matplotlib
import torch
import torchvision
import transformers
import diffusers
import open_clip
import trimesh
import open3d as o3d
import PIL
import sklearn

print("✔ NumPy version:", np.__version__)
print("✔ Pandas version:", pd.__version__)
print("✔ Matplotlib version:", matplotlib.__version__)
print("✔ Torch:", torch.__version__)
print("✔ Torchvision:", torchvision.__version__)
print("✔ Transformers:", transformers.__version__)
print("✔ Diffusers:", diffusers.__version__)
print("✔ OpenCLIP:", open_clip.__version__ if hasattr(open_clip, '__version__') else "Imported")
print("✔ Trimesh:", trimesh.__version__)
print("✔ Open3D:", o3d.__version__)
print("✔ PIL:", PIL.__version__)
print("✔ Sklearn:", sklearn.__version__)
