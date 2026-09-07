import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import tacoreader
import rasterio as rio
from src.models.loader import load_pretrained
from src.inference import super_resolve

# ---- 1. Save one SEN2NAIP patch to disk as a GeoTIFF ----
dataset = tacoreader.load("tacofoundation:sen2naipv2-unet")
lr_path = dataset.read(4000).read(0)

with rio.open(lr_path) as src:
    data = src.read([1, 2, 3])          # just RGB
    profile = src.profile.copy()

profile.update(count=3, dtype=data.dtype)

test_input = "data/test_input.tif"
with rio.open(test_input, "w", **profile) as dst:
    dst.write(data)

print("Saved test input:", test_input, data.shape)

# ---- 2. Load the model ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = load_pretrained("checkpoints/esrgan_1S2.pth", num_ch=3, device=device)

# ---- 3. Run the pipeline ----
super_resolve(
    in_path=test_input,
    out_path="outputs/test_output.tif",
    model=model,
    device=device,
    tile=128,
    overlap=16,
)