import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import tacoreader
import rasterio as rio
from PIL import Image

OUT = pathlib.Path("test_inputs")
OUT.mkdir(exist_ok=True)
DIV = 3000.0

# pick a few varied scenes
SAMPLES = {
    "farmland": 8000,
    "coastal": 30000,
    "suburban": 15000,
    "forest": 45000,
}

ds = tacoreader.load("tacofoundation:sen2naipv2-unet")

for name, idx in SAMPLES.items():
    lr_path = ds.read(idx).read(0)

    with rio.open(lr_path) as s:
        arr = s.read([1, 2, 3])
        profile = s.profile.copy()

    # georeferenced version
    profile.update(count=3, dtype=arr.dtype)
    with rio.open(OUT / f"{name}.tif", "w", **profile) as dst:
        dst.write(arr)

    # plain image version
    rgb = np.clip(arr.astype(np.float32) / DIV, 0, 1)
    Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8)) \
         .save(OUT / f"{name}.png")

    print(f"{name}: {arr.shape[2]}x{arr.shape[1]} -> tif + png")

print("\nSaved to", OUT.resolve())