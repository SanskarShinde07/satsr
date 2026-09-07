import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import tacoreader
import rasterio as rio
import matplotlib.pyplot as plt
from src.models.loader import load_pretrained

device = "mps" if torch.backends.mps.is_available() else "cpu"

# NOTE: put your real checkpoint filename here
model = load_pretrained("checkpoints/esrgan_1S2.pth", num_ch=3, device=device)

dataset = tacoreader.load("tacofoundation:sen2naipv2-unet")
idx = 4000
lr_path = dataset.read(idx).read(0)
hr_path = dataset.read(idx).read(1)

with rio.open(lr_path) as src, rio.open(hr_path) as dst:
    lr = src.read().astype(np.float32)
    hr = dst.read().astype(np.float32)

x = torch.from_numpy(np.clip(lr[:3] / 3000.0, 0, 1)).unsqueeze(0).to(device)

with torch.no_grad():
    sr = model(x).clamp(0, 1).squeeze(0).cpu().numpy()

print("LR:", lr.shape, "SR:", sr.shape, "HR:", hr.shape)

def show(a, scale=1.0):
    return np.clip(a[:3].transpose(1, 2, 0) / scale, 0, 1)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(show(lr, 3000)); ax[0].set_title("LR input (10 m)")
ax[1].imshow(show(sr));       ax[1].set_title("SR output (2.5 m)")
ax[2].imshow(show(hr, 3000)); ax[2].set_title("HR truth (2.5 m)")
for a in ax: a.axis("off")
plt.tight_layout()
plt.savefig("outputs/inference_check.png", dpi=120)
print("Saved outputs/inference_check.png")