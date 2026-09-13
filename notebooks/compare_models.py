import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import tacoreader
import rasterio as rio
import matplotlib.pyplot as plt
from src.models.loader import load_pretrained
from src.models.rrdbnet import RRDBNet

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- baseline (original pretrained) ---
base = load_pretrained("checkpoints/esrgan_1S2.pth", num_ch=3, device=device)

# --- our fine-tuned model ---
tuned = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32)
ck = torch.load("checkpoints/satsr_finetuned.pth", map_location="cpu")
tuned.load_state_dict(ck["model"])
tuned = tuned.to(device).eval()
print("Loaded fine-tuned model from epoch", ck["epoch"])

# --- data ---
dataset = tacoreader.load("tacofoundation:sen2naipv2-unet")
idx = 4000
with rio.open(dataset.read(idx).read(0)) as s:
    lr = s.read().astype(np.float32)
with rio.open(dataset.read(idx).read(1)) as s:
    hr = s.read().astype(np.float32)

x = torch.from_numpy(np.clip(lr[:3] / 3000.0, 0, 1)).unsqueeze(0).to(device)

with torch.no_grad():
    sr_base = base(x).clamp(0, 1).squeeze(0).cpu().numpy()
    sr_tuned = tuned(x).clamp(0, 1).squeeze(0).cpu().numpy()

def show(a, scale=1.0):
    return np.clip(a[:3].transpose(1, 2, 0) / scale, 0, 1)

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(show(lr, 3000));  ax[0].set_title("Input — 10 m")
ax[1].imshow(show(sr_base));   ax[1].set_title("Pretrained baseline")
ax[2].imshow(show(sr_tuned));  ax[2].set_title("Our fine-tuned — 2.5 m")
ax[3].imshow(show(hr, 3000));  ax[3].set_title("Reference — 2.5 m")
for a in ax: a.axis("off")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=200)
print("Saved outputs/model_comparison.png")