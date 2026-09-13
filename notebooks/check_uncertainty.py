import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch, numpy as np, tacoreader, rasterio as rio
import matplotlib.pyplot as plt
from src.models.rrdbnet import RRDBNet
from src.uncertainty import uncertainty_map

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = RRDBNet(3, 3, 64, 23, 32)
ck = torch.load("checkpoints/satsr_finetuned.pth", map_location="cpu")
model.load_state_dict(ck["model"])
model = model.to(device).eval()

ds = tacoreader.load("tacofoundation:sen2naipv2-unet")
with rio.open(ds.read(4000).read(0)) as s:
    lr = s.read().astype(np.float32)

x = torch.from_numpy(np.clip(lr[:3] / 3000.0, 0, 1)).unsqueeze(0)
mean, unc = uncertainty_map(model, x, device)

print("uncertainty range:", unc.min(), unc.max(), "mean:", unc.mean())

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(np.clip(lr[:3].transpose(1,2,0)/3000, 0, 1)); ax[0].set_title("Input 10 m")
ax[1].imshow(np.clip(mean.transpose(1,2,0), 0, 1));        ax[1].set_title("Output 2.5 m")
im = ax[2].imshow(unc, cmap="inferno");                    ax[2].set_title("Uncertainty")
plt.colorbar(im, ax=ax[2], fraction=0.046)
for a in ax: a.axis("off")
plt.tight_layout()
plt.savefig("outputs/uncertainty_check.png", dpi=200)
print("Saved outputs/uncertainty_check.png")