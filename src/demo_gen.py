import os
import json
import numpy as np
import torch
import tacoreader
import rasterio as rio
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.models.rrdbnet import RRDBNet
from src.uncertainty import uncertainty_map
from src.inference import super_resolve

OUT = "app/static/demos"
DIV = 3000.0


def to_png(arr, path, scale=1.0):
    """arr: (3,H,W) float -> RGB png"""
    img = np.clip(arr[:3].transpose(1, 2, 0) / scale, 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def unc_to_png(unc, path):
    """uncertainty map -> inferno colour png"""
    import matplotlib.cm as cm
    u = unc / max(unc.max(), 1e-6)
    rgba = cm.inferno(u)
    Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).save(path)


def build(indices, names, ckpt="checkpoints/satsr_finetuned.pth"):
    os.makedirs(OUT, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = RRDBNet(3, 3, 64, 23, 32)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model = model.to(device).eval()

    ds = tacoreader.load("tacofoundation:sen2naipv2-unet")
    results = []

    for idx, name in zip(indices, names):
        row = ds.read(idx)
        lr_path = row.read(0)

        with rio.open(lr_path) as s:
            lr = s.read().astype(np.float32)
        with rio.open(row.read(1)) as s:
            hr = s.read().astype(np.float32)

        x = torch.from_numpy(np.clip(lr[:3] / DIV, 0, 1)).unsqueeze(0)
        sr, unc = uncertainty_map(model, x, device)

        hr_n = np.clip(hr[:3] / DIV, 0, 1)

        # "before" image, upscaled with nearest so it stays blocky
        lr_big = Image.fromarray(
            (np.clip(lr[:3].transpose(1, 2, 0) / DIV, 0, 1) * 255).astype(np.uint8)
        ).resize((sr.shape[2], sr.shape[1]), Image.NEAREST)
        lr_big.save(f"{OUT}/{name}_before.png")

        to_png(sr, f"{OUT}/{name}_after.png")
        to_png(hr_n, f"{OUT}/{name}_reference.png")
        unc_to_png(unc, f"{OUT}/{name}_uncertainty.png")

        # georeferenced output
        tif_ok = False
        try:
            super_resolve(str(lr_path), f"{OUT}/{name}_output.tif",
                          model, device=device, tile=128, overlap=16)
            tif_ok = True
        except Exception as e:
            print("  GeoTIFF skipped:", e)

        p = psnr(hr_n, sr, data_range=1.0)
        s_val = ssim(hr_n.transpose(1, 2, 0), sr.transpose(1, 2, 0),
                     channel_axis=2, data_range=1.0)

        results.append({
            "id": name,
            "psnr": round(float(p), 2),
            "ssim": round(float(s_val), 3),
            "uncertainty": round(float(unc.mean()), 4),
            "tif": tif_ok,
        })
        print(f"{name}: PSNR {p:.2f}  SSIM {s_val:.3f}  "
              f"unc {unc.mean():.4f}  tif {tif_ok}")

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", f"{OUT}/metrics.json")


if __name__ == "__main__":
    build(
        indices=[4000, 12000, 25000],
        names=["urban", "mixed", "rural"],
    )