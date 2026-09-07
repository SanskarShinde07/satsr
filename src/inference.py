import numpy as np
import torch
import rasterio as rio
from rasterio.windows import Window


def _blend_weights(size, overlap):
    """Feathered weight mask so tile seams fade out."""
    w = np.ones(size, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w[:, None] * w[None, :]


def super_resolve(in_path, out_path, model, device="cpu",
                  tile=128, overlap=16, scale=4, divisor=3000.0):
    """Read a Sentinel-2 GeoTIFF, super-resolve it, write a georeferenced GeoTIFF."""

    with rio.open(in_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        H, W = src.height, src.width

        out_h, out_w = H * scale, W * scale
        acc = np.zeros((3, out_h, out_w), dtype=np.float32)
        wsum = np.zeros((out_h, out_w), dtype=np.float32)

        step = tile - overlap
        mask = _blend_weights(tile * scale, overlap * scale)

        for row in range(0, H, step):
            for col in range(0, W, step):
                h = min(tile, H - row)
                w = min(tile, W - col)
                if h < 8 or w < 8:
                    continue

                patch = src.read([1, 2, 3], window=Window(col, row, w, h))
                patch = np.clip(patch.astype(np.float32) / divisor, 0, 1)

                # pad partial edge tiles up to full size
                ph, pw = tile - h, tile - w
                if ph or pw:
                    patch = np.pad(patch, ((0, 0), (0, ph), (0, pw)), mode="reflect")

                x = torch.from_numpy(patch).unsqueeze(0).to(device)
                with torch.no_grad():
                    sr = model(x).clamp(0, 1).squeeze(0).cpu().numpy()

                r0, c0 = row * scale, col * scale
                r1, c1 = r0 + tile * scale, c0 + tile * scale
                r1, c1 = min(r1, out_h), min(c1, out_w)
                th, tw = r1 - r0, c1 - c0

                acc[:, r0:r1, c0:c1] += sr[:, :th, :tw] * mask[:th, :tw]
                wsum[r0:r1, c0:c1] += mask[:th, :tw]

    out = acc / np.maximum(wsum, 1e-6)
    out = (out * divisor).astype(np.uint16)

    # pixel size shrinks by `scale`; origin stays put
    new_transform = rio.Affine(
        transform.a / scale, transform.b, transform.c,
        transform.d, transform.e / scale, transform.f,
    )

    profile.update(
        height=out_h, width=out_w, count=3,
        dtype="uint16", transform=new_transform,
        compress="deflate", tiled=True,
        blockxsize=512, blockysize=512,
    )

    with rio.open(out_path, "w", **profile) as dst:
        dst.write(out)
        dst.update_tags(
            AI_GENERATED="true",
            SR_MODEL="RRDBNet-x4",
            SR_SCALE=str(scale),
            SOURCE=str(in_path),
        )

    print(f"Wrote {out_path}  {out_h}x{out_w}  {new_transform.a:.2f} m/px")
    return out_path