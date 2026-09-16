import json
import uuid
import pathlib
import sys

import numpy as np
import torch
import rasterio as rio
from rasterio.warp import transform_bounds
from PIL import Image
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.models.rrdbnet import RRDBNet
from src.uncertainty import uncertainty_map
from src.inference import super_resolve

BASE = pathlib.Path(__file__).resolve().parent
DEMOS = BASE / "static" / "demos"
UPLOADS = BASE / "static" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

CKPT = BASE.parent / "checkpoints" / "satsr_finetuned.pth"
DIV = 3000.0
ALLOWED = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = RRDBNet(3, 3, 64, 23, 32)
model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
model = model.to(device).eval()
print("Model loaded on", device)


def unc_png(unc, path):
    import matplotlib.cm as cm
    u = unc / max(unc.max(), 1e-6)
    Image.fromarray((cm.inferno(u)[..., :3] * 255).astype(np.uint8)).save(path)


def geo_info(path):
    """Return lat/lon bounds, CRS and pixel size, or None."""
    try:
        with rio.open(path) as s:
            if s.crs is None:
                return None
            w, sth, e, n = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
            return {
                "bounds": [[sth, w], [n, e]],
                "center": [(sth + n) / 2, (w + e) / 2],
                "crs": str(s.crs),
                "px_in": round(abs(s.transform.a), 2),
                "px_out": round(abs(s.transform.a) / 4, 2),
            }
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/demos")
def demos():
    with open(DEMOS / "metrics.json") as f:
        data = json.load(f)
    for d in data:
        d["geo"] = geo_info(DEMOS / f"{d['id']}_output.tif")
    return jsonify(data)


@app.route("/api/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    f = request.files["file"]
    ext = pathlib.Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported type {ext}. Use TIF, PNG or JPG."}), 400

    uid = uuid.uuid4().hex[:8]
    in_path = UPLOADS / f"{uid}_in{ext}"
    f.save(in_path)
    is_tif = ext in {".tif", ".tiff"}

    try:
        if is_tif:
            with rio.open(in_path) as s:
                arr = np.clip(s.read()[:3].astype(np.float32) / DIV, 0, 1)
        else:
            img = Image.open(in_path).convert("RGB")
            arr = np.asarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0

        h, w = arr.shape[1], arr.shape[2]
        if h > 256 or w > 256:
            return jsonify({"error": f"Image too large ({w}x{h}). Max 1024x1024."}), 400

        x = torch.from_numpy(arr).unsqueeze(0)
        sr, unc = uncertainty_map(model, x, device)

        Image.fromarray((arr.transpose(1, 2, 0) * 255).astype(np.uint8)) \
            .resize((sr.shape[2], sr.shape[1]), Image.NEAREST) \
            .save(UPLOADS / f"{uid}_before.png")

        Image.fromarray((np.clip(sr.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)) \
            .save(UPLOADS / f"{uid}_after.png")

        unc_png(unc, UPLOADS / f"{uid}_uncertainty.png")

        tif_url, geo = None, None
        if is_tif:
            geo = geo_info(in_path)
            try:
                super_resolve(str(in_path), str(UPLOADS / f"{uid}_output.tif"),
                              model, device=device)
                tif_url = f"/static/uploads/{uid}_output.tif"
            except Exception as e:
                print("GeoTIFF failed:", e)

        return jsonify({
            "id": uid,
            "before": f"/static/uploads/{uid}_before.png",
            "after": f"/static/uploads/{uid}_after.png",
            "uncertainty": f"/static/uploads/{uid}_uncertainty.png",
            "reference": None,
            "tif": tif_url,
            "geo": geo,
            "uncertainty_mean": round(float(unc.mean()), 4),
            "size": f"{w}x{h} to {sr.shape[2]}x{sr.shape[1]}",
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=7860)