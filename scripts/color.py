"""Extract rough vial colors; abort failed extraction and preserve prior results."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import VIAL_FILL, data_root, outputs_root, require

S = data_root(sys.argv[1] if len(sys.argv) > 1 else None)
R = require(S / VIAL_FILL, "vial_fill dataset folder")
O = outputs_root()


def lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055)**2.4, c / 12.92)
    matrix = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])
    xyz = c @ matrix.T / np.array([0.95047, 1, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787*xyz + 16/116)
    return np.stack([116*f[..., 1]-16, 500*(f[..., 0]-f[..., 1]),
                     200*(f[..., 1]-f[..., 2])], -1)


with (R / "labels/video5_vial_recipes.csv").open() as stream:
    vials = list(csv.DictReader(stream))
if not vials:
    raise SystemExit("no vial recipes; previous colors were preserved")
res = []
with tempfile.TemporaryDirectory(prefix=".color-", dir=O) as scratch:
    frame = Path(scratch) / "frame.png"
    for vial in vials:
        acc = []
        for offset in (15, 30, 45):
            frame.unlink(missing_ok=True)
            number = int(vial["end_frame"]) + offset
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-ss", f"{number/30:.3f}",
                "-i", str(R / "videos/vial_fill_video_5.mp4"),
                "-frames:v", "1", str(frame),
            ], check=True)
            with Image.open(frame) as image:
                pixels = np.asarray(image.convert("RGB"), np.float32)[150:430, 180:420]
            mx, mn = pixels.max(-1), pixels.min(-1)
            saturation = (mx-mn)/(mx+1)
            mask = saturation > np.percentile(saturation, 99.3)
            if not mask.any():
                raise ValueError(f"no selected liquid pixels for vial {vial['vial']}")
            acc.append(np.median(pixels[mask], 0))
        rgb = np.median(np.array(acc), 0)
        if not np.isfinite(rgb).all():
            raise ValueError(f"nonfinite color for vial {vial['vial']}")
        res.append({"vial": int(vial["vial"]), "recipe": int(vial["recipe_id"]),
                    "rgb": [int(x) for x in rgb],
                    "lab": [round(float(x), 1) for x in lab(rgb)]})
    for row in res:
        last = [x for x in res if x["recipe"] == row["recipe"]][-1]
        row["dE_vs_last"] = round(float(np.linalg.norm(
            np.array(row["lab"])-np.array(last["lab"]))), 1)
    output = Path(scratch) / "colors.json"
    output.write_text(json.dumps(res, allow_nan=False))
    output.replace(O / "colors.json")
for row in res:
    print(row["vial"], row["recipe"], row["rgb"], row["dE_vs_last"])
