"""Correct video-5 timing; validate input and replace outputs only after generation."""
import csv
import datetime as D
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import VIAL_FILL, data_root, outputs_root, require

S = data_root(sys.argv[1] if len(sys.argv) > 1 else None)
R = require(S / VIAL_FILL, "vial_fill dataset folder")
O = outputs_root()
T0 = D.datetime(2026, 9, 23, 19, 47, 11, 161000)
K = (D.datetime(2026, 9, 23, 20, 21, 33, 830000)-T0).total_seconds()/1697.0
with (R / "labels/video5_vial_recipes.csv").open() as stream:
    vials = list(csv.DictReader(stream))
with (R / "image_index.csv").open() as stream:
    idx = [row for row in csv.DictReader(stream) if row["video_id"] == "5"]
if not vials or not idx:
    raise SystemExit("video-5 recipes and stills are required; previous outputs were preserved")
out, last = [], {}
for row in idx:
    seconds = (D.datetime.fromisoformat(row["timestamp"])-T0).total_seconds()/K
    hit = next((v for v in vials if float(v["start_s"]) <= seconds <= float(v["end_s"])), None)
    corrected = {"filename": row["filename"], "timestamp": row["timestamp"],
                 "video_s_corrected": round(seconds, 3), "in_fill": 1 if hit else 0,
                 "vial": hit["vial"] if hit else "", "recipe_id": hit["recipe_id"] if hit else ""}
    for column in ("type", "t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order"):
        corrected[column] = hit[column] if hit else ""
    out.append(corrected)
    if hit:
        last[int(hit["vial"])] = row["filename"]
with tempfile.TemporaryDirectory(prefix=".labels-", dir=O) as scratch:
    labels = Path(scratch) / "labels.csv"
    with labels.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    mapping = Path(scratch) / "last.tsv"
    mapping.write_text("\n".join(f"{k}\t{v}" for k, v in sorted(last.items())))
    labels.replace(O / "video5_frame_labels_corrected.csv")
    mapping.replace(O / "last_corrected.tsv")
print("K=%.5f  real fps=%.2f" % (K, 30/K))
print("frames", len(out), "in_fill", sum(row["in_fill"] for row in out))
print("vials with stills", sorted(last))
print("max video t covered by stills %.1f" % max(row["video_s_corrected"] for row in out))
