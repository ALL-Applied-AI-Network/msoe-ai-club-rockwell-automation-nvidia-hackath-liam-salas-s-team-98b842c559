"""Relabel video-5 stills with the measured clock fit and compare with the source frame labels.

    python cli/fluidmix/core/measure_video_clock.py --videos 5     # first: writes still_video_map.csv
    python cli/fluidmix/core/align_still_labels.py

The vial windows in video5_vial_recipes.csv are in video seconds (checked against the video).
The source video5_frame_labels.csv places stills with video_s = wall time - first still, which
assumes 1 wall second per video second; the burned-in clock shows ~1.2158. This script uses the
measured mapping instead and writes FLUIDMIX_OUTPUTS/video5_frame_labels_aligned.csv.
boundary_margin_s is the distance to the nearest fill start/end; label boundaries were placed by
eye, so treat stills within ~2 s of a boundary as uncertain.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fluidmix.core.paths import outputs_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cli.fluidmix.core.inventory_data import find_dataset

COPY = ("recipe_id", "type", "t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", help="csi-vial-fill-dataset folder (default: under the data root)")
    ap.add_argument("--camera", default="vial_fill", help="still camera to label (all share timestamps)")
    a = ap.parse_args()
    ds = find_dataset(a.dataset)
    out = outputs_root()
    mapping = out / "still_video_map.csv"
    if not mapping.is_file():
        raise SystemExit(f"{mapping} not found; run python cli/fluidmix/core/measure_video_clock.py first")
    labels = ds / "vial_fill/labels"
    vials = list(csv.DictReader((labels / "video5_vial_recipes.csv").open()))
    source = {r["filename"].replace("vial_fill_", ""): r
              for r in csv.DictReader((labels / "video5_frame_labels.csv").open())}
    stills = [r for r in csv.DictReader(mapping.open()) if r["camera"] == a.camera and r["video_id"] == "5"]
    if not stills:
        raise SystemExit(f"no video-5 {a.camera} stills in {mapping}")
    bounds = [float(v[k]) for v in vials for k in ("start_s", "end_s")]
    rows = []
    for s in stills:
        src = source.get(s["filename"].replace(f"{a.camera}_", ""), {})
        row = {"filename": s["filename"], "timestamp": s["timestamp"], "status": s["status"],
               "video_s_source": src.get("video_s", ""), "video_s": s["video_s"], "frame": s["frame"],
               "in_fill": "", "vial": "", "boundary_margin_s": "",
               "source_in_fill": src.get("in_fill", ""), "source_vial": src.get("vial", "")}
        row.update({k: "" for k in COPY})
        if s["status"] == "in_video":
            t = float(s["video_s"])
            hit = next((v for v in vials if float(v["start_s"]) <= t <= float(v["end_s"])), None)
            row.update(in_fill=int(bool(hit)), vial=hit["vial"] if hit else "",
                       boundary_margin_s=round(min(abs(t - b) for b in bounds), 2))
            if hit:
                row.update({k: hit[k] for k in COPY})
        rows.append(row)
    dest = out / "video5_frame_labels_aligned.csv"
    with dest.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    placed = [r for r in rows if r["status"] == "in_video"]
    same_fill = sum(str(r["in_fill"]) == r["source_in_fill"] for r in placed)
    same_vial = sum(str(r["vial"]) == r["source_vial"] for r in placed)
    src_fill = [r for r in placed if r["source_in_fill"] == "1"]
    src_fill_right = sum(r["vial"] == r["source_vial"] for r in src_fill)
    covered = sorted({int(r["vial"]) for r in placed if r["vial"]})
    per_vial = Counter(int(r["vial"]) for r in placed if r["vial"])
    print(f"wrote {dest}")
    print(f"stills: {len(rows)}; placed in video: {len(placed)}; other: "
          f"{dict(Counter(r['status'] for r in rows if r['status'] != 'in_video'))}")
    print(f"aligned in_fill: {sum(r['in_fill'] == 1 for r in placed)} (source: "
          f"{sum(r['source_in_fill'] == '1' for r in rows)})")
    print(f"agreement with source labels: in_fill {same_fill}/{len(placed)}, vial {same_vial}/{len(placed)}")
    print(f"source 'in fill' stills whose vial is still right after alignment: {src_fill_right}/{len(src_fill)}")
    print(f"stills within 2 s of a fill boundary: "
          f"{sum(r['boundary_margin_s'] != '' and r['boundary_margin_s'] < 2 for r in placed)}")
    print(f"max video_s covered by stills: {max(float(r['video_s']) for r in placed):.1f}")
    print(f"vials with stills: {covered}")
    print(f"vials without stills: {sorted(set(range(1, len(vials) + 1)) - set(covered))}")
    print(f"stills per vial: {dict(sorted(per_vial.items()))}")


if __name__ == "__main__":
    main()
