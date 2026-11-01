"""Inventory Dataset 2 (CSI vial-fill): one row per camera and video ID.

    python cli/fluidmix/core/inventory_data.py                 # print the table
    python cli/fluidmix/core/inventory_data.py --write         # also write data/dataset_inventory.csv
    python cli/fluidmix/core/inventory_data.py <dataset dir>   # explicit csi-vial-fill-dataset folder

Reads only the raw folder. Video metadata needs ffprobe; without it those columns are "unknown".
Anything the dataset does not tell us is written as "unknown" in missing_unknown, not guessed.
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fluidmix.core.paths import REPO, data_root

CAMERAS = ("cap_seal", "tank_level", "vial_fill")
OUT = REPO / "data/dataset_inventory.csv"
FIELDS = [
    "camera", "video_id", "video_file", "video_bytes", "duration_s", "frames", "fps", "resolution",
    "codec", "n_stills", "first_still", "last_still", "still_span_s", "n_vials_reported", "labeled",
    "label_files", "recipe_ids", "license", "missing_unknown",
]


def find_dataset(explicit):
    if explicit:
        return Path(explicit).resolve()
    root = data_root()
    # The organizer archive extracts to <name>; older copies used ds/<name>.
    for rel in ("csi-vial-fill-dataset", "ds/csi-vial-fill-dataset"):
        if (root / rel / "videos.csv").is_file():
            return root / rel
    raise SystemExit(f"no csi-vial-fill-dataset with videos.csv under {root}; see docs/data.md")


def probe(path):
    if not shutil.which("ffprobe"):
        return {}
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    s, f = info["streams"][0], info["format"]
    num, den = s["r_frame_rate"].split("/")
    return {
        "duration_s": f"{float(f['duration']):.3f}", "frames": s.get("nb_frames", ""),
        "fps": f"{int(num) / int(den):g}", "resolution": f"{s['width']}x{s['height']}",
        "codec": s["codec_name"],
    }


def inventory(ds):
    videos = {r["video_id"]: r for r in csv.DictReader((ds / "videos.csv").open())}
    labels = ds / "vial_fill/labels"
    recipes = sorted({int(r["recipe_id"]) for r in csv.DictReader((labels / "video5_vial_recipes.csv").open())})
    has_license = any(p.name.lower().startswith(("license", "licence")) for p in ds.iterdir())
    rows, extras = [], []
    for cam in CAMERAS:
        stills = defaultdict(list)
        index = list(csv.DictReader((ds / cam / "image_index.csv").open()))
        for r in index:
            stills[r["video_id"]].append(datetime.fromisoformat(r["timestamp"]))
        on_disk = {p.name for p in (ds / cam / "images").glob("*.jpg")}
        indexed = {r["filename"] for r in index}
        extras += [f"{cam}/images/{n}" for n in sorted(on_disk - indexed)]
        extras += [f"{cam}/image_index.csv lists missing {n}" for n in sorted(indexed - on_disk)]
        if stills.get("unassigned"):
            extras.append(f"{cam}: {len(stills['unassigned'])} still(s) with video_id=unassigned")
        for vid in sorted(videos):
            meta = videos[vid]
            mp4 = ds / cam / "videos" / f"{cam}_video_{vid}.mp4"
            ts = sorted(stills.get(vid, []))
            labeled = cam == "vial_fill" and vid == "5"
            missing = []
            if not labeled:
                missing.append("recipes, vial fill windows and defect labels unknown")
            else:
                missing.append("no defect labels (recipe/fill-window only); prior recipe before vial 1 unknown")
            if vid == "5":
                missing.append("timing measured: 1.2157 wall s per video s, so stills are missing for the last ~18% of each recording (see docs/video_timing.md)")
            if len(ts) < 0.25 * float(meta["duration_s"]) / 2:
                missing.append(f"sparse stills ({len(ts)})")
            if not has_license:
                missing.append("license not stated")
            row = {
                "camera": cam, "video_id": vid,
                "video_file": mp4.relative_to(ds).as_posix() if mp4.is_file() else "MISSING",
                "video_bytes": mp4.stat().st_size if mp4.is_file() else "",
                "duration_s": "unknown", "frames": "unknown", "fps": "unknown",
                "resolution": "unknown", "codec": "unknown",
                "n_stills": len(ts),
                "first_still": ts[0].isoformat(sep=" ") if ts else "",
                "last_still": ts[-1].isoformat(sep=" ") if ts else "",
                "still_span_s": f"{(ts[-1] - ts[0]).total_seconds():.1f}" if ts else "",
                "n_vials_reported": meta["n_vials"],
                "labeled": "yes" if labeled else "no",
                "label_files": "vial_fill/labels/video5_vial_recipes.csv;.json;video5_frame_labels.csv" if labeled else "",
                "recipe_ids": f"{recipes[0]}-{recipes[-1]}" if labeled or vid == "5" else "unknown",
                "license": "stated" if has_license else "not stated",
                "missing_unknown": "; ".join(missing),
            }
            if mp4.is_file():
                row.update(probe(mp4))
            rows.append(row)
    return rows, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", help="csi-vial-fill-dataset folder (default: under the data root)")
    ap.add_argument("--write", action="store_true", help=f"write {OUT.relative_to(REPO)}")
    a = ap.parse_args()
    ds = find_dataset(a.dataset)
    print(f"dataset: {ds}")
    rows, extras = inventory(ds)
    for r in rows:
        print(f"{r['camera']:<10} v{r['video_id']} {r['duration_s']:>9}s {r['frames']:>6}f "
              f"stills={r['n_stills']:<4} span={r['still_span_s']:>7}s vials={r['n_vials_reported']:<3} "
              f"labeled={r['labeled']}")
    print("files outside the index / other anomalies:")
    for e in extras or ["none"]:
        print(f"  {e}")
    if a.write:
        with OUT.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
