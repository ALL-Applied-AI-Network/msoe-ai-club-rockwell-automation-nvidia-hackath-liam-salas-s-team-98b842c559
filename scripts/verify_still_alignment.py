"""Check the still-to-video mapping by pixel matching, independent of the clock OCR.

    python scripts/measure_video_clock.py        # first: writes still_video_map.csv
    python scripts/verify_still_alignment.py --per-video 12

For sampled stills, compare the still with every video frame within --window frames of its
mapped frame (grey, mean absolute difference, clock overlay rows excluded) and report the
offset of the best match. A still of a static scene matches many frames equally, so a sample
only counts as informative when its best match beats every frame more than SHARP_FRAMES away by
at least MIN_CONTRAST grey levels. Writes FLUIDMIX_OUTPUTS/still_alignment_check.csv.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import outputs_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventory_data import CAMERAS, find_dataset

FPS = 30
ROWS = 430          # skip the bottom rows that carry the clock overlay
SHARP_FRAMES = 5
MIN_CONTRAST = 0.3  # grey levels


def frames(mp4, start, n):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start / FPS:.4f}", "-i", str(mp4), "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    k = len(raw) // (640 * 480)
    return np.frombuffer(raw, np.uint8)[: k * 640 * 480].reshape(k, 480, 640)[:, :ROWS].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", help="csi-vial-fill-dataset folder (default: under the data root)")
    ap.add_argument("--per-video", type=int, default=12, help="stills sampled per camera and video")
    ap.add_argument("--window", type=int, default=90, help="frames searched either side of the mapped frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    ap.add_argument("--videos", nargs="+", default=["1", "2", "3", "4", "5"])
    a = ap.parse_args()
    ds = find_dataset(a.dataset)
    out = outputs_root()
    mapping = out / "still_video_map.csv"
    if not mapping.is_file():
        raise SystemExit(f"{mapping} not found; run python scripts/measure_video_clock.py first")
    stills = [r for r in csv.DictReader(mapping.open()) if r["status"] == "in_video"]
    rng = np.random.default_rng(a.seed)
    results = []
    for cam in a.cameras:
        for vid in sorted({r["video_id"] for r in stills if r["camera"] == cam} & set(a.videos)):
            pool = [r for r in stills if r["camera"] == cam and r["video_id"] == vid]
            picks = [pool[i] for i in sorted(rng.choice(len(pool), min(a.per_video, len(pool)), replace=False))]
            mp4 = ds / cam / "videos" / f"{cam}_video_{vid}.mp4"
            for r in picks:
                f = int(r["frame"])
                f0 = max(0, f - a.window)
                d = np.abs(frames(mp4, f0, 2 * a.window + 1)
                           - np.asarray(Image.open(ds / cam / "images" / r["filename"]).convert("L"))[:ROWS]).mean(axis=(1, 2))
                best = int(np.argmin(d))
                far = np.abs(np.arange(len(d)) - best) > SHARP_FRAMES
                contrast = float(d[far].min() - d[best]) if far.any() else 0.0
                results.append({"camera": cam, "video_id": vid, "filename": r["filename"], "mapped_frame": f,
                                "best_offset_frames": f0 + best - f, "best_diff": round(float(d[best]), 2),
                                "contrast": round(contrast, 2), "informative": int(contrast >= MIN_CONTRAST)})
            sel = [x for x in results if x["camera"] == cam and x["video_id"] == vid and x["informative"]]
            offs = [x["best_offset_frames"] for x in sel]
            print(f"{cam:<10} v{vid}: {len(picks)} sampled, {len(sel)} informative; offsets (frames) {sorted(offs)}")
    dest = out / "still_alignment_check.csv"
    with dest.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    inf = [x["best_offset_frames"] for x in results if x["informative"]]
    within = sum(abs(o) <= 2 for o in inf)
    print(f"wrote {dest}")
    print(f"informative samples: {len(inf)}/{len(results)}; within ±2 frames: {within}; "
          f"median |offset| {np.median(np.abs(inf)) if inf else float('nan'):.1f} frames")


if __name__ == "__main__":
    main()
