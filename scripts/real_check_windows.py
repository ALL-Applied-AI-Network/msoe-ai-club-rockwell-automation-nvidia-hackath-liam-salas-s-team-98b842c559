"""Find each video-5 vial's final-mixture window: after its fill ends, before its cart moves.

    python scripts/real_check_windows.py            # check data/real_check_windows.csv is current
    python scripts/real_check_windows.py --write

Tanks dispense one after another (T200 > T400 > T100), so a vial shows its final mixture only
after the last tank stops. The filled vial then leaves the camera view within seconds, often
between two ~2 s stills, so the window is measured on video frames (30 per video second).

For each vial: take the frame END_OFFSET_S after the labelled fill end (end_s) as the reference,
and mark the cart as moving at the first later frame whose mean grey difference from the reference
exceeds MOVE_GREY (frames downscaled to 160x120, clock rows excluded). The window is
[end_s + END_OFFSET_S, departure - MARGIN_S]. Windows shorter than MIN_FRAMES are kept but marked
ineligible. Needs ffmpeg and the vial_fill video 5; reads data/video5_transitions.csv.
"""
import argparse
import csv
import io
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import REPO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventory_data import find_dataset

FPS = 30
END_OFFSET_S = 0.3   # let the last drops fall after the labelled fill end
MARGIN_S = 0.1       # stop before the first frame that moves
MOVE_GREY = 6.0      # mean absolute grey difference (0-255) that counts as the cart moving
SEARCH_S = 7.0       # look this far past end_s for the departure
MIN_FRAMES = 3
W, H, ROWS = 160, 120, 100
OUT = REPO / "data/real_check_windows.csv"


def departure(mp4, end_s):
    start = end_s - 1.0
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(mp4), "-t", str(SEARCH_S + 1),
                          "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W)[:, :ROWS].astype(np.float32)
    rel = np.arange(len(frames)) / FPS + start - end_s
    ref = frames[int(round((1.0 + END_OFFSET_S) * FPS))]
    diff = np.abs(frames - ref).mean(axis=(1, 2))
    moving = np.flatnonzero((rel > END_OFFSET_S) & (diff > MOVE_GREY))
    return float(rel[moving[0]]) if len(moving) else None


def build(ds):
    mp4 = ds / "vial_fill/videos/vial_fill_video_5.mp4"
    rows = []
    for t in csv.DictReader((REPO / "data/video5_transitions.csv").open()):
        end_s = float(t["end_s"])
        dep = departure(mp4, end_s)
        row = {"video_id": t["video_id"], "vial": t["vial"], "recipe_id": t["recipe_id"],
               "prev_recipe_id": t["prev_recipe_id"], "recipe_changed": t["recipe_changed"],
               "end_s": f"{end_s:.3f}", "departure_after_end_s": "", "window_start_s": "", "window_end_s": "",
               "start_frame": "", "end_frame": "", "n_frames": 0, "role": "real_validation",
               "eligible": 0, "exclusion_reason": ""}
        if dep is None:
            row["exclusion_reason"] = "no_departure_found"
        else:
            w0, w1 = end_s + END_OFFSET_S, end_s + dep - MARGIN_S
            f0, f1 = int(np.ceil(w0 * FPS)), int(np.floor(w1 * FPS))
            row.update(departure_after_end_s=f"{dep:.3f}", window_start_s=f"{w0:.3f}", window_end_s=f"{w1:.3f}",
                       start_frame=f0, end_frame=f1, n_frames=max(0, f1 - f0 + 1))
            if t["prev_recipe_id"] == "unknown":
                row["exclusion_reason"] = "previous_recipe_unknown"
            elif row["n_frames"] < MIN_FRAMES:
                row["exclusion_reason"] = "window_too_short"
        row["eligible"] = int(row["exclusion_reason"] == "")
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", help="csi-vial-fill-dataset folder (default: under the data root)")
    ap.add_argument("--write", action="store_true", help=f"write {OUT.relative_to(REPO)}")
    a = ap.parse_args()
    rows = build(find_dataset(a.dataset))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    deps = [float(r["departure_after_end_s"]) for r in rows if r["departure_after_end_s"]]
    frames = [r["n_frames"] for r in rows if r["eligible"]]
    print(f"{len(rows)} vials; cart moves {min(deps):.2f}-{max(deps):.2f} s after end_s (median "
          f"{np.median(deps):.2f}); eligible {sum(r['eligible'] for r in rows)}, frames per window "
          f"{min(frames)}-{max(frames)}; excluded: "
          f"{ {r['vial']: r['exclusion_reason'] for r in rows if not r['eligible']} }")
    if a.write:
        OUT.write_text(buf.getvalue())
        print("wrote", OUT)
    elif not OUT.is_file() or OUT.read_text() != buf.getvalue():
        raise SystemExit(f"{OUT.name} is out of date; rerun with --write")
    else:
        print(f"{OUT.name} is up to date")


if __name__ == "__main__":
    main()
