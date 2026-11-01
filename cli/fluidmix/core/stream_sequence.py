"""INN-52: find dispense bursts and their stream colour in each video-5 vial (inspection only).

    python cli/fluidmix/core/stream_sequence.py                 # all 30 vials, writes to outputs/inn52/
    python cli/fluidmix/core/stream_sequence.py --vials 10 25   # a subset
    python cli/fluidmix/core/stream_sequence.py --video PATH    # explicit vial_fill_video_5.mp4

Each tank has its own tube, but the tubes are out of frame and the camera sees one outlet, so it
cannot show which tube flows. What it can show is the stream below the nozzle tip. The tip ends
at about the vial rim, so the whole visible stream is seen through the vial wall (rows 95-190,
above the black holder). See docs/stream.md for the measurements behind these constants.

Per frame: luma difference from a pre-pour reference frame, per row. A row holds the stream
when its peak difference is at least LUMA_MIN, lies under the nozzle, and is 2-STREAM_MAX_W px
wide at half maximum (wider blobs are bubbles, splashes or liquid, not the stream). A frame is
flowing when at least MIN_ROWS rows hold the stream. Its colour is the median CIELAB shift
(stream core minus reference) over those rows.

Flowing frames separated by at most MAX_GAP frames form one burst; bursts shorter than
MIN_BURST frames are dropped. Each burst gets a colour name: clear when the a*b* shift is below
CLEAR_MAX, orange when it is mostly +b*, blue when it is mostly -a* (the blue tank reads cyan).
Then, per vial, three checks against the recipe log (never the other way round, see
docs/labels.md):
  - burst count vs number of tanks in the recipe;
  - burst duration per recipe mL, for each tank (a steady per-tank rate supports the logged
    order and amounts; the burst is matched to the tank in the same position);
  - burst colours vs two hypotheses: "current" (each burst shows the tank now dispensing) and
    "previous" (each burst shows the previous dose, pushed out of a section the tubes share). The previous
    dose of a vial's first burst is the last tank of the previous vial.

Thresholds were set by looking at these same 30 vials. Counts describe whether the stream
carries usable sequence information; they are not a held-out evaluation. Needs ffmpeg.
"""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fluidmix.core.paths import REPO, VIAL_FILL, data_root, outputs_root

FPS = 30
X0, W, H = 270, 80, 200          # crop: nozzle at x 293-315, vial rim y ~75, holder from y ~198
ROW0, ROW1 = 95, 190             # stream rows inside the vial, above the holder
NOZZLE_X = (285, 325)            # stream peak must lie under the nozzle
REF_BEFORE = 15                  # reference frame: this many frames before the labelled start
PAD_AFTER = 15
LUMA_MIN = 8.0                   # 0-255 luma difference that counts as a stream row
STREAM_MAX_W = 12                # px at half maximum; the stream measures 5-8 px
MIN_ROWS = 30                    # of 95 rows
MAX_GAP = 15                     # frames of no stream allowed inside one burst
MIN_BURST = 15                   # frames (0.5 video s)
CLEAR_MAX = 4.0                  # a*b* shift below this is clear liquid
TANK_COLOUR = {"T100": "clear", "T200": "orange", "T400": "blue"}
TANK_ML = {"T100": "t100_ml", "T200": "t200_ml", "T400": "t400_ml"}
DRAW = {"none": (40, 40, 40), "clear": (230, 230, 230), "orange": (0, 120, 255), "blue": (255, 150, 0)}


def read_frames(mp4, first, count):
    """BGR crops (count, H, W, 3) starting at video frame `first`."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{first / FPS:.4f}", "-i", str(mp4),
                          "-frames:v", str(count), "-vf", f"crop={W}:{H}:{X0}:0",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def fwhm(profile):
    """Width at half maximum around the peak, and the peak index."""
    k = int(np.argmax(profile))
    half = profile[k] / 2
    left = k
    while left > 0 and profile[left - 1] > half:
        left -= 1
    right = k
    while right < len(profile) - 1 and profile[right + 1] > half:
        right += 1
    return right - left + 1, k


def stream_rows(luma, ref_luma):
    """[(row, peak column, width)] for rows that look like the stream."""
    diff = np.abs(luma.astype(np.float32) - ref_luma.astype(np.float32))
    lo, hi = NOZZLE_X[0] - X0, NOZZLE_X[1] - X0
    rows = []
    for y in range(ROW0, ROW1):
        profile = diff[y]
        width, k = fwhm(profile)
        if profile[k] >= LUMA_MIN and lo <= k < hi and 2 <= width <= STREAM_MAX_W:
            rows.append((y, k, width))
    return rows


def colour_name(da, db):
    if np.hypot(da, db) < CLEAR_MAX:
        return "clear"
    return "orange" if db >= -da else "blue"


def measure(frame, ref, ref_lab):
    luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = stream_rows(luma, cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    if len(rows) < MIN_ROWS:
        return {"rows": len(rows), "width": np.nan, "da": np.nan, "db": np.nan}
    lab = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    ys = np.array([r[0] for r in rows])
    xs = np.clip(np.array([r[1] for r in rows])[:, None] + np.array([-1, 0, 1]), 0, W - 1)
    shift = (lab[ys[:, None], xs] - ref_lab[ys[:, None], xs]).reshape(-1, 3)
    return {"rows": len(rows), "width": float(np.median([r[2] for r in rows])),
            "da": float(np.median(shift[:, 1])), "db": float(np.median(shift[:, 2]))}


def bursts(flowing, max_gap=MAX_GAP, min_len=MIN_BURST):
    """[(first, last)] index ranges of flowing frames, joining gaps of at most max_gap frames."""
    out = []
    for i in np.flatnonzero(flowing):
        if out and i - out[-1][1] <= max_gap + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out if b - a + 1 >= min_len]


def hypotheses(order, prev_last):
    """Expected burst colours if bursts show the current dose, or the previous one."""
    current = [TANK_COLOUR[t] for t in order]
    previous = [TANK_COLOUR[prev_last] if prev_last else None] + current[:-1]
    return current, previous


def agreement(seen, expected):
    """Matching positions, counting only positions where both are known and counts agree."""
    if len(seen) != len(expected):
        return None
    return sum(s == e for s, e in zip(seen, expected) if e is not None)


def vial_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="vial_fill_video_5.mp4 (default: under the data root)")
    ap.add_argument("--vials", type=int, nargs="*", help="vial numbers (default: all 30)")
    a = ap.parse_args()
    mp4 = Path(a.video) if a.video else data_root() / VIAL_FILL / "videos/vial_fill_video_5.mp4"
    if not mp4.is_file():
        raise SystemExit(f"missing {mp4}; see docs/data.md")
    out = outputs_root() / "inn52"
    out.mkdir(parents=True, exist_ok=True)
    every = vial_rows(REPO / "data/video5_vial_recipes.csv")
    prev_last = {v["vial"]: (every[i - 1]["dispense_order"].split(">")[-1] if i else None)
                 for i, v in enumerate(every)}
    vials = [v for v in every if not a.vials or int(v["vial"]) in a.vials]

    frame_out, burst_out, vial_out, strips, seconds, n_frames = [], [], [], [], 0.0, 0
    for v in vials:
        start, end = int(v["start_frame"]), int(v["end_frame"])
        ref_frame = max(0, start - REF_BEFORE)
        frames = read_frames(mp4, ref_frame, end + PAD_AFTER - ref_frame + 1)
        ref = frames[0]
        ref_lab = cv2.cvtColor(ref.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        t0 = time.perf_counter()
        meas = [measure(f, ref, ref_lab) for f in frames]
        seconds += time.perf_counter() - t0
        n_frames += len(frames)
        flowing = np.array([not np.isnan(m["da"]) for m in meas])
        labels = [colour_name(m["da"], m["db"]) if f else "none" for m, f in zip(meas, flowing)]
        for i, (m, lab) in enumerate(zip(meas, labels)):
            frame_out.append({"vial": v["vial"], "frame": ref_frame + i, "in_window": int(start <= ref_frame + i <= end),
                              "stream_rows": m["rows"], "width_px": f"{m['width']:.1f}",
                              "dA": f"{m['da']:.2f}", "dB": f"{m['db']:.2f}", "colour": lab})

        order = v["dispense_order"].split(">")
        seen, durations = [], []
        for k, (b0, b1) in enumerate(bursts(flowing)):
            idx = [i for i in range(b0, b1 + 1) if flowing[i]]
            da = float(np.median([meas[i]["da"] for i in idx]))
            db = float(np.median([meas[i]["db"] for i in idx]))
            seen.append(colour_name(da, db))
            durations.append(b1 - b0 + 1)
            burst_out.append({"vial": v["vial"], "burst": k + 1, "first_frame": ref_frame + b0,
                              "last_frame": ref_frame + b1, "frames": b1 - b0 + 1, "flowing_frames": len(idx),
                              "dA": f"{da:.2f}", "dB": f"{db:.2f}", "colour": seen[-1],
                              "width_px": f"{np.median([meas[i]['width'] for i in idx]):.1f}",
                              "tank": "", "recipe_ml": "", "frames_per_ml": ""})
        current, previous = hypotheses(order, prev_last[v["vial"]])
        ml = [float(v[TANK_ML[t]]) for t in order]
        if len(durations) == len(order):
            for row, tank, mls in zip(burst_out[-len(durations):], order, ml):
                row.update(tank=tank, recipe_ml=f"{mls:g}", frames_per_ml=f"{row['frames'] / mls:.2f}")
        cur, prv = agreement(seen, current), agreement(seen, previous)
        vial_out.append({"vial": v["vial"], "recipe_id": v["recipe_id"], "dispense_order": v["dispense_order"],
                         "tanks": len(order), "bursts": len(seen), "count_match": int(len(seen) == len(order)),
                         "burst_frames": ">".join(map(str, durations)), "recipe_ml": ">".join(f"{x:g}" for x in ml),
                         "colours": ">".join(seen) or "-",
                         "current_hypothesis": ">".join(current), "previous_hypothesis": ">".join(c or "?" for c in previous),
                         "current_matches": "" if cur is None else cur,
                         "previous_matches": "" if prv is None else prv,
                         "previous_known": sum(c is not None for c in previous),
                         "no_prepour_reference": int(start - REF_BEFORE < 0)})
        strips.append((v["vial"], labels, start - ref_frame, end - ref_frame))
        print(f"vial {v['vial']:>2} {v['dispense_order']:<15} bursts {len(seen)}/{len(order)} "
              f"frames {'>'.join(map(str, durations)):<14} mL {'>'.join(f'{x:g}' for x in ml):<9} "
              f"colours {'>'.join(seen) or '-':<20} current {cur} previous {prv}")

    for name, rows in (("stream_frames.csv", frame_out), ("stream_bursts.csv", burst_out),
                       ("stream_vials.csv", vial_out)):
        with (out / name).open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    longest = max(len(s[1]) for s in strips)
    img = np.zeros((len(strips) * 14, longest // 2 + 60, 3), np.uint8)
    for r, (vial, labels, s, e) in enumerate(strips):
        y = r * 14
        cv2.putText(img, vial, (2, y + 11), 0, 0.35, (255, 255, 255), 1)
        for i, lab in enumerate(labels[::2]):
            cv2.line(img, (60 + i, y + 2), (60 + i, y + 11), DRAW[lab], 1)
        for f in (s, e):
            cv2.line(img, (60 + f // 2, y), (60 + f // 2, y + 13), (0, 0, 255), 1)
    cv2.imwrite(str(out / "stream_timelines.png"), img)

    usable = [r for r in vial_out if not r["no_prepour_reference"]]
    counted = [r for r in usable if r["count_match"]]
    ok = {r["vial"] for r in counted}
    for tank in TANK_COLOUR:
        fpm = [float(b["frames_per_ml"]) for b in burst_out if b["tank"] == tank and b["vial"] in ok]
        print(f"{tank}: {len(fpm)} bursts, {np.median(fpm):.2f} video frames per mL "
              f"(min {min(fpm):.2f}, max {max(fpm):.2f}, CV {np.std(fpm) / np.mean(fpm):.1%})")
    cur = sum(int(r["current_matches"]) for r in counted)
    prv = sum(int(r["previous_matches"]) for r in counted)
    print(f"{len(usable)} vials with a pre-pour reference: burst count = tank count in {len(counted)}; "
          f"colour matches over those "
          f"{len(counted)} vials: current {cur}/{sum(r['tanks'] for r in counted)}, previous {prv}/"
          f"{sum(r['previous_known'] for r in counted)}; {1000 * seconds / n_frames:.2f} ms per frame "
          f"(measurement only, excluding decode); wrote {out}")


if __name__ == "__main__":
    main()
