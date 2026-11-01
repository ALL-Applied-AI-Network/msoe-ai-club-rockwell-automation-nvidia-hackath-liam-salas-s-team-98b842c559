"""INN-52: find dispense bursts and their stream colour in each video-5 vial (inspection only).

    python scripts/stream_sequence.py                 # all 30 vials, writes to outputs/inn52/
    python scripts/stream_sequence.py --vials 10 25   # a subset
    python scripts/stream_sequence.py --video PATH    # explicit vial_fill_video_5.mp4
    python scripts/stream_sequence.py --store         # also rewrite data/video5_stream_vial_colours.csv

Each tank has its own tube, and the tubes join into one shared dispensing tube before the outlet
(team, 2026-10-26). The tubes are out of frame and the camera sees only that outlet, so it cannot
show which tube flows. What it can show is the stream below the nozzle tip. The tip ends
at about the vial rim, so the whole visible stream is seen through the vial wall (rows 95-190,
above the black holder). See docs/stream.md for the measurements behind these constants.

Per frame: luma difference from a pre-pour reference frame, per row. A row holds the stream
when its peak difference is at least LUMA_MIN, lies under the nozzle, and is 2-STREAM_MAX_W px
wide at half maximum (wider blobs are bubbles, splashes or liquid, not the stream). A frame is
flowing when at least MIN_ROWS rows hold the stream. Its colour is measured, not named: the
median CIELAB shift (stream core minus reference) over those rows, its hue angle and chroma, and
the median sRGB of the stream core pixels.

Flowing frames separated by at most MAX_GAP frames form one burst. Bursts shorter than
MIN_BURST frames are dropped. Then, per vial, three checks against the recipe log (never the
other way round, see docs/labels.md):
  - burst count vs number of tanks in the recipe;
  - burst duration per recipe mL, for each tank (a steady per-tank rate supports the logged
    order and amounts; the burst is matched to the tank in the same position);
  - burst colours vs two hypotheses: "current" (each burst shows the tank now dispensing) and
    "previous" (each burst shows the previous dose, pushed out of the shared dispensing tube).
    The previous dose of a vial's first burst is the last tank of the previous vial. No tank
    colour is assumed: under each hypothesis, each tank's reference shift is the median of the
    bursts that hypothesis attributes to it in the other recipes' vials, and a burst matches
    when its nearest reference (a*b* distance) is the expected tank.

Every frame also gets the vial's colour shift and liquid level, and the burst it belongs to
(0 = none), so outputs/inn52/stream_frames.csv traces stream and vial through each dose; see
scripts/burst_trace.py. After each burst the vial's own colour is read too: the median sRGB of the liquid below the
holder, VIAL_AFTER frames after the burst, with the same region of the pre-pour frame as its
empty-vial reference. Stream and vial colour per burst, and the gap between them, are stored
with --store (see docs/data.md).

Thresholds were set by looking at these same 30 vials. Counts describe whether the stream
carries usable sequence information; they are not a held-out evaluation. Needs ffmpeg.
"""
import argparse
import csv
import itertools
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import REPO, VIAL_FILL, data_root, outputs_root

FPS = 30
X0, W, H = 270, 80, 400          # crop: nozzle at x 293-315, vial rim y ~75, holder, vial base y ~395
ROW0, ROW1 = 95, 190             # stream rows inside the vial, above the holder
VIAL_X, VIAL_Y = (292, 332), (320, 370)   # liquid below the holder; filled by 30 mL, the smallest dose total
VIAL_AFTER = 5                   # vial colour is read this many frames after a burst ends
FINAL_AFTER = 10                 # final vial colour: this many frames after the labelled end
VIAL_CHANGE = 6.0                # 0-255 mean RGB change that counts a vial-region row as changed
LEVEL_X = (296, 328)             # liquid-level column, inside the vial walls
HOLDER_ROWS = (240, 263)         # black holder: rows where the vial is hidden
LEVEL_BOTTOM = 395               # vial base
LEVEL_FRAC = 0.2                 # liquid row: change >= this share of the column's 90th percentile below the holder
LEVEL_FLOOR = 6.0                # ... and at least this (0-255), so an empty vial reads no liquid
LEVEL_STOP = 15                  # rows of no liquid that end the scan up from the base; bridges the dark
                                 # conveyor band behind the vial (y ~300-330), where liquid changes little
# LEVEL_FRAC and LEVEL_STOP were chosen on half the recipes and tested on the other half (docs/stream.md).
NOZZLE_X = (285, 325)            # stream peak must lie under the nozzle
REF_BEFORE = 15                  # reference frame: this many frames before the labelled start
PAD_AFTER = 15
LUMA_MIN = 8.0                   # 0-255 luma difference that counts as a stream row
STREAM_MAX_W = 12                # px at half maximum; the stream measures 5-8 px
MIN_ROWS = 30                    # of 95 rows
MAX_GAP = 15                     # frames of no stream allowed inside one burst
MIN_BURST = 15                   # frames (0.5 video s)
TANK_ML = {"T100": "t100_ml", "T200": "t200_ml", "T400": "t400_ml"}
DRAW_GAIN = 4.0                  # timeline only: a*b* shift is scaled up so pale streams show hue


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


def hue_chroma(da, db):
    """CIELAB hue angle (degrees: 0 = +a* red, 90 = +b* yellow, 180 green, 270 blue) and chroma."""
    return float(np.degrees(np.arctan2(db, da)) % 360), float(np.hypot(da, db))


def draw_bgr(da, db):
    """Timeline colour for an a*b* shift: its own hue at fixed lightness, chroma scaled up."""
    lab = np.array([[[70.0, DRAW_GAIN * da, DRAW_GAIN * db]]], np.float32)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
    return tuple(int(c) for c in np.clip(bgr * 255, 0, 255))


def tank_refs(samples):
    """Reference a*b* shift per tank: median over [(tank, da, db)] attributed to that tank."""
    by_tank = {}
    for tank, da, db in samples:
        if tank:
            by_tank.setdefault(tank, []).append((da, db))
    return {t: np.median(v, axis=0) for t, v in by_tank.items()}


def nearest_tank(da, db, refs):
    """Tank whose reference shift is closest in the a*b* plane."""
    return min(refs, key=lambda t: np.hypot(da - refs[t][0], db - refs[t][1]))


def vial_rgb(frame):
    """Median sRGB (0-255) over the vial's liquid region, all pixels (not only the most saturated)."""
    roi = frame[VIAL_Y[0]:VIAL_Y[1], VIAL_X[0] - X0:VIAL_X[1] - X0].reshape(-1, 3)
    return np.median(roi, axis=0)[::-1]


def vial_changed(frame, ref):
    """Fraction of the vial region's rows whose mean colour moved from the empty vial. After a
    small first dose the liquid is still below the region, so its colour there is empty glass."""
    rows = (slice(VIAL_Y[0], VIAL_Y[1]), slice(VIAL_X[0] - X0, VIAL_X[1] - X0))
    diff = np.abs(frame[rows].astype(np.float32) - ref[rows].astype(np.float32)).mean(axis=(1, 2))
    return float(np.mean(diff > VIAL_CHANGE))


def liquid_level(frame, ref):
    """Top row of the liquid column and where it is: 'below_holder' (clear view), 'above_holder'
    (the holder top and vial labels can hide it) or 'none'. Works on brightness change as well as
    colour, so clear liquid counts: per row, the mean RGB change from the empty vial over LEVEL_X.
    The threshold scales with the column's own change below the holder, so pale and dark mixes are
    judged alike. Scans up from the base, skips the holder rows, stops after LEVEL_STOP dry rows."""
    cols = slice(LEVEL_X[0] - X0, LEVEL_X[1] - X0)
    change = np.abs(frame[:, cols].astype(np.float32) - ref[:, cols].astype(np.float32)).mean(axis=(1, 2))
    limit = max(LEVEL_FLOOR, LEVEL_FRAC * float(np.percentile(change[HOLDER_ROWS[1]:LEVEL_BOTTOM + 1], 90)))
    top, dry = None, 0
    for y in range(LEVEL_BOTTOM, ROW0, -1):
        if HOLDER_ROWS[0] <= y < HOLDER_ROWS[1]:
            continue
        if change[y] >= limit:
            top, dry = y, 0
        else:
            dry += 1
            if dry >= LEVEL_STOP:
                break
    if top is None:
        return None, "none"
    return top, "below_holder" if top >= HOLDER_ROWS[1] else "above_holder"


def rgb_to_lab(rgb):
    bgr = np.asarray(rgb, np.float32)[::-1].reshape(1, 1, 3) / 255
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def hue_gap(h1, h2):
    """Signed hue difference h1 - h2 in degrees, wrapped to [-180, 180)."""
    return (h1 - h2 + 180) % 360 - 180


def measure(frame, ref, ref_lab):
    luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = stream_rows(luma, cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    if len(rows) < MIN_ROWS:
        return {"rows": len(rows), "width": np.nan, "dl": np.nan, "da": np.nan, "db": np.nan,
                "rgb": (np.nan,) * 3}
    lab = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    ys = np.array([r[0] for r in rows])
    xs = np.clip(np.array([r[1] for r in rows])[:, None] + np.array([-1, 0, 1]), 0, W - 1)
    shift = (lab[ys[:, None], xs] - ref_lab[ys[:, None], xs]).reshape(-1, 3)
    bgr = np.median(frame[ys[:, None], xs].reshape(-1, 3), axis=0)
    return {"rows": len(rows), "width": float(np.median([r[2] for r in rows])),
            "dl": float(np.median(shift[:, 0])), "da": float(np.median(shift[:, 1])),
            "db": float(np.median(shift[:, 2])), "rgb": tuple(float(c) for c in bgr[::-1])}


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
    """Tank whose liquid each burst shows if bursts show the current dose, or the previous one."""
    return list(order), [prev_last] + list(order[:-1])


def agreement(seen, expected):
    """Matching positions, counting only positions where both are known and counts agree."""
    if len(seen) != len(expected):
        return None
    return sum(s == e for s, e in zip(seen, expected) if e is not None)


def infer_order(frames, ml_by_tank, rate):
    """Tank order whose expected burst lengths (rate x mL) best fit the measured ones (log error)."""
    def err(perm):
        return sum(np.log(f / (rate[t] * ml_by_tank[t])) ** 2 for f, t in zip(frames, perm))
    return list(min(itertools.permutations(ml_by_tank), key=err))


def vial_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="vial_fill_video_5.mp4 (default: under the data root)")
    ap.add_argument("--vials", type=int, nargs="*", help="vial numbers (default: all 30)")
    ap.add_argument("--store", action="store_true",
                    help="write stream and vial colours to data/ (all 30 vials only)")
    a = ap.parse_args()
    if a.store and a.vials:
        raise SystemExit("--store needs all 30 vials; drop --vials")
    mp4 = Path(a.video) if a.video else data_root() / VIAL_FILL / "videos/vial_fill_video_5.mp4"
    if not mp4.is_file():
        raise SystemExit(f"missing {mp4}; see docs/data.md")
    out = outputs_root() / "inn52"
    out.mkdir(parents=True, exist_ok=True)
    every = vial_rows(REPO / "data/video5_vial_recipes.csv")
    assumed_colours = {v["vial"]: int(i > 0 and v["recipe_id"] == every[i - 1]["recipe_id"])
                       for i, v in enumerate(every)}
    prev_last = {v["vial"]: (every[i - 1]["dispense_order"].split(">")[-1] if i else None)
                 for i, v in enumerate(every)}
    vials = [v for v in every if not a.vials or int(v["vial"]) in a.vials]

    frame_out, burst_out, vial_out, strips, seconds, n_frames = [], [], [], [], 0.0, 0
    stored, finals = [], []
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
        empty_rgb = vial_rgb(ref)
        empty_lab = rgb_to_lab(empty_rgb)
        burst_of = np.zeros(len(frames), int)
        for k, (b0, b1) in enumerate(bursts(flowing)):
            burst_of[b0:b1 + 1] = k + 1
        for i, (m, f) in enumerate(zip(meas, frames)):
            vdl, vda, vdb = rgb_to_lab(vial_rgb(f)) - empty_lab
            level, zone = liquid_level(f, ref)
            frame_out.append({"vial": v["vial"], "frame": ref_frame + i, "in_window": int(start <= ref_frame + i <= end),
                              "burst": burst_of[i], "flowing": int(flowing[i]),
                              "stream_rows": m["rows"], "width_px": f"{m['width']:.1f}",
                              "dL": f"{m['dl']:.2f}", "dA": f"{m['da']:.2f}", "dB": f"{m['db']:.2f}",
                              "vial_dL": f"{vdl:.2f}", "vial_dA": f"{vda:.2f}", "vial_dB": f"{vdb:.2f}",
                              "vial_region_changed": f"{vial_changed(f, ref):.2f}",
                              "level_row": level, "level_zone": zone})

        order = v["dispense_order"].split(">")
        final_frame = frames[min(len(frames) - 1, end - ref_frame + FINAL_AFTER)]
        final_rgb = vial_rgb(final_frame)
        shifts, durations = [], []
        for k, (b0, b1) in enumerate(bursts(flowing)):
            idx = [i for i in range(b0, b1 + 1) if flowing[i]]
            dl, da, db = (float(np.median([meas[i][c] for i in idx])) for c in ("dl", "da", "db"))
            rgb = np.median([meas[i]["rgb"] for i in idx], axis=0)
            hue, chroma = hue_chroma(da, db)
            vframe = frames[min(len(frames) - 1, b1 + VIAL_AFTER)]
            vrgb = vial_rgb(vframe)
            vdl, vda, vdb = rgb_to_lab(vrgb) - empty_lab
            vhue, vchroma = hue_chroma(vda, vdb)
            stored.append({"vial": v["vial"], "recipe_id": v["recipe_id"],
                           "assumed_correct_colour": assumed_colours[v["vial"]], "burst": k + 1,
                           "tank": "", "first_frame": ref_frame + b0, "last_frame": ref_frame + b1,
                           "vial_frame": ref_frame + min(len(frames) - 1, b1 + VIAL_AFTER),
                           **{f"stream_{c}": f"{x:.0f}" for c, x in zip("rgb", rgb)},
                           "stream_dL": f"{dl:.2f}", "stream_dA": f"{da:.2f}", "stream_dB": f"{db:.2f}",
                           "stream_hue_deg": f"{hue:.0f}", "stream_chroma": f"{chroma:.2f}",
                           **{f"vial_{c}": f"{x:.0f}" for c, x in zip("rgb", vrgb)},
                           **{f"empty_{c}": f"{x:.0f}" for c, x in zip("rgb", empty_rgb)},
                           "vial_dL": f"{vdl:.2f}", "vial_dA": f"{vda:.2f}", "vial_dB": f"{vdb:.2f}",
                           "vial_hue_deg": f"{vhue:.0f}", "vial_chroma": f"{vchroma:.2f}",
                           "vial_region_changed": f"{vial_changed(vframe, ref):.2f}",
                           **dict(zip(("level_row", "level_zone"), liquid_level(vframe, ref))),
                           "cumulative_ml": "",
                           "stream_minus_vial_ab": f"{np.hypot(da - vda, db - vdb):.2f}",
                           "stream_minus_vial_hue_deg": f"{hue_gap(hue, vhue):.0f}"})
            shifts.append((da, db))
            durations.append(b1 - b0 + 1)
            burst_out.append({"vial": v["vial"], "burst": k + 1, "first_frame": ref_frame + b0,
                              "last_frame": ref_frame + b1, "frames": b1 - b0 + 1, "flowing_frames": len(idx),
                              "dL": f"{dl:.2f}", "dA": f"{da:.2f}", "dB": f"{db:.2f}",
                              "hue_deg": f"{hue:.0f}", "chroma": f"{chroma:.2f}",
                              "stream_rgb": "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb),
                              "width_px": f"{np.median([meas[i]['width'] for i in idx]):.1f}",
                              "tank": "", "recipe_ml": "", "frames_per_ml": "",
                              "nearest_current": "", "nearest_previous": ""})
        current, previous = hypotheses(order, prev_last[v["vial"]])
        ml = [float(v[TANK_ML[t]]) for t in order]
        if len(durations) == len(order):
            for row, kept, tank, mls, total in zip(burst_out[-len(durations):], stored[-len(durations):], order, ml,
                                                   np.cumsum(ml)):
                row.update(tank=tank, recipe_ml=f"{mls:g}", frames_per_ml=f"{row['frames'] / mls:.2f}")
                kept.update(tank=tank, cumulative_ml=f"{total:g}")
        fdl, fda, fdb = rgb_to_lab(final_rgb) - empty_lab
        finals.append({"vial": v["vial"], "recipe_id": v["recipe_id"],
                       "assumed_correct_colour": assumed_colours[v["vial"]],
                       "frame": ref_frame + min(len(frames) - 1, end - ref_frame + FINAL_AFTER),
                       "empty_frame": ref_frame,
                       **{f"vial_{c}": f"{x:.0f}" for c, x in zip("rgb", final_rgb)},
                       **{f"empty_{c}": f"{x:.0f}" for c, x in zip("rgb", empty_rgb)},
                       "vial_dL": f"{fdl:.2f}", "vial_dA": f"{fda:.2f}", "vial_dB": f"{fdb:.2f}",
                       "vial_region_changed": f"{vial_changed(final_frame, ref):.2f}",
                       **dict(zip(("level_row", "level_zone"), liquid_level(final_frame, ref))),
                       "total_ml": v["total_ml"],
                       "no_prepour_reference": int(start - REF_BEFORE < 0)})
        hc = [hue_chroma(*s) for s in shifts]
        vial_out.append({"vial": v["vial"], "recipe_id": v["recipe_id"], "dispense_order": v["dispense_order"],
                         "tanks": len(order), "bursts": len(shifts), "count_match": int(len(shifts) == len(order)),
                         "burst_frames": ">".join(map(str, durations)), "recipe_ml": ">".join(f"{x:g}" for x in ml),
                         "hues_deg": ">".join(f"{h:.0f}" for h, _ in hc) or "-",
                         "chromas": ">".join(f"{c:.1f}" for _, c in hc) or "-",
                         "current_hypothesis": ">".join(current), "previous_hypothesis": ">".join(c or "?" for c in previous),
                         "nearest_current": "", "nearest_previous": "",
                         "current_matches": "", "previous_matches": "",
                         "previous_known": sum(c is not None for c in previous),
                         "no_prepour_reference": int(start - REF_BEFORE < 0)})
        strips.append((v["vial"], [draw_bgr(m["da"], m["db"]) if f else (40, 40, 40) for m, f in zip(meas, flowing)],
                       start - ref_frame, end - ref_frame))
        print(f"vial {v['vial']:>2} {v['dispense_order']:<15} bursts {len(shifts)}/{len(order)} "
              f"frames {'>'.join(map(str, durations)):<14} mL {'>'.join(f'{x:g}' for x in ml):<9} "
              f"hue {vial_out[-1]['hues_deg']:<12} chroma {vial_out[-1]['chromas']}")

    # Colour vs each hypothesis, with no assumed tank colours: per hypothesis, fit each tank's
    # reference shift on the other recipes' vials, then match each burst to its nearest reference.
    usable = [r for r in vial_out if not r["no_prepour_reference"]]
    counted = [r for r in usable if r["count_match"]]
    ok = {r["vial"] for r in counted}
    recipe = {r["vial"]: r["recipe_id"] for r in vial_out}
    by_vial = {}
    for b in burst_out:
        by_vial.setdefault(b["vial"], []).append(b)
    colour_totals = {}
    for key in ("current", "previous"):
        expect = {r["vial"]: [t if t != "?" else None for t in r[f"{key}_hypothesis"].split(">")]
                  for r in counted}
        attributed = [(r["vial"], t, float(b["dA"]), float(b["dB"]))
                      for r in counted for b, t in zip(by_vial[r["vial"]], expect[r["vial"]])]
        dist = []
        for r in counted:
            refs = tank_refs([(t, da, db) for vial, t, da, db in attributed if recipe[vial] != r["recipe_id"]])
            seen = [nearest_tank(float(b["dA"]), float(b["dB"]), refs) for b in by_vial[r["vial"]]]
            for b, s in zip(by_vial[r["vial"]], seen):
                b[f"nearest_{key}"] = s
            r[f"nearest_{key}"] = ">".join(seen)
            r[f"{key}_matches"] = agreement(seen, expect[r["vial"]])
            dist += [float(np.hypot(float(b["dA"]) - refs[e][0], float(b["dB"]) - refs[e][1]))
                     for b, e in zip(by_vial[r["vial"]], expect[r["vial"]]) if e in refs]
        colour_totals[key] = (sum(r[f"{key}_matches"] for r in counted), float(np.median(dist)))

    targets = [(out / name, rows) for name, rows in
               (("stream_frames.csv", frame_out), ("stream_bursts.csv", burst_out), ("stream_vials.csv", vial_out),
                ("stream_vial_colours.csv", stored), ("vial_final_colours.csv", finals))]
    if a.store:
        targets += [(REPO / "data/video5_stream_vial_colours.csv", stored),
                    (REPO / "data/video5_vial_final_colours.csv", finals)]
    for path, rows in targets:
        with path.open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    longest = max(len(s[1]) for s in strips)
    img = np.zeros((len(strips) * 14, longest // 2 + 60, 3), np.uint8)
    for r, (vial, colours, s, e) in enumerate(strips):
        y = r * 14
        cv2.putText(img, vial, (2, y + 11), 0, 0.35, (255, 255, 255), 1)
        for i, colour in enumerate(colours[::2]):
            cv2.line(img, (60 + i, y + 2), (60 + i, y + 11), colour, 1)
        for f in (s, e):
            cv2.line(img, (60 + f // 2, y), (60 + f // 2, y + 13), (0, 0, 255), 1)
    cv2.imwrite(str(out / "stream_timelines.png"), img)

    for tank in TANK_ML:
        fpm = [float(b["frames_per_ml"]) for b in burst_out if b["tank"] == tank and b["vial"] in ok]
        print(f"{tank}: {len(fpm)} bursts, {np.median(fpm):.2f} video frames per mL "
              f"(min {min(fpm):.2f}, max {max(fpm):.2f}, CV {np.std(fpm) / np.mean(fpm):.1%})")
    # Dispense order changes between recipes, so try to recover it from timing alone: per-tank
    # rates come from the other recipes' vials, and every order of the vial's tanks is scored.
    right, misses = 0, []
    for r in counted:
        rate = {t: np.median([float(b["frames_per_ml"]) for b in burst_out
                              if b["tank"] == t and b["vial"] in ok and recipe[b["vial"]] != r["recipe_id"]])
                for t in TANK_ML}
        order = r["dispense_order"].split(">")
        frames = [int(x) for x in r["burst_frames"].split(">")]
        ml_by_tank = dict(zip(order, (float(x) for x in r["recipe_ml"].split(">"))))
        got = infer_order(frames, ml_by_tank, rate)
        right += got == order
        if got != order:
            misses.append(f"vial {r['vial']} -> {'>'.join(got)}")
    print(f"order from timing alone (leave-one-recipe-out rates): {right} of {len(counted)} vials; "
          f"misses: {', '.join(misses) or 'none'}")
    (cur, cur_d), (prv, prv_d) = colour_totals["current"], colour_totals["previous"]
    print(f"{len(usable)} vials with a pre-pour reference: burst count = tank count in {len(counted)}; "
          f"nearest fitted tank colour over those {len(counted)} vials (leave-one-recipe-out): "
          f"current {cur}/{sum(r['tanks'] for r in counted)} (median a*b* distance {cur_d:.2f}), "
          f"previous {prv}/{sum(r['previous_known'] for r in counted)} (median {prv_d:.2f}); "
          f"{1000 * seconds / n_frames:.2f} ms per frame (measurement only, excluding decode); wrote {out}")


if __name__ == "__main__":
    main()
