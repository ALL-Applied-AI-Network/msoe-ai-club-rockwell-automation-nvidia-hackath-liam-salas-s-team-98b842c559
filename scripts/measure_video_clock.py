"""Measure each video's burned-in wall clock and map every still onto video time.

    python scripts/measure_video_clock.py                  # all cameras and videos, 5 s step
    python scripts/measure_video_clock.py --step 10 --videos 5 --cameras vial_fill

Every frame carries a wall-clock overlay (bottom left; a white copy with an offset black copy).
We OCR the white copy every --step video seconds and model wall = a_k + b * video_s: one
rate b per video, and a new offset a_k wherever the clock jumps (frames lost while recording).
Breaks are pinned to the frame by bisection. Writes to FLUIDMIX_OUTPUTS:
    clock_readings.csv  every OCR sample, its segment and residual (the evidence)
    clock_fits.csv      one row per camera, video and segment: frames, wall span, rate, gap, residuals
    still_video_map.csv every indexed still -> segment, video_s and frame, or why it has none
Stills are placed by their filename time, which matched the burned-in still clock in spot
checks; --check-stills OCRs a sample of stills to test that per camera.
Needs ffmpeg, ffprobe and tesseract on PATH.
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import outputs_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventory_data import CAMERAS, find_dataset

FPS = 30
CROP = (0, 436, 300, 478)  # x0, y0, x1, y1 of the clock overlay in 640x480 frames
FULL = re.compile(r"(\d{2})[:.\-]\s?(\d{2})[:.]\s?(\d{2})[.\s]\s?(\d{3})")
SECS = re.compile(r"(\d{2})[.\s]\s?(\d{3})\s*\D*$")
INLIER_S = 0.25  # an OCR reading farther than this from its segment is treated as a misread
JUMP_S = 0.5     # a sustained offset step larger than this is a recording gap
MIN_SEG = 3      # readings needed to accept a segment


def frames_at(mp4, every):
    """Return [(frame_index, crop)] for about every `every`-th frame, decoded in one pass.

    The frame index comes from each output frame's own timestamp (showinfo), not its position:
    select can skip frames, and counting positions then mislabels every later sample.
    """
    x0, y0, x1, y1 = CROP
    w, h = x1 - x0, y1 - y0
    cmd = ["ffmpeg", "-v", "info", "-nostats", "-i", str(mp4), "-vf",
           f"select=not(mod(n\\,{every})),showinfo,crop={w}:{h}:{x0}:{y0}", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    run = subprocess.run(cmd, capture_output=True, check=True)
    times = [float(t) for t in re.findall(r"Parsed_showinfo.*?pts_time:\s*([\d.]+)", run.stderr.decode())]
    n = len(run.stdout) // (w * h * 3)
    if n != len(times):
        raise SystemExit(f"{mp4}: {n} frames but {len(times)} timestamps from showinfo")
    crops = np.frombuffer(run.stdout, np.uint8)[: n * w * h * 3].reshape(n, h, w, 3)
    return [(round(t * FPS), crops[k]) for k, t in enumerate(times)]


def ocr(crop, scratch, key):
    """OCR the white copy of the overlay (the offset black copy is too broken to read). Returns the raw string."""
    # Near-white and unsaturated; looser thresholds swallow light-grey backgrounds.
    mask = (crop.min(axis=2) > 220) & (np.ptp(crop, axis=2) < 30)
    im = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8))
    im = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)
    im = ImageOps.expand(im, border=30, fill=255)  # tesseract drops text that touches the edge
    path = Path(scratch) / f"{key}.png"
    im.save(path)
    # One thread per tesseract: the caller already runs several in parallel.
    out = subprocess.run(["tesseract", str(path), "-", "--psm", "7", "-c",
                          "tessedit_char_whitelist=0123456789-:. "], capture_output=True, text=True,
                         env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    if out.returncode:
        raise SystemExit(f"tesseract failed on {path}: {out.stderr.strip()}")
    return out.stdout.strip()


def parse(text, day):
    """Return (full wall datetime or None, seconds-within-minute or None)."""
    # OCR often turns ':' into '.' or drops spaces, but the overlay always has 17 digits
    # (YYYYMMDDHHMMSSmmm), so slice by position when exactly that many are read.
    digits = re.sub(r"\D", "", text)
    if len(digits) == 17:
        hh, mm, ss, ms = int(digits[8:10]), int(digits[10:12]), int(digits[12:14]), int(digits[14:17])
        if hh < 24 and mm < 60 and ss < 60:
            return day + timedelta(hours=hh, minutes=mm, seconds=ss + ms / 1000), ss + ms / 1000
    full = None
    body = text[10:] if len(text) > 10 else text  # drop the fixed date
    m = FULL.search(body)
    if m:
        hh, mm, ss, frac = m.groups()
        if int(hh) < 24 and int(mm) < 60 and int(ss) < 60:
            full = day + timedelta(hours=int(hh), minutes=int(mm), seconds=int(ss) + int(frac) / 10 ** len(frac))
    m = SECS.search(text)
    secs = int(m.group(1)) + int(m.group(2)) / 10 ** len(m.group(2)) if m and int(m.group(1)) < 60 else None
    return full, secs


def read_frame(mp4, frame, scratch, key):
    """OCR the clock of one frame (seeked), for pinning down a break."""
    x0, y0, x1, y1 = CROP
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{frame / FPS:.4f}", "-i", str(mp4), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(480, 640, 3)
    return ocr(img[y0:y1, x0:x1], scratch, key)


def snap(secs, pred):
    """Put a seconds-of-minute reading in the minute that is nearest the predicted wall second."""
    base = pred - (pred % 60)
    cands = np.array([base - 60, base, base + 60]) + secs
    return float(cands[np.argmin(np.abs(cands - pred))])


def segments(t, off):
    """Split readings into runs of constant clock offset; returns a segment id per reading (-1 = misread).

    Isolated OCR misreads are removed by a 5-reading running median; a sustained step in that
    median larger than JUMP_S starts a new segment.
    """
    n = len(off)
    med = np.array([np.median(off[max(0, i - 2):i + 3]) for i in range(n)])
    starts = [0]
    for i in range(1, n):
        if abs(med[i] - np.median(med[starts[-1]:i])) > JUMP_S:
            starts.append(i)
    # Keep the consistent readings of each run and drop runs that are only a burst of misreads,
    # then merge neighbours whose levels turn out to be the same.
    runs = []
    for s0, s1 in zip(starts, starts[1:] + [n]):
        idx = np.arange(s0, s1)
        keep = idx[np.abs(off[idx] - np.median(off[idx])) < INLIER_S]
        if len(keep) >= MIN_SEG:
            runs.append(list(keep))
    merged = runs[:1]
    for run in runs[1:]:
        if abs(np.median(off[run]) - np.median(off[merged[-1]])) < JUMP_S:
            merged[-1] += run
        else:
            merged.append(run)
    ids = np.full(n, -1)
    for k, run in enumerate(merged):
        idx = np.array(run)
        ids[idx[np.abs(off[idx] - np.median(off[idx])) < INLIER_S]] = k
    # Renumber so segments are 0..k-1 in time order.
    for new, old in enumerate(sorted(set(ids) - {-1})):
        ids[ids == old] = new
    return ids


def measure(ds, cam, vid, step, scratch):
    mp4 = ds / cam / "videos" / f"{cam}_video_{vid}.mp4"
    samples = frames_at(mp4, step * FPS)
    with ThreadPoolExecutor(8) as pool:
        texts = list(pool.map(lambda s: ocr(s[1], scratch, f"{cam}_{vid}_{s[0]}"), samples))
    # OCR misreads the date too; take it from this video's still filenames (any camera).
    day = None
    for c in (cam, *CAMERAS):
        stamps = [r["timestamp"] for r in csv.DictReader((ds / c / "image_index.csv").open()) if r["video_id"] == vid]
        if stamps:
            day = datetime.fromisoformat(stamps[0]).replace(hour=0, minute=0, second=0, microsecond=0)
            break
    if day is None:
        raise SystemExit(f"{mp4}: no stills for video {vid} to date it")
    rows = [{"camera": cam, "video_id": vid, "frame": frame, "video_s": frame / FPS, "ocr": text}
            for (frame, _), text in zip(samples, texts)]
    for r in rows:
        r["full"], r["secs"] = parse(r["ocr"], day)
    # Seconds-of-minute alone is ambiguous by 60 s, so a global fit can alias to a slope off by
    # 60/duration. Take the slope from nearby readings (up to 3 steps apart; --step is capped so
    # that span stays under 60 wall s).
    read = [r for r in rows if r["secs"] is not None]
    slopes = [((q["secs"] - p["secs"]) % 60) / (q["video_s"] - p["video_s"])
              for i, p in enumerate(read) for q in read[i + 1:i + 4] if q["video_s"] - p["video_s"] <= 3 * step]
    if len(slopes) < 3:
        raise SystemExit(f"{mp4}: too few consecutive clock readings")
    b = float(np.median(slopes))
    # Anchor each reading's minute with the nearest fully parsed readings, so a jump in the clock
    # does not pull far-away readings into the wrong minute.
    full = [(r["video_s"], (r["full"] - day).total_seconds() - b * r["video_s"]) for r in rows if r["full"] is not None]
    if not full:
        raise SystemExit(f"{mp4}: no fully parsed clock reading to anchor the minute")
    ft = np.array([f[0] for f in full])
    fo = np.array([f[1] for f in full])
    for r in rows:
        r["wall_s"] = None
        if r["secs"] is not None:
            near = np.argsort(np.abs(ft - r["video_s"]))[:9]
            r["wall_s"] = snap(r["secs"], float(np.median(fo[near])) + b * r["video_s"])
    ok = [r for r in rows if r["wall_s"] is not None]
    tt = np.array([r["video_s"] for r in ok])
    ww = np.array([r["wall_s"] for r in ok])
    ids = segments(tt, ww - b * tt)
    nseg = int(ids.max()) + 1
    if nseg == 0:
        raise SystemExit(f"{mp4}: no consistent run of clock readings")

    # Pin each break to the frame by bisection between the last reading of one segment and the
    # first of the next, classifying each probed frame by which segment's offset it matches.
    offs = [float(np.median((ww - b * tt)[ids == k])) for k in range(nseg)]
    breaks = []
    for k in range(nseg - 1):
        lo = int(round(tt[ids == k].max() * FPS))
        hi = int(round(tt[ids == k + 1].min() * FPS))
        misses = 0
        while hi - lo > 1 and misses < 16:
            # On a failed read, probe a little to either side instead of the same frame again.
            jitter = (misses + 1) // 2 * 7 * (1 if misses % 2 else -1)
            mid = min(max((lo + hi) // 2 + jitter, lo + 1), hi - 1)
            _, secs = parse(read_frame(mp4, mid, scratch, f"{cam}_{vid}_probe_{mid}"), day)
            if secs is None:
                misses += 1
                continue
            o = snap(secs, offs[k] + b * mid / FPS) - b * mid / FPS
            if abs(o - offs[k]) < INLIER_S:
                lo, misses = mid, 0
            elif abs(o - offs[k + 1]) < INLIER_S:
                hi, misses = mid, 0
            else:
                misses += 1
        breaks.append((lo, hi))

    # Common slope, one intercept per segment, least squares on inliers.
    m = ids >= 0
    X = np.column_stack([tt[m]] + [(ids[m] == k).astype(float) for k in range(nseg)])
    coef, *_ = np.linalg.lstsq(X, ww[m], rcond=None)
    b, icpt = float(coef[0]), coef[1:]
    res = ww - (b * tt + np.where(ids >= 0, icpt[np.maximum(ids, 0)], np.nan))
    for r, e, k in zip(ok, res, ids):
        r["segment"] = int(k) if k >= 0 else ""
        r["residual_s"] = round(float(e), 3) if k >= 0 else ""
        r["inlier"] = int(k >= 0)
    for r in rows:
        r.setdefault("segment", "")
        r.setdefault("residual_s", "")
        r.setdefault("inlier", 0)
        r["wall_time"] = (day + timedelta(seconds=r["wall_s"])).isoformat(sep=" ") if r["wall_s"] is not None else ""
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "csv=p=0", str(mp4)], capture_output=True, text=True).stdout)
    last_frame = int(round(duration * FPS)) - 1
    fits = []
    for k in range(nseg):
        f0 = 0 if k == 0 else breaks[k - 1][1]
        f1 = last_frame if k == nseg - 1 else breaks[k][0]
        sel = ids == k
        r_k = res[sel]
        own = float(np.polyfit(tt[sel], ww[sel], 1)[0]) if sel.sum() >= 10 else float("nan")
        fits.append({
            "camera": cam, "video_id": vid, "segment": k, "frame_start": f0, "frame_end": f1,
            "video_s_start": round(f0 / FPS, 3), "video_s_end": round(f1 / FPS, 3),
            "wall_start": (day + timedelta(seconds=icpt[k] + b * f0 / FPS)).isoformat(sep=" "),
            "wall_end": (day + timedelta(seconds=icpt[k] + b * f1 / FPS)).isoformat(sep=" "),
            "wall_at_video_0": (day + timedelta(seconds=icpt[k])).isoformat(sep=" "),
            "wall_s_per_video_s": round(b, 6), "capture_fps": round(FPS / b, 3),
            "segment_own_slope": round(own, 6),
            "gap_before_s": "" if k == 0 else round(float(icpt[k] - icpt[k - 1]), 3),
            "break_uncertainty_frames": "" if k == 0 else breaks[k - 1][1] - breaks[k - 1][0] - 1,
            "samples": len(rows), "readings": int(sel.sum()),
            "rms_residual_s": round(float(np.sqrt(np.mean(r_k ** 2))), 4),
            "max_abs_residual_s": round(float(np.max(np.abs(r_k))), 4),
            "video_duration_s": round(duration, 3),
        })
    return fits, rows


def map_stills(ds, fits):
    """Place each still (by filename wall time) in its video's segment; flag stills in gaps or outside."""
    by = {}
    for f in fits:
        by.setdefault((f["camera"], f["video_id"]), []).append(f)
    out = []
    for cam in CAMERAS:
        for r in csv.DictReader((ds / cam / "image_index.csv").open()):
            segs = by.get((cam, r["video_id"]))
            row = {"camera": cam, "filename": r["filename"], "timestamp": r["timestamp"], "video_id": r["video_id"],
                   "segment": "", "video_s": "", "frame": "", "status": "no_video_fit"}
            if segs:
                w = datetime.fromisoformat(r["timestamp"])
                half = 0.5 / FPS
                for f in segs:
                    vs = (w - datetime.fromisoformat(f["wall_at_video_0"])).total_seconds() / f["wall_s_per_video_s"]
                    if f["video_s_start"] - half <= vs <= f["video_s_end"] + half:
                        row.update(segment=f["segment"], video_s=round(vs, 3), frame=round(vs * FPS), status="in_video")
                        break
                else:
                    if w < datetime.fromisoformat(segs[0]["wall_start"]):
                        row["status"] = "before_video"
                    elif w > datetime.fromisoformat(segs[-1]["wall_end"]):
                        row["status"] = "after_video"
                    else:
                        # Between two segments: a real gap only if that break was pinned to the frame.
                        nxt = next(f for f in segs if w < datetime.fromisoformat(f["wall_start"]))
                        row["status"] = ("in_recording_gap" if nxt["break_uncertainty_frames"] == 0
                                         else "near_unresolved_break")
            out.append(row)
    return out


def check_stills(ds, cam, n, scratch):
    """OCR n evenly spaced stills and compare the burned-in clock with the filename time."""
    idx = list(csv.DictReader((ds / cam / "image_index.csv").open()))
    picks = idx[:: max(1, len(idx) // n)][:n]
    diffs = []
    for r in picks:
        img = np.asarray(Image.open(ds / cam / "images" / r["filename"]).convert("RGB"))
        x0, y0, x1, y1 = CROP
        text = ocr(img[y0:y1, x0:x1], scratch, "still_" + r["filename"])
        ts = datetime.fromisoformat(r["timestamp"])
        _, secs = parse(text, ts.replace(hour=0, minute=0, second=0, microsecond=0))
        if secs is not None:
            d = (secs - (ts.second + ts.microsecond / 1e6) + 30) % 60 - 30
            diffs.append(d)
    return diffs


def write(path, rows, fields):
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", help="csi-vial-fill-dataset folder (default: under the data root)")
    ap.add_argument("--step", type=int, default=5, help="video seconds between clock readings")
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    ap.add_argument("--videos", nargs="+", default=["1", "2", "3", "4", "5"])
    ap.add_argument("--check-stills", type=int, default=0, metavar="N", help="OCR N stills per camera")
    a = ap.parse_args()
    # Slopes use readings up to 3 steps apart, read as seconds-of-minute; past 60 wall s they alias.
    if not 1 <= a.step <= 12:
        ap.error("--step must be 1-12 video seconds (3 steps at ~1.22 wall s each must stay under 60 s)")
    ds = find_dataset(a.dataset)
    out = outputs_root()
    fits, readings = [], []
    with tempfile.TemporaryDirectory(prefix=".clock-", dir=out) as scratch:
        for cam in a.cameras:
            for vid in a.videos:
                segs, rows = measure(ds, cam, vid, a.step, scratch)
                fits += segs
                readings += rows
                for f in segs:
                    gap = f"gap {f['gap_before_s']:+.3f}s (±{f['break_uncertainty_frames']} fr) " if f["segment"] else ""
                    print(f"{cam:<10} v{vid} seg{f['segment']}: frames {f['frame_start']}-{f['frame_end']} {gap}"
                          f"b={f['wall_s_per_video_s']:.5f} own={f['segment_own_slope']:.5f} "
                          f"n={f['readings']}/{f['samples']} rms {f['rms_residual_s']:.3f}s "
                          f"max {f['max_abs_residual_s']:.3f}s wall {f['wall_start']} -> {f['wall_end']}")
        if a.check_stills:
            for cam in a.cameras:
                d = check_stills(ds, cam, a.check_stills, scratch)
                close = sum(abs(x) <= 0.01 for x in d)
                # Whole-second or 0.1 s differences are OCR digit misreads, not clock disagreement.
                print(f"stills {cam}: {len(d)} read, {close} within 10 ms of the filename time; "
                      f"others {sorted(round(x, 3) for x in d if abs(x) > 0.01)}")
    write(out / "clock_readings.csv", readings,
          ["camera", "video_id", "frame", "video_s", "ocr", "wall_time", "segment", "residual_s", "inlier"])
    write(out / "clock_fits.csv", fits, list(fits[0]))
    stills = map_stills(ds, fits)
    write(out / "still_video_map.csv", stills, list(stills[0]))
    print(f"wrote clock_readings.csv, clock_fits.csv, still_video_map.csv to {out}")
    counts = {}
    for st in stills:
        counts.setdefault((st["camera"], st["video_id"]), {}).setdefault(st["status"], 0)
        counts[(st["camera"], st["video_id"])][st["status"]] += 1
    for key in sorted(counts):
        print(f"stills {key[0]:<10} v{key[1]}: {counts[key]}")


if __name__ == "__main__":
    main()
