"""INN-52: how well does SAM 2.1 segment the vial liquid and the dispense stream? Inspection only.

Runs on a CUDA machine (the team's DGX Spark / GB10). Setup, remote use from a laptop and how
to read the results are in docs/sam.md.

    python scripts/sam_eval.py --check                       # GPU, packages, weights and data only
    python scripts/sam_eval.py                               # video-5 vial + stream evaluation
    python scripts/sam_eval.py --model small --vials 8 20    # another model, a subset
    python scripts/sam_eval.py --clip PATH --point 305 140   # track one prompt through any clip
    python scripts/sam_eval.py --dry-run                     # no SAM: flood-fill stand-in, CPU only

SAM is prompted, not trained: it gets a frame plus a click (and for the vial a box) and returns
the object under the click. Nothing here trains on real video. Three checks on video 5:

  vials       final frame of each fill, click on the liquid. Is the mask the liquid column? Its
              colour is compared with the fixed-region colour, and its height with the recipe's
              total mL (a mask that tracks the liquid level should correlate).
  stream      one frame in the middle of each burst found by scripts/stream_sequence.py, and the
              pre-pour frame, click under the nozzle. A stream mask must be a thin vertical stripe
              (2-STREAM_MAX_W px wide over at least MIN_ROWS of the stream rows), the same test
              the deterministic method uses. Counts: stripes on burst frames and on pre-pour
              frames (false alarms).
  propagate   SAM 2 video mode on the first --propagate bursts: click once on the first flowing
              frame and let SAM follow the stream; per frame, is it still a stripe?

The deterministic detector is the comparison, not ground truth: there are no hand-drawn masks.
Every run writes outputs/sam/<tag>/ with env.json (host, GPU, versions, weight hashes), CSVs,
overlay images and summary.txt.
"""
import argparse
import csv
import getpass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fluidmix.paths import REPO, VIAL_FILL, data_root, outputs_root
from stream_sequence import (FPS, MIN_ROWS, NOZZLE_X, REF_BEFORE, ROW0, ROW1, STREAM_MAX_W, bursts, measure,
                             read_frames, vial_rows)

URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
MODELS = {                       # name: (checkpoint, config inside the sam2 package)
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
MAX_PROPAGATE = 300              # frames (10 s) followed per burst in video mode, to bound GPU memory
VIAL_POINT = (312, 345)          # liquid below the holder (scripts/stream_sequence.py vial region)
VIAL_BOX = (280, 70, 345, 405)   # whole vial, rim to base, x0 y0 x1 y1
FINAL_AFTER = 10                 # final frame: this many frames after the labelled fill end
STREAM_Y = (ROW0 + ROW1) // 2    # click height for the stream
FLOOD_TOL = 12                   # dry-run stand-in: flood-fill colour tolerance (0-255)


def weights_dir():
    return data_root() / "models/sam2.1"


def read_full(mp4, first, count, fps=FPS):
    """Full BGR frames (count, 480, 640, 3) starting at frame `first` of a constant-rate video."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{first / fps:.4f}", "-i", str(mp4),
                          "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 480, 640, 3)


def read_clip(mp4, start_s, seconds):
    """All frames of a clip section, at its own frame rate."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start_s:.3f}", "-t", f"{seconds:.3f}", "-i",
                          str(mp4), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(mp4)],
                           capture_output=True, check=True, text=True).stdout.split(",")
    w, h = int(probe[0]), int(probe[1])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


# ------------------------------------------------------------------ mask measurements
def mask_stats(mask):
    """Pixel count, top/bottom row, height, median row width and median column of a mask."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return {"px": 0, "top": -1, "bottom": -1, "height": 0, "width": 0.0, "x": -1.0}
    widths = mask.sum(axis=1)
    return {"px": int(mask.sum()), "top": int(ys.min()), "bottom": int(ys.max()),
            "height": int(ys.max() - ys.min() + 1), "width": float(np.median(widths[widths > 0])),
            "x": float(np.median(xs))}


def is_stripe(mask, rows=(ROW0, ROW1), max_w=STREAM_MAX_W, min_rows=MIN_ROWS):
    """Stream test, as in stream_sequence: enough stream rows hold a 2..max_w px run under the nozzle."""
    band = mask[rows[0]:rows[1]]
    widths = band.sum(axis=1)
    centred = np.array([NOZZLE_X[0] <= np.median(np.nonzero(r)[0]) < NOZZLE_X[1] if r.any() else False
                        for r in band])
    return int(((widths >= 2) & (widths <= max_w) & centred).sum()) >= min_rows


def median_rgb(frame, mask):
    if not mask.any():
        return np.full(3, np.nan)
    return np.median(frame[mask], axis=0)[::-1]


def lab(rgb):
    return cv2.cvtColor(np.asarray(rgb, np.float32)[::-1].reshape(1, 1, 3) / 255, cv2.COLOR_BGR2LAB)[0, 0]


def overlay(frame, mask, points, label, colour=(0, 0, 255)):
    out = frame.copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array(colour)).astype(np.uint8)
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
    cv2.putText(out, label, (5, 20), 0, 0.6, (0, 255, 255), 2)
    return out


def contact_sheet(images, cols=6, width=320):
    if not images:
        return None
    small = [cv2.resize(i, (width, int(i.shape[0] * width / i.shape[1]))) for i in images]
    blank = np.zeros_like(small[0])
    rows = [np.hstack((small[i:i + cols] + [blank] * cols)[:cols]) for i in range(0, len(small), cols)]
    return np.vstack(rows)


# ------------------------------------------------------------------ segmenters
class FloodFill:
    """Dry-run stand-in with SAM's interface: flood fill from the click. Not SAM; tests the pipeline."""
    name = "dry-run flood fill"

    def image(self, frame, points, box=None):
        mask = np.zeros((frame.shape[0] + 2, frame.shape[1] + 2), np.uint8)
        x, y = points[0]
        cv2.floodFill(frame.copy(), mask, (int(x), int(y)), 0, (FLOOD_TOL,) * 3, (FLOOD_TOL,) * 3,
                      4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
        out = mask[1:-1, 1:-1].astype(bool)
        if box is not None:
            x0, y0, x1, y1 = box
            keep = np.zeros_like(out)
            keep[y0:y1, x0:x1] = True
            out &= keep
        return out

    def video(self, frames, points):
        return [self.image(f, points) for f in frames]


class Sam2:
    """SAM 2.1 image and video predictors on CUDA (or CPU, slowly)."""

    def __init__(self, model, device):
        import torch
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt, cfg = MODELS[model]
        path = weights_dir() / ckpt
        if not path.is_file():
            raise SystemExit(f"missing {path}; run scripts/setup_sam_spark.sh (docs/sam.md)")
        self.torch, self.device, self.name = torch, device, f"SAM 2.1 {model}"
        self.image_predictor = SAM2ImagePredictor(build_sam2(cfg, str(path), device=device))
        self.video_predictor = build_sam2_video_predictor(cfg, str(path), device=device)

    def _ctx(self):
        if self.device == "cuda":
            return self.torch.autocast("cuda", dtype=self.torch.bfloat16)
        return self.torch.autocast("cpu", enabled=False)

    def image(self, frame, points, box=None):
        with self.torch.inference_mode(), self._ctx():
            self.image_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks, scores, _ = self.image_predictor.predict(
                point_coords=np.array(points, np.float32), point_labels=np.ones(len(points), np.int32),
                box=None if box is None else np.array(box, np.float32), multimask_output=True)
        return masks[int(np.argmax(scores))].astype(bool)

    def video(self, frames, points):
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(frames):              # the video predictor reads a JPEG folder
                cv2.imwrite(str(Path(tmp) / f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with self.torch.inference_mode(), self._ctx():
                state = self.video_predictor.init_state(video_path=tmp)
                self.video_predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=1,
                    points=np.array(points, np.float32), labels=np.ones(len(points), np.int32))
                out = [None] * len(frames)
                for idx, _, logits in self.video_predictor.propagate_in_video(state):
                    out[idx] = (logits[0] > 0).cpu().numpy().reshape(frames[0].shape[:2])
        return out


# ------------------------------------------------------------------ environment
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def environment(model):
    env = {"host": platform.node(), "user": getpass.getuser(), "machine": platform.machine(),
           "python": platform.python_version(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": model}
    try:
        env["repo_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            env.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
    except ImportError:
        env["torch"] = None
    try:
        from importlib.metadata import version
        env["sam2"] = next((version(n) for n in ("SAM-2", "sam2") if _has(n)), None)
    except Exception:
        env["sam2"] = None
    ckpt = weights_dir() / MODELS[model][0]
    env["weights"] = str(ckpt)
    env["weights_sha256"] = sha256(ckpt) if ckpt.is_file() else None
    return env


def _has(dist):
    from importlib.metadata import PackageNotFoundError, version
    try:
        version(dist)
        return True
    except PackageNotFoundError:
        return False


def check(model, mp4):
    env = environment(model)
    for k, v in env.items():
        print(f"{k:16s} {v}")
    problems = [msg for ok, msg in (
        (env.get("torch"), "PyTorch is not installed"),
        (env.get("cuda_available"), "no CUDA GPU visible to PyTorch"),
        (env.get("sam2"), "the sam2 package is not installed"),
        (env.get("weights_sha256"), f"weights missing: {env['weights']}"),
        (mp4.is_file(), f"video missing: {mp4} (set FLUIDMIX_DATA, see docs/data.md)")) if not ok]
    if not problems:
        try:
            Sam2(model, "cuda")
            print("SAM 2.1 model loads on cuda")
        except Exception as exc:             # report any load failure plainly
            problems.append(f"model failed to load: {exc}")
    for p in problems:
        print("PROBLEM:", p)
    return not problems


# ------------------------------------------------------------------ evaluations
def deterministic(frames_crop):
    """stream_sequence's per-frame result on crops whose first frame is the pre-pour reference."""
    ref = frames_crop[0]
    ref_lab = cv2.cvtColor(ref.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return [measure(f, ref, ref_lab) for f in frames_crop]


def eval_vials(seg, mp4, vials, sheets):
    rows, images, seconds = [], [], []
    for v in vials:
        f = int(v["end_frame"]) + FINAL_AFTER
        frame = read_full(mp4, f, 1)[0]
        t0 = time.perf_counter()
        mask = seg.image(frame, [VIAL_POINT], VIAL_BOX)
        seconds.append(time.perf_counter() - t0)
        s = mask_stats(mask)
        fixed = np.median(frame[320:370, 292:332].reshape(-1, 3), axis=0)[::-1]
        rgb = median_rgb(frame, mask)
        x0, y0, x1, y1 = VIAL_BOX
        rows.append({"vial": v["vial"], "recipe_id": v["recipe_id"], "total_ml": v["total_ml"], "frame": f,
                     **{f"mask_{k}": s[k] for k in ("px", "top", "bottom", "height", "width")},
                     "inside_vial_box": f"{mask[y0:y1, x0:x1].sum() / max(1, mask.sum()):.2f}",
                     "mask_rgb": "-" if np.isnan(rgb).any() else "#%02x%02x%02x" % tuple(int(c) for c in rgb),
                     "fixed_region_rgb": "#%02x%02x%02x" % tuple(int(c) for c in fixed),
                     "deltaE_mask_vs_fixed": "" if np.isnan(rgb).any() else f"{np.linalg.norm(lab(rgb) - lab(fixed)):.1f}",
                     "seconds": f"{seconds[-1]:.3f}"})
        images.append(overlay(frame, mask, [VIAL_POINT], f"vial {v['vial']} {v['total_ml']} mL"))
    sheets["vials"] = images
    ml = np.array([float(r["total_ml"]) for r in rows])
    height = np.array([r["mask_height"] for r in rows], float)
    r_fill = float(np.corrcoef(ml, height)[0, 1]) if len(rows) > 2 and height.std() > 0 and ml.std() > 0 else None
    inside = np.median([float(r["inside_vial_box"]) for r in rows])
    de = [float(r["deltaE_mask_vs_fixed"]) for r in rows if r["deltaE_mask_vs_fixed"]]
    return rows, {"vials": len(rows), "fill_height_vs_total_ml_r": None if r_fill is None else round(r_fill, 3),
                  "median_inside_vial_box": round(float(inside), 2),
                  "median_deltaE_mask_vs_fixed_region": round(float(np.median(de)), 1) if de else None,
                  "seconds_per_image": round(float(np.median(seconds)), 3)}


def eval_stream(seg, mp4, vials, n_propagate, sheets):
    frame_rows, prop_rows, images, seconds, prop_seconds = [], [], [], [], []
    propagated = 0
    for v in vials:
        start, end = int(v["start_frame"]), int(v["end_frame"])
        if start - REF_BEFORE < 0:
            continue                                  # no pre-pour reference (vial 1)
        first = start - REF_BEFORE
        meas = deterministic(read_frames(mp4, first, end + 15 - first + 1))
        flowing = np.array([not np.isnan(m["da"]) for m in meas])
        found = bursts(flowing)
        ref = read_full(mp4, first, 1)[0]
        picks = [(0, "pre-pour", ref, NOZZLE_X[0] + (NOZZLE_X[1] - NOZZLE_X[0]) // 2)]
        for k, (b0, b1) in enumerate(found):
            mid = (b0 + b1) // 2
            mid = min(range(b0, b1 + 1), key=lambda i: (not flowing[i], abs(i - mid)))
            frame = read_full(mp4, first + mid, 1)[0]
            picks.append((mid, f"burst {k + 1}", frame, _stream_x(frame, ref)))
        for idx, kind, frame, x in picks:
            t0 = time.perf_counter()
            mask = seg.image(frame, [(x, STREAM_Y)])
            seconds.append(time.perf_counter() - t0)
            s = mask_stats(mask)
            frame_rows.append({"vial": v["vial"], "frame": first + idx, "kind": kind,
                               "deterministic_flowing": int(flowing[idx]), "sam_stripe": int(is_stripe(mask)),
                               "click_x": x, **{f"mask_{k}": s[k] for k in ("px", "top", "bottom", "height", "width")},
                               "seconds": f"{seconds[-1]:.3f}"})
            if kind != "pre-pour" or len(images) < 6:
                images.append(overlay(frame, mask, [(x, STREAM_Y)], f"v{v['vial']} {kind}"))
        for k, (b0, b1) in enumerate(found):
            if propagated >= n_propagate:
                break
            start_i = next(i for i in range(b0, b1 + 1) if flowing[i])
            stop = min(b1 + 1, start_i + MAX_PROPAGATE)
            clip = read_full(mp4, first + start_i, stop - start_i)
            x = _stream_x(clip[0], ref)
            t0 = time.perf_counter()
            masks = seg.video(list(clip), [(x, STREAM_Y)])
            prop_seconds.append((time.perf_counter() - t0) / max(1, len(masks)))
            stripes = [is_stripe(m) for m in masks]
            det = flowing[start_i:stop]
            prop_rows.append({"vial": v["vial"], "burst": k + 1, "frames": len(masks),
                              "deterministic_flowing": int(det.sum()), "sam_stripe": int(sum(stripes)),
                              "both": int(sum(s and d for s, d in zip(stripes, det))),
                              "seconds_per_frame": f"{prop_seconds[-1]:.3f}"})
            propagated += 1
    sheets["stream"] = images
    burst = [r for r in frame_rows if r["kind"] != "pre-pour"]
    pre = [r for r in frame_rows if r["kind"] == "pre-pour"]
    summary = {"burst_frames": len(burst), "sam_stripe_on_burst_frames": sum(r["sam_stripe"] for r in burst),
               "prepour_frames": len(pre), "sam_stripe_on_prepour_frames": sum(r["sam_stripe"] for r in pre),
               "seconds_per_image": round(float(np.median(seconds)), 3) if seconds else None}
    if prop_rows:
        summary.update(propagated_bursts=len(prop_rows),
                       propagated_frames=sum(r["frames"] for r in prop_rows),
                       propagated_stripe_frames=sum(r["sam_stripe"] for r in prop_rows),
                       propagated_deterministic_flowing=sum(r["deterministic_flowing"] for r in prop_rows),
                       propagated_both=sum(r["both"] for r in prop_rows),
                       seconds_per_video_frame=round(float(np.median(prop_seconds)), 4))
    return frame_rows, prop_rows, summary


def _stream_x(frame, ref):
    """Column of the strongest luma difference in the stream rows, under the nozzle."""
    diff = np.abs(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                  - cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32))[ROW0:ROW1, NOZZLE_X[0]:NOZZLE_X[1]]
    return NOZZLE_X[0] + int(np.argmax(diff.mean(axis=0)))


def eval_clip(seg, clip, point, start_s, seconds_len, sheets):
    frames = read_clip(clip, start_s, seconds_len)
    t0 = time.perf_counter()
    masks = seg.video(list(frames), [tuple(point)])
    per_frame = (time.perf_counter() - t0) / max(1, len(masks))
    rows = []
    for i, (f, m) in enumerate(zip(frames, masks)):
        s = mask_stats(m)
        rgb = median_rgb(f, m)
        rows.append({"frame": i, **{f"mask_{k}": s[k] for k in ("px", "top", "bottom", "height", "width", "x")},
                     "mask_rgb": "-" if np.isnan(rgb).any() else "#%02x%02x%02x" % tuple(int(c) for c in rgb)})
    step = max(1, len(frames) // 12)
    sheets["clip"] = [overlay(frames[i], masks[i], [point] if i == 0 else [], f"frame {i}")
                      for i in range(0, len(frames), step)]
    return rows, {"frames": len(frames), "frames_with_mask": sum(r["mask_px"] > 0 for r in rows),
                  "median_width_px": float(np.median([r["mask_width"] for r in rows if r["mask_px"]] or [0])),
                  "seconds_per_video_frame": round(per_frame, 4)}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=MODELS, default="tiny")
    ap.add_argument("--video", help="vial_fill_video_5.mp4 (default: under the data root)")
    ap.add_argument("--vials", type=int, nargs="*", help="vial numbers (default: all 30)")
    ap.add_argument("--propagate", type=int, default=5, help="bursts to follow in SAM video mode")
    ap.add_argument("--clip", help="any video: track --point from its first frame instead of video 5")
    ap.add_argument("--point", type=int, nargs=2, default=(305, 140), metavar=("X", "Y"))
    ap.add_argument("--start", type=float, default=0.0, help="--clip start, seconds")
    ap.add_argument("--seconds", type=float, default=10.0, help="--clip length, seconds")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"), help="output folder name")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--check", action="store_true", help="check GPU, packages, weights and data, then exit")
    ap.add_argument("--dry-run", action="store_true", help="flood-fill stand-in instead of SAM (no GPU)")
    a = ap.parse_args()
    mp4 = Path(a.video) if a.video else data_root() / VIAL_FILL / "videos/vial_fill_video_5.mp4"
    if a.check:
        raise SystemExit(0 if check(a.model, mp4) else 1)

    seg = FloodFill() if a.dry_run else Sam2(a.model, a.device)
    out = outputs_root() / "sam" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    env = environment(a.model)
    env["segmenter"] = seg.name
    env["argv"] = sys.argv[1:]
    sheets, summary = {}, {"segmenter": seg.name}
    if a.clip:
        rows, summary["clip"] = eval_clip(seg, Path(a.clip), a.point, a.start, a.seconds, sheets)
        write_csv(out / "clip_frames.csv", rows)
    else:
        if not mp4.is_file():
            raise SystemExit(f"missing {mp4}; see docs/data.md")
        vials = [v for v in vial_rows(REPO / "data/video5_vial_recipes.csv") if not a.vials or int(v["vial"]) in a.vials]
        rows, summary["vials"] = eval_vials(seg, mp4, vials, sheets)
        write_csv(out / "vials.csv", rows)
        frame_rows, prop_rows, summary["stream"] = eval_stream(seg, mp4, vials, a.propagate, sheets)
        write_csv(out / "stream_frames.csv", frame_rows)
        write_csv(out / "stream_propagation.csv", prop_rows)
    for name, images in sheets.items():
        sheet = contact_sheet(images)
        if sheet is not None:
            cv2.imwrite(str(out / f"{name}_overlays.png"), sheet)
    (out / "env.json").write_text(json.dumps(env, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    text = json.dumps(summary, indent=2)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
