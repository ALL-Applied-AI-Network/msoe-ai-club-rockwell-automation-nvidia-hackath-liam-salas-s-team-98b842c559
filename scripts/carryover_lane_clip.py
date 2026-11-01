#!/usr/bin/env python
"""Lane-pass clip: follow one vial's camera ID from its birth at the lane bottom to its reference-line crossing.

    python scripts/carryover_lane_clip.py VIAL [VIAL ...] [--dataset DIR] [--records-dir DIR] [--out DIR]

Reads RECORDS_DIR/{records.json, links.json, topdown/detections.jsonl, topdown/summary.json} and the cap_seal video
under DATASET (default: the csi-vial-fill-dataset folder below fluidmix.paths.data_root(), so FLUIDMIX_DATA is
honoured), and writes OUT/lanepass_vialNN_<decision>.mp4 (default OUT = RECORDS_DIR/demo): the inspection lane at
3x with boxes, the vial's ID highlighted, its decision card, and a flash when the ID crosses the reference line
(y=100, the scorer's definition), plus a JPEG at the crossing frame. A vial whose linked ID never crossed the line
gets no clip (the run script's line_crossings decide that).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix.paths import data_root, require

DATASET_LAYOUT = Path("csi-vial-fill-dataset")
DEFAULT_RECORDS_DIR = ROOT / "outputs/carryover-video5"
X0, X1, S, REF = 100, 240, 3, 100


def default_dataset(explicit=None):
    """The organizers' csi-vial-fill-dataset folder: --dataset itself when given, else that folder below
    fluidmix.paths.data_root() (FLUIDMIX_DATA or .env, else <repo>/external)."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return data_root() / DATASET_LAYOUT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vials", type=int, nargs="+")
    ap.add_argument("--dataset", type=Path, default=None, help="csi-vial-fill-dataset folder (default: fluidmix.paths data root)")
    ap.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR, help="folder with records.json, links.json and topdown/")
    ap.add_argument("--out", type=Path, default=None, help="where the clips go (default: RECORDS_DIR/demo)")
    a = ap.parse_args()
    dataset = default_dataset(a.dataset)
    video = require(dataset / "cap_seal/videos/cap_seal_video_5.mp4", "cap_seal video 5")
    rd = a.records_dir
    out = a.out or rd / "demo"
    out.mkdir(parents=True, exist_ok=True)
    rec = {r['vial']: r for r in json.load(open(rd / 'records.json'))['vials']}
    L = json.load(open(rd / 'links.json')); links = {l['vial']: l for l in (L if isinstance(L, list) else L.get('links', []))}
    summary = json.load(open(rd / 'topdown/summary.json')); cross = {}
    for c in summary.get('line_crossings', []): cross.setdefault(c['track_id'], c['frame'])
    det = {}
    for l in open(rd / 'topdown/detections.jsonl'):
        d = json.loads(l); det[d['frame']] = d
    for vial in a.vials:
        r = rec[vial]; link = links.get(vial) or {}; ids = link.get('fragment_ids') or ([link['camera_id']] if link.get('camera_id') else [])
        if not ids: print(f'vial {vial}: no camera id'); continue
        xf = min((cross[i] for i in ids if i in cross), default=None)
        if xf is None: print(f'vial {vial}: id {ids} never crossed the line'); continue
        start, end = max(0, link['birth_frame'] - 30), xf + 60
        cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        name = f"lanepass_vial{vial:02d}_{r['decision']}"; wr = None; col = (60, 60, 230) if r['decision'] == 'discard' else (60, 200, 90)
        for f in range(start, end + 1):
            ok, img = cap.read()
            if not ok: break
            strip = cv2.resize(img[:, X0:X1], ((X1 - X0) * S, 480 * S), interpolation=cv2.INTER_CUBIC)
            cv2.line(strip, (0, REF * S), ((X1 - X0) * S, REF * S), (0, 200, 255), 2); cv2.putText(strip, 'reference line y=100', (4, REF * S - 8), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 1)
            d = det.get(f, {})
            for b in d.get('boxes_xyxy', []):
                if 140 <= (b[0] + b[2]) / 2 <= 200: cv2.rectangle(strip, (int((b[0] - X0) * S), int(b[1] * S)), (int((b[2] - X0) * S), int(b[3] * S)), (0, 220, 120), 1)
            for t in d.get('tracks', []):
                x, y, w, h = t['bbox_xywh']; cx, cy = int((x + w / 2 - X0) * S), int((y + h / 2) * S); mine = t['id'] in ids
                cv2.putText(strip, t['id'], (cx + 14, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, .6 if mine else .45, col if mine else (200, 200, 200), 2 if mine else 1)
                if mine: cv2.circle(strip, (cx, cy), 26, col, 2)
            if xf <= f <= xf + 45: cv2.rectangle(strip, (0, 0), (strip.shape[1] - 1, strip.shape[0] - 1), (0, 200, 255), 8); cv2.putText(strip, f'crossed the line: vial {vial} {r["decision"].upper()}', (8, 50), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 200, 255), 2)
            card = np.zeros((480 * S, 420, 3), np.uint8) + 24
            lines = [f'cap_seal_video_5  frame {f}', '', f'vial {vial}   recipe {r["recipe_id"]}', f'previous recipe {r.get("prev_recipe_id")}', f'position in run {r.get("position_in_run")}', '', f'camera ID {ids[-1]}', f'born at frame {link["birth_frame"]}', f'line crossing at frame {xf}', '', f'colour error vs settled {r.get("dE_vs_reference")}', f'threshold {r.get("tau")}', '']
            for i, s in enumerate(lines): cv2.putText(card, s, (16, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, .6, (235, 235, 235), 1)
            cv2.rectangle(card, (16, 40 + 34 * len(lines)), (300, 40 + 34 * len(lines) + 60), col, -1); cv2.putText(card, r['decision'].upper(), (30, 40 + 34 * len(lines) + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            frame = np.concatenate([strip, card], axis=1)
            if wr is None: wr = cv2.VideoWriter(str(out / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
            wr.write(frame)
            if f == xf: cv2.imwrite(str(out / f'{name}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if wr is not None: wr.release()
        cap.release(); print(f'vial {vial}: {name}.mp4 frames {start}-{end} ({(end-start+1)/30:.1f} s), crossing at {xf}')


if __name__ == '__main__':
    main()
