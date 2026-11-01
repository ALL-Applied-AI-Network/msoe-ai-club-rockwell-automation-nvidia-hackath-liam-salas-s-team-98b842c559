#!/usr/bin/env python
"""Run the synthetic-only top-down cap segmenter on a cap_seal video and write an overlay clip plus per-frame JSON.

    python scripts/topdown/run_topdown_segmenter.py external/.../cap_seal/videos/cap_seal_video_2.mp4 \
        --weights models/topdown/dense53-width12-ema-none.pt --out outputs/topdown-demo/video2

Pipeline per frame: DenseCapNet probability map -> 8-connected components of >= 9 px -> confidence = mean
probability -> drop components under the synthetic-derived floor -> ConveyorGuardedTracker (camera IDs,
lane entry, reference-line exits). Tracker settings, lane and floor come from the frozen evaluation protocol
(models/topdown/real-evaluation-protocol.json) so this matches the scored configuration exactly.
Needs torch, opencv-python, numpy. CPU works (about 15 fps on a laptop); CUDA is used when available.
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fluidmix.topdown.segmenter import DenseCapNet, image_tensor, instances
from fluidmix.topdown.conveyor_guarded_tracking import ConveyorGuardedTracker


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_weights = next((str(c) for c in (ROOT / 'models/topdown/dense53-width12-ema-none.pt', Path('/home/hackathon5/outputs/topdown-01a17f15-20261029/dense53/train01/best.pt')) if c.exists()), str(ROOT / 'models/topdown/dense53-width12-ema-none.pt'))
    ap.add_argument('video'); ap.add_argument('--weights', default=default_weights, help='dense53 checkpoint; weights are kept out of git, see models/topdown/README.md')
    ap.add_argument('--protocol', default=str(ROOT / 'models/topdown/real-evaluation-protocol.json'))
    ap.add_argument('--floor', type=float, default=None, help='override the confidence floor (default: the protocol value, 0.90 for dense53; use 0.95 for dense60)')
    ap.add_argument('--out', default=str(ROOT / 'outputs/topdown-demo')); ap.add_argument('--start', type=int, default=0); ap.add_argument('--frames', type=int, default=0, help='0 = whole video')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--reference-line', type=int, default=100, help='y of the reference line; a track whose centre moves from below to above it emits a line_crossing event (the scorer\'s definition)')
    a = ap.parse_args()
    p = json.load(open(a.protocol)); lane = p['inference']['lane_xyxy']; floor = a.floor if a.floor is not None else p['dense_component_confidence_floor']['value']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = DenseCapNet(); m.load_state_dict(torch.load(a.weights, map_location='cpu', weights_only=True)); m.to(dev).eval()
    tr = ConveyorGuardedTracker(**p['tracker'], **p['tracker_options'])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_FRAMES, a.start); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wr = None if a.no_video else cv2.VideoWriter(str(out / 'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))
    log = open(out / 'detections.jsonl', 'w'); exits = []; crossings = []; last_cy = {}; f = a.start; t0 = time.time(); n = 0
    while True:
        ok, img = cap.read()
        if not ok or (a.frames and n >= a.frames): break
        if img.shape != (480, 640, 3): raise SystemExit('expected 640x480 cap_seal frames')
        with torch.no_grad(): prob = m(image_tensor([img], dev)).sigmoid()[0, 0].float().cpu().numpy()
        boxes, scores, _ = instances(prob)
        keep = [(b, s) for b, s in zip(boxes, scores) if s >= floor]
        xywh, conf = [], []
        for b, s in keep:
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            if lane[0] <= (x0 + x1) / 2 <= lane[2]:
                xywh.append([max(0, x0), max(0, y0), max(1, min(640, x1) - max(0, x0)), max(1, min(480, y1) - max(0, y0))]); conf.append(float(s))
        visible, emitted = tr.update(f, xywh, confidences=conf)
        for e in emitted: exits.append(dict(e, frame=f))
        for t in visible:
            x, y, w, h = t.bbox; cy = y + h / 2; prev = last_cy.get(t.track_id)
            if prev is not None and prev[0] == f - 1 and prev[1] > a.reference_line >= cy: crossings.append(dict(track_id=t.track_id, frame=f, y_from=prev[1], y_to=cy))
            last_cy[t.track_id] = (f, cy)
        log.write(json.dumps(dict(frame=f, boxes_xyxy=[[float(v) for v in b] for b, _ in keep], confidence=[float(s) for _, s in keep],
                                  tracks=[dict(id=t.track_id, bbox_xywh=list(t.bbox), state=t.state, exit_frame=t.exit_frame) for t in visible])) + '\n')
        if wr is not None:
            o = img.copy(); cv2.rectangle(o, (int(lane[0]), int(lane[1])), (int(lane[2]), int(lane[3]) - 1), (255, 220, 0), 1)
            cv2.line(o, (int(lane[0]), p['tracker']['exit_y']), (int(lane[2]), p['tracker']['exit_y']), (255, 220, 0), 1)
            for b, s in keep:
                x0, y0, x1, y1 = [int(round(v)) for v in b]; cv2.rectangle(o, (x0, y0), (x1, y1), (0, 220, 120), 1)
            for t in visible:
                x, y, w, h = t.bbox; cv2.putText(o, t.track_id, (int(x + w) + 3, int(y + h / 2) + 4), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 220, 120), 1)
            cv2.line(o, (int(lane[0]), a.reference_line), (int(lane[2]), a.reference_line), (0, 200, 255), 1)
            recent = [e for e in crossings if f - 15 <= e['frame'] <= f]
            if recent: cv2.putText(o, 'LINE ' + str(recent[-1]['track_id']), (int(lane[2]) + 8, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 255), 2)
            cv2.putText(o, f'frame {f}  boxes {len(keep)}  tracks {len(visible)}  line crossings {len(crossings)}', (8, 470), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            wr.write(o)
        f += 1; n += 1
    cap.release(); log.close()
    if wr is not None: wr.release()
    json.dump(dict(video=a.video, weights=a.weights, floor=floor, frames=n, reference_line_y=a.reference_line, line_crossings=crossings, tracker_exit_events=exits, elapsed_seconds=time.time() - t0), open(out / 'summary.json', 'w'), indent=1)
    print(f'{n} frames in {time.time() - t0:.1f}s ({n / max(1e-9, time.time() - t0):.1f} fps on {dev}); reference-line crossings {len(crossings)}; tracker exit events {len(exits)}; wrote {out}')


if __name__ == '__main__':
    main()
