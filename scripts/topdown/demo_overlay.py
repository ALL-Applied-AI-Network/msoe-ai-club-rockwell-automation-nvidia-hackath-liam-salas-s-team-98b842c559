"""Demo clip: raw lane | YOLO baseline (trained on real) | ours (synthetic-only dense53), from the saved harness outputs.
Boxes, tracker IDs, the reference line, exit flashes and running counters. CPU only. Usage: demo_overlay.py VIDEO START END OUT.mp4"""
import json, sys, collections, cv2, numpy as np
from pathlib import Path
N = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029')
V = '/home/hackathon5/liamreporoot/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/external/external/csi-vial-fill-dataset/cap_seal/videos/'
v, start, end, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
stem = f'cap_seal_video_{v}'
def load(root):
    det = {}; trk = collections.defaultdict(list)
    for l in open(root / stem / 'detections.jsonl'):
        d = json.loads(l)
        if start - 1 <= d['frame'] <= end: det[d['frame']] = list(zip(d['boxes_xyxy'], d['confidence']))
    for l in open(root / stem / 'tracks.jsonl'):
        d = json.loads(l)
        if start - 1 <= d['frame'] <= end: trk[d['frame']].append(d['track'])
    ex = [e for e in json.load(open(root / stem / 'summary.json'))['exits'] if start <= e.get('frame', -1) <= end]
    return det, trk, ex
ours = load(N / 'dense-real53/real-dense53'); base = load(N / 'real-test12/real-train12')
X0, X1 = 100, 240; LW = X1 - X0  # lane strip
S = 3  # scale factor for the strip
def panel(img, det, trk, exits_so_far, f, title, color):
    strip = cv2.resize(img[:, X0:X1], (LW * S, 480 * S), interpolation=cv2.INTER_CUBIC)
    cv2.line(strip, (0, 100 * S), (LW * S, 100 * S), (255, 220, 0), 2); cv2.putText(strip, 'reference line', (4, 100 * S - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 220, 0), 1)
    for b, c in det.get(f, []):
        if not (140 <= (b[0] + b[2]) / 2 <= 200): continue
        p = [int((b[0] - X0) * S), int(b[1] * S), int((b[2] - X0) * S), int(b[3] * S)]
        cv2.rectangle(strip, (p[0], p[1]), (p[2], p[3]), color, 2)
    for t in trk.get(f, []):
        x, y, w, h = t['bbox']; cx, cy = int((x + w / 2 - X0) * S), int((y + h / 2) * S)
        if not (140 <= x + w / 2 <= 200): continue
        cv2.putText(strip, t['track_id'], (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2); cv2.putText(strip, t['track_id'], (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
    recent = [e for e in exits_so_far if f - 20 <= e['frame'] <= f]
    if recent: cv2.rectangle(strip, (0, 0), (LW * S - 1, 480 * S - 1), (255, 220, 0), 6); cv2.putText(strip, 'EXIT ' + str(recent[-1].get('track_id', '')), (8, 60), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 220, 0), 2)
    hdr = np.zeros((70, LW * S, 3), np.uint8); cv2.putText(hdr, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    cv2.putText(hdr, f'boxes {sum(1 for b, c in det.get(f, []) if 140 <= (b[0]+b[2])/2 <= 200)}  tracks {len(trk.get(f, []))}  exits {len([e for e in exits_so_far if e["frame"] <= f])}', (8, 54), cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)
    return np.concatenate([hdr, strip], axis=0)
cap = cv2.VideoCapture(V + stem + '.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, start)
W = None; wr = None
for f in range(start, end + 1):
    ok, img = cap.read()
    if not ok: break
    raw = cv2.resize(img, (640 * 3 // 2, 480 * 3 // 2), interpolation=cv2.INTER_AREA); cv2.rectangle(raw, (int(140 * 1.5), int(32 * 1.5)), (int(200 * 1.5), int(455 * 1.5)), (255, 220, 0), 1)
    hdr = np.zeros((70, raw.shape[1], 3), np.uint8); cv2.putText(hdr, f'cap_seal_video_{v}  frame {f}  (real CSI line, top-down camera)', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    cv2.putText(hdr, 'inspection lane outlined', (8, 54), cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)
    rawp = np.concatenate([hdr, raw], axis=0)
    pb = panel(img, base[0], base[1], base[2], f, 'YOLO baseline (trained on real video)', (80, 160, 255))
    po = panel(img, ours[0], ours[1], ours[2], f, 'Ours: U-Net trained on synthetic renders only', (0, 220, 120))
    H = max(rawp.shape[0], pb.shape[0]); pad = lambda a: cv2.copyMakeBorder(a, 0, H - a.shape[0], 0, 12, cv2.BORDER_CONSTANT)
    frame = np.concatenate([pad(rawp), pad(pb), pad(po)], axis=1)
    if wr is None: wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
    wr.write(frame)
wr.release(); print('wrote', out, frame.shape)
