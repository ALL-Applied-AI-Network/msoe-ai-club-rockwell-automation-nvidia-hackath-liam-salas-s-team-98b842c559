"""Confidence-floor analysis: synthetic VAL true-cap confidence percentiles (the only place a floor may be chosen)
and the real proxy precision/recall that would result at several floors. Usage: conf_analysis.py RUN [stride]"""
import json, sys, time, collections
import numpy as np, cv2, torch
from pathlib import Path
N = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029'); D = N / 'edge28/dataset02'
run = sys.argv[1]; stride = int(sys.argv[2]) if len(sys.argv) > 2 else 10
sys.path.insert(0, str(N / run / 'source'))
from topdown_dense_segmenter import DenseCapNet, image_tensor, instances
fm = json.load(open(N / run / 'train01/frozen-model.json'))
m = DenseCapNet(); m.load_state_dict(torch.load(fm['weights'], map_location='cpu', weights_only=True)); m.cuda().eval()
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i) if i else 0
def predict(images):
    with torch.no_grad(): probs = m(image_tensor(images, 'cuda')).sigmoid().cpu().numpy()[:, 0]
    return [instances(p)[:2] for p in probs]
# synthetic VAL: confidences of matched components
man = json.load(open(D / 'manifest.json')); val = [r for r in man['records'] if r['role'] == 'val']
tpc, fpc = [], []
for i in range(0, len(val), 16):
    rows = val[i:i+16]; out = predict([cv2.imread(str(D / r['path'])) for r in rows])
    for r, (boxes, scores) in zip(rows, out):
        tb = [o['bbox_xyxy'] for o in r['objects']]
        for b, s in zip(boxes, scores): (tpc if any(iou(b, t) >= .5 for t in tb) else fpc).append(s)
tpc = np.array(tpc); print(f'{run} synthetic VAL: TP comps {len(tpc)} conf p0.5/1/5/50 = {np.percentile(tpc,[0.5,1,5,50]).round(4)} | FP comps {len(fpc)} conf {np.round(sorted(fpc),3)[:10]}')
floor = float(np.floor(np.percentile(tpc, 1) * 20) / 20)  # 1st percentile rounded down to 0.05
print(f'{run} synthetic-derived floor (1st pct rounded down to 0.05): {floor}')
# real proxy at floors
V = '/home/hackathon5/liamreporoot/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/external/external/csi-vial-fill-dataset/cap_seal/videos/'
traj_all = json.load(open(N / 'dense-real41/trajectories.json')); inroi = lambda b: 140 <= (b[0]+b[2])/2 <= 200 and 32 <= (b[1]+b[3])/2 <= 455
floors = sorted({0.0, 0.8, 0.9, 0.95, floor}); tot = {f: collections.Counter() for f in floors}; rtp, rfp = [], []
for v in (1, 3, 5):
    stem = f'cap_seal_video_{v}'; traj = next(x for x in traj_all['videos'] if stem in x['video'])
    truth = collections.defaultdict(list); ignored = set()
    for t in traj['tracks']:
        for p in t['points']: truth[p['frame']].append(p['bbox_xyxy'])
        for g in t['gaps']: ignored.update(range(g['start'], g['stop']))
    cap = cv2.VideoCapture(V + stem + '.mp4'); batch, idx, f = [], [], 0
    def flush():
        for fr, (boxes, scores) in zip(idx, predict(batch)):
            ts = [b for b in truth.get(fr, []) if inroi(b)]
            for fl in floors:
                keep = [(b, s) for b, s in zip(boxes, scores) if inroi(b) and s >= fl]; used = set()
                for b, s in keep:
                    j = max(range(len(ts)), key=lambda j: iou(b, ts[j]), default=None)
                    if j is not None and iou(b, ts[j]) >= .5 and j not in used: used.add(j); tot[fl]['TP'] += 1; (rtp.append(s) if fl == 0 else None)
                    else: tot[fl]['FP'] += 1; (rfp.append(s) if fl == 0 else None)
                tot[fl]['FN'] += len(ts) - len(used)
    while True:
        ok = cap.grab()
        if not ok: break
        if f % stride == 0 and f not in ignored:
            ok, img = cap.retrieve()
            if ok: batch.append(img); idx.append(f)
            if len(batch) == 16: flush(); batch, idx = [], []
        f += 1
    if batch: flush()
    cap.release()
print(f'{run} real TP conf p10/50 = {np.percentile(rtp,[10,50]).round(3)}  real FP conf p10/50/90 = {np.percentile(rfp,[10,50,90]).round(3)}')
for fl in floors:
    c = tot[fl]; p = c['TP'] / (c['TP'] + c['FP']) if c['TP'] + c['FP'] else 0; r = c['TP'] / (c['TP'] + c['FN']) if c['TP'] + c['FN'] else 0
    print(f'{run} real proxy floor {fl:.2f}: TP/FP/FN {c["TP"]}/{c["FP"]}/{c["FN"]} precision {p:.3f} recall {r:.3f} F1 {2*p*r/(p+r) if p+r else 0:.3f}')
