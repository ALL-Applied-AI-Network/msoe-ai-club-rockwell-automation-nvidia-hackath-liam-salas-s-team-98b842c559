"""Quick real-video detection proxy (GPU): stride-sampled frames, trajectory-ROI precision/recall at IoU 0.5.

Usage: real_proxy.py RUN [stride] [videos...]   e.g. real_proxy.py dense45 10 1 3 5
Not the frozen harness: no tracking, draft trajectory truth only, gap frames excluded. For choosing which
candidates deserve the full 40-minute evaluation. Never used for training or checkpoint selection.
"""
import json, sys, time, collections
import numpy as np, cv2, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
N = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029')
run = sys.argv[1]; stride = int(sys.argv[2]) if len(sys.argv) > 2 else 10; vids = [int(v) for v in sys.argv[3:]] or [1, 3, 5]
sys.path.insert(0, str(ROOT / 'src'))
from fluidmix.topdown.segmenter import DenseCapNet, image_tensor, instances
fm = json.load(open(N / run / 'train01/frozen-model.json'))
m = DenseCapNet(); m.load_state_dict(torch.load(fm['weights'], map_location='cpu', weights_only=True)); m.cuda().eval()
V = '/home/hackathon5/liamreporoot/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/external/external/csi-vial-fill-dataset/cap_seal/videos/'
traj_all = json.load(open(N / 'dense-real41/trajectories.json'))
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i) if i else 0
inroi = lambda b: 140 <= (b[0]+b[2])/2 <= 200 and 32 <= (b[1]+b[3])/2 <= 455
tot = collections.Counter(); t0 = time.time(); ybands = collections.Counter(); missbands = collections.Counter(); hitbands = collections.Counter()
for v in vids:
    stem = f'cap_seal_video_{v}'; traj = next(x for x in traj_all['videos'] if stem in x['video'])
    truth = collections.defaultdict(list); ignored = set()
    for t in traj['tracks']:
        for p in t['points']: truth[p['frame']].append(p['bbox_xyxy'])
        for g in t['gaps']: ignored.update(range(g['start'], g['stop']))
    cap = cv2.VideoCapture(V + stem + '.mp4'); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); c = collections.Counter(); batch = []; idx = []
    def flush():
        with torch.no_grad(): probs = m(image_tensor(batch, 'cuda')).sigmoid().cpu().numpy()[:, 0]
        for f, p in zip(idx, probs):
            boxes = [b for b in instances(p)[0] if inroi(b)]; ts = [b for b in truth.get(f, []) if inroi(b)]
            used = set()
            for b in boxes:
                j = max(range(len(ts)), key=lambda j: iou(b, ts[j]), default=None)
                if j is not None and iou(b, ts[j]) >= .5 and j not in used: used.add(j); c['TP'] += 1
                else: c['FP'] += 1; ybands[int((b[1]+b[3])/2 // 48)] += 1
            c['FN'] += len(ts) - len(used)
            for j, b in enumerate(ts): (hitbands if j in used else missbands)[int((b[1]+b[3])/2 // 96)] += 1
    f = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if f % stride == 0 and f not in ignored:
            ok, img = cap.retrieve()
            if ok and img.shape == (480, 640, 3): batch.append(img); idx.append(f)
            if len(batch) == 16: flush(); batch, idx = [], []
        f += 1
    if batch: flush()
    cap.release(); tot.update(c)
    p = c['TP'] / (c['TP'] + c['FP']) if c['TP'] + c['FP'] else 0; r = c['TP'] / (c['TP'] + c['FN']) if c['TP'] + c['FN'] else 0
    print(f'{run} {stem}: TP/FP/FN {c["TP"]}/{c["FP"]}/{c["FN"]} precision {p:.3f} recall {r:.3f}', flush=True)
p = tot['TP'] / (tot['TP'] + tot['FP']) if tot['TP'] + tot['FP'] else 0; r = tot['TP'] / (tot['TP'] + tot['FN']) if tot['TP'] + tot['FN'] else 0
print(f'{run} ALL (stride {stride}, videos {vids}): TP/FP/FN {tot["TP"]}/{tot["FP"]}/{tot["FN"]} precision {p:.3f} recall {r:.3f} | FP y-bands {dict(sorted(ybands.items()))} | recall by y-band(96px) { {k: round(hitbands[k]/(hitbands[k]+missbands[k]),2) for k in sorted(set(hitbands)|set(missbands))} } | {time.time()-t0:.0f}s')
