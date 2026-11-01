"""Why are reference exits missed? For each expected exit interval of each real video, look at the window
[start-15, end+15] in a package's harness output and classify: truth present, raw detection matched (IoU>=0.5),
tracker-visible output present, tracker output matched to the truth. Usage: exit_miss_causes.py PACKAGE_DIR"""
import json, sys, collections
from pathlib import Path
N = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029'); root = Path(sys.argv[1])
traj_all = json.load(open(N / 'dense-real53/trajectories.json'))
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix*iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i) if i else 0
cy = lambda b: (b[1] + b[3]) / 2
summary = collections.Counter(); rows = []
for v in range(1, 6):
    traj = next(x for x in traj_all['videos'] if f'cap_seal_video_{v}' in x['video'])
    det = {}; trk = collections.defaultdict(list)
    for l in open(root / f'cap_seal_video_{v}/detections.jsonl'):
        d = json.loads(l); det[d['frame']] = d['boxes_xyxy']
    for l in open(root / f'cap_seal_video_{v}/tracks.jsonl'):
        d = json.loads(l); x, y, w, h = d['track']['bbox']; trk[d['frame']].append((d['track']['track_id'], [x, y, x + w, y + h]))
    for t in traj['tracks']:
        pts = {p['frame']: p['bbox_xyxy'] for p in t['points']}
        for e in t['events']:
            a, b = e['frame_interval']; win = range(a - 15, b + 16)
            n_truth = n_det = n_trk = n_trkm = 0; ids = set(); cross_det = cross_trk = False; prev_det = prev_trk = None
            for f in win:
                tb = pts.get(f)
                if tb is None: prev_det = prev_trk = None; continue
                n_truth += 1
                dm = any(iou(d, tb) >= .5 for d in det.get(f, [])); n_det += dm
                tm = [i for i, bb in trk.get(f, []) if iou(bb, tb) >= .5]; n_trk += bool(trk.get(f)); n_trkm += bool(tm); ids.update(tm)
                # consecutive crossing of y=100 by matched boxes
                db = next((d for d in det.get(f, []) if iou(d, tb) >= .5), None)
                if prev_det is not None and db is not None and cy(prev_det) > 100 >= cy(db): cross_det = True
                prev_det = db
                tbx = next((bb for i, bb in trk.get(f, []) if iou(bb, tb) >= .5), None)
                if prev_trk is not None and tbx is not None and cy(prev_trk) > 100 >= cy(tbx): cross_trk = True
                prev_trk = tbx
            cause = ('found' if cross_trk else 'detected crossing, tracker output missing/withheld' if cross_det else
                     'detections too sparse at the line' if n_det < n_truth * .5 else 'detections present, no consecutive matched pair across y=100')
            summary[cause] += 1
            rows.append((v, t['physical_id'], a, b, n_truth, n_det, n_trkm, len(ids), cause))
print('| video | id | crossing interval | truth frames in window | raw det matched | tracker matched | tracker ids | cause |')
print('|---|---|---|---|---|---|---|---|')
for r in rows:
    if r[-1] != 'found': print('| ' + ' | '.join(str(x) for x in r) + ' |')
print(); print(dict(summary))
