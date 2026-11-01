"""Re-score detection on videos 2 and 4 against REVIEWED labels built from Liam's answers.

Inputs: decisions exported from the review artifact (one JSON per document under DEC_DIR/decisions/), the draft
trajectories, and each package's official harness detections.jsonl. Rules applied to the draft labels:
  model_unmatched + yes   -> the model's boxes in that run become extra truth boxes (label was missing a cap)
  model_unmatched + no    -> unchanged (model false box stands)
  label_unmatched + no    -> that track's points in the run are deleted (label was wrong)
  label_unmatched + yes   -> unchanged (model miss stands)
  spot_check + no         -> that track's points within +-45 frames are IGNORED (label drift, not adjudicated per frame)
  any 'unsure'            -> the run (or +-45 frames for a spot check) is IGNORED for that track/box
Scores both the draft and the reviewed truth for every model given. Detection only (ROI-scoped, IoU 0.5); tracking
metrics are not re-scored here. Usage: rescore_reviewed.py DEC_DIR name=package_dir ..."""
import json, sys, glob, collections
from pathlib import Path
N = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029')
dec_dir = Path(sys.argv[1]); models = dict(a.split('=', 1) for a in sys.argv[2:])
traj_all = json.load(open(N / 'dense-real53/trajectories.json')); roi = traj_all['roi_xyxy']
inroi = lambda b: roi[0] <= (b[0]+b[2])/2 <= roi[2] and roi[1] <= (b[1]+b[3])/2 <= roi[3]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix*iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i) if i else 0
decisions = [json.load(open(f)) for f in glob.glob(str(dec_dir / 'decisions' / '*.json'))]
print('decisions', len(decisions), collections.Counter((d['kind'], d['answer']) for d in decisions))
# detections per model per video (official harness output, post-floor)
det = {m: {} for m in models}
for m, root in models.items():
    for v in (2, 4):
        p = Path(root) / f'cap_seal_video_{v}/detections.jsonl'
        det[m][v] = {json.loads(l)['frame']: [b for b in json.loads(l)['boxes_xyxy'] if inroi(b)] for l in open(p)}
def build_truth(v, reviewed):
    traj = next(x for x in traj_all['videos'] if f'cap_seal_video_{v}' in x['video'])
    truth = collections.defaultdict(list); ignore = collections.defaultdict(set)  # frame -> boxes; frame -> set of track ids ignored
    for ti, t in enumerate(traj['tracks']):
        for p in t['points']:
            if inroi(p['bbox_xyxy']): truth[p['frame']].append((ti, p['bbox_xyxy']))
    extra = collections.defaultdict(list); ignore_boxes = collections.defaultdict(list)
    if reviewed:
        for d in decisions:
            if d['video'] != v: continue
            a, b = (d['run'] or [d['frame'], d['frame']])
            if d['kind'] == 'spot_check': a, b = d['frame'] - 45, d['frame'] + 45
            if d['answer'] == 'unsure' or (d['kind'] == 'spot_check' and d['answer'] == 'no'):
                if d['track'] is not None:
                    for f in range(a, b + 1): ignore[f].add(d['track'])
                else:
                    for f in range(a, b + 1): ignore_boxes[f].append(d['box'])
            elif d['kind'] == 'label_unmatched' and d['answer'] == 'no':
                for f in range(a, b + 1): truth[f] = [(ti, bb) for ti, bb in truth[f] if ti != d['track']]
            elif d['kind'] == 'model_unmatched' and d['answer'] == 'yes':
                # the model's own boxes in that run (from the dense53 harness output, which defined the run) become truth
                src = det.get('dense53', det[next(iter(det))])[v]
                for f in range(a, b + 1):
                    for bb in src.get(f, []):
                        if iou(bb, d['box']) >= .3 or f == d['frame']: extra[f].append((-1, bb))
    for f, bs in extra.items():
        for bb in bs:
            if not any(iou(bb[1], t[1]) >= .5 for t in truth[f]): truth[f].append(bb)
    return truth, ignore, ignore_boxes
def score(m, v, truth, ignore, ignore_boxes):
    c = collections.Counter()
    for f, ds in det[m][v].items():
        ts = [(ti, b) for ti, b in truth.get(f, []) if ti not in ignore.get(f, ())]
        ign = [b for ti, b in truth.get(f, []) if ti in ignore.get(f, ())] + ignore_boxes.get(f, [])
        used = set()
        for b in ds:
            if any(iou(b, g) >= .3 for g in ign): continue
            j = max(range(len(ts)), key=lambda j: iou(b, ts[j][1]), default=None)
            if j is not None and iou(b, ts[j][1]) >= .5 and j not in used: used.add(j); c['TP'] += 1
            else: c['FP'] += 1
        c['FN'] += len(ts) - len(used)
    return c
print(f"| video | model | truth | TP/FP/FN | precision | recall |\n|---|---|---|---|---|---|")
tot = {}
for v in (2, 4):
    for label, reviewed in (('draft', False), ('reviewed', True)):
        truth, ignore, ignb = build_truth(v, reviewed)
        for m in models:
            c = score(m, v, truth, ignore, ignb); tot.setdefault((label, m), collections.Counter()).update(c)
            p = c['TP'] / (c['TP'] + c['FP']) if c['TP'] + c['FP'] else 0; r = c['TP'] / (c['TP'] + c['FN']) if c['TP'] + c['FN'] else 0
            print(f"| {v} | {m} | {label} | {c['TP']}/{c['FP']}/{c['FN']} | {p:.3f} | {r:.3f} |")
for (label, m), c in tot.items():
    p = c['TP'] / (c['TP'] + c['FP']) if c['TP'] + c['FP'] else 0; r = c['TP'] / (c['TP'] + c['FN']) if c['TP'] + c['FN'] else 0
    print(f"| 2+4 | {m} | {label} | {c['TP']}/{c['FP']}/{c['FN']} | {p:.3f} | {r:.3f} |")
