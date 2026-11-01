"""Build the human-review queue: frames where dense53's official harness output and the draft labels disagree on
videos 2 and 4 (every run of consecutive disagreement frames contributes one representative), plus a spot check of
every draft track at regular intervals. Writes items JSON with base64 crops. Usage: review_items.py [videos...]"""
import json, sys, base64, collections, numpy as np, cv2
from pathlib import Path
N = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029'); root = N / 'dense-real53/real-dense53'
V = '/home/hackathon5/liamreporoot/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/external/external/csi-vial-fill-dataset/cap_seal/videos/'
OUT = Path('/tmp/claude-1001/-home-hackathon5/aaa6ebb1-4fc4-49e4-b5af-b3e121e51663/scratchpad/page')
vids = [int(v) for v in sys.argv[1:]] or [2, 4]
traj_all = json.load(open(N / 'dense-real53/trajectories.json')); roi = traj_all['roi_xyxy']
inroi = lambda b: roi[0] <= (b[0]+b[2])/2 <= roi[2] and roi[1] <= (b[1]+b[3])/2 <= roi[3]
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix*iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i) if i else 0
def runs(frames):
    frames = sorted(frames); out = []; start = prev = None
    for f in frames:
        if prev is None or f != prev + 1:
            if start is not None: out.append((start, prev))
            start = f
        prev = f
    if start is not None: out.append((start, prev))
    return out
items = []
for v in vids:
    stem = f'cap_seal_video_{v}'; traj = next(x for x in traj_all['videos'] if stem in x['video'])
    det = {}
    for l in open(root / stem / 'detections.jsonl'):
        d = json.loads(l); det[d['frame']] = [b for b in d['boxes_xyxy'] if inroi(b)]
    truth = collections.defaultdict(list)
    for ti, t in enumerate(traj['tracks']):
        for p in t['points']: truth[p['frame']].append((ti, p['bbox_xyxy']))
    fn_frames = collections.defaultdict(list); fp_frames = []
    for f in range(traj['frames']):
        ts = [(ti, b) for ti, b in truth.get(f, []) if inroi(b)]; ds = det.get(f, []); used = set()
        for b in ds:
            j = max(range(len(ts)), key=lambda j: iou(b, ts[j][1]), default=None)
            if j is not None and iou(b, ts[j][1]) >= .5 and j not in used: used.add(j)
            else: fp_frames.append((f, b))
        for j, (ti, b) in enumerate(ts):
            if j not in used: fn_frames[ti].append(f)
    # FN runs per track -> one representative (middle frame) per run; FP runs -> same, grouped by box proximity
    for ti, fr in fn_frames.items():
        for a, b in runs(fr):
            f = (a + b) // 2; box = next(bb for tj, bb in truth[f] if tj == ti)
            items.append(dict(video=v, frame=f, kind='label_unmatched', box=box, run=[a, b], track=ti, q='Is there a cap inside the green box?'))
    fpr = []; cur = []
    for f, b in sorted(fp_frames):
        if cur and (f != cur[-1][0] + 1 or iou(b, cur[-1][1]) < .3): fpr.append(cur); cur = []
        cur.append((f, b))
    if cur: fpr.append(cur)
    for r in fpr:
        f, b = r[len(r) // 2]; items.append(dict(video=v, frame=f, kind='model_unmatched', box=b, run=[r[0][0], r[-1][0]], track=None, q='Is the red box a cap?'))
    # spot checks: every track, every 90 frames
    for ti, t in enumerate(traj['tracks']):
        pts = [p for p in t['points'] if inroi(p['bbox_xyxy'])]
        for p in pts[::90]:
            items.append(dict(video=v, frame=p['frame'], kind='spot_check', box=p['bbox_xyxy'], run=None, track=ti, q='Is the green box tight on a cap?'))
print('items', len(items), collections.Counter((i['video'], i['kind']) for i in items))
# crops
by_video = collections.defaultdict(list)
for i, it in enumerate(items): by_video[it['video']].append(i)
for v, idx in by_video.items():
    cap = cv2.VideoCapture(V + f'cap_seal_video_{v}.mp4')
    for i in sorted(idx, key=lambda k: items[k]['frame']):
        it = items[i]; cap.set(cv2.CAP_PROP_POS_FRAMES, it['frame']); ok, img = cap.read()
        if not ok: it['img'] = None; continue
        x0, y0, x1, y1 = [int(round(c)) for c in it['box']]; cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        X0 = max(0, min(640 - 160, cx - 80)); Y0 = max(0, min(480 - 120, cy - 60))
        crop = img[Y0:Y0 + 120, X0:X0 + 160].copy()
        crop = cv2.resize(crop, (480, 360), interpolation=cv2.INTER_CUBIC)
        col = (0, 0, 255) if it['kind'] == 'model_unmatched' else (0, 200, 0)
        cv2.rectangle(crop, ((x0 - X0) * 3, (y0 - Y0) * 3), ((x1 - X0) * 3, (y1 - Y0) * 3), col, 2)
        ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 80]); it['img'] = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
        # context strip: whole lane column downscaled with the box marked
        lane = img[:, max(0, roi[0] - 40):roi[2] + 40].copy(); cv2.rectangle(lane, (x0 - max(0, roi[0] - 40), y0), (x1 - max(0, roi[0] - 40), y1), col, 2)
        lane = cv2.resize(lane, (lane.shape[1] // 2, 240), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', lane, [cv2.IMWRITE_JPEG_QUALITY, 70]); it['ctx'] = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
    cap.release()
items = [it for it in items if it.get('img')]
for k, it in enumerate(items): it['id'] = f"v{it['video']}-f{it['frame']}-{it['kind'][:5]}-{k}"
json.dump(dict(items=items, roi=roi, model='dense53 + floor 0.90 (official harness detections)'), open(OUT / 'review_items.json', 'w'))
print('wrote', len(items), 'items,', (OUT / 'review_items.json').stat().st_size // 1024, 'KB')
