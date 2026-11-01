"""Score confidence rules offline against the frozen scorer, from floor-free dense53 components (raw/*.jsonl, written by raw_infer.py beside this script on the Spark).

Rules (declared before any real scoring of the continuation rule, 2026-11-01 01:50 CST):
  official      component kept iff confidence >= 0.90 (the frozen floor; must reproduce the harness report)
  continuation  component kept iff confidence >= 0.90, OR it overlaps (IoU >= 0.3) a component kept in the previous
                frame and that chain of sub-floor continuations is at most 5 frames long (the tracker's own max_gap).
                The floor gates births; continuations may not be lower than 0.5 (the component threshold itself).
Everything downstream (lane filter, tracker, scorer) is the frozen harness code imported from dense-real53/source.
Usage: score_rules.py [rule ...]"""
import json, sys, collections
from pathlib import Path
from dataclasses import asdict
N = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029'); H = Path(__file__).resolve().parent
sys.path.insert(0, str(N / 'dense-real53/source'))
from fluidmix.conveyor_guarded_tracking import ConveyorGuardedTracker
from heldout_tracking import score_tracking
from evaluate_pair import tracker_inputs, dense_detection_score
from heldout_evaluate import iou
proto = json.load(open(N / 'dense-real53/protocol.json')); lane = proto['inference']['lane_xyxy']
traj_all = json.load(open(N / 'dense-real53/trajectories.json')); man = json.load(open(N / 'dense-real53/manifest.json'))
FLOOR, IOU, MAXRUN = 0.90, 0.3, 5

def apply(rule, rows):
    out = []; prev = []  # prev: list of (box, confidence, consecutive sub-floor frames so far)
    for r in rows:
        kept = []
        for b, s in zip(r['boxes_xyxy'], r['confidence']):
            if s >= FLOOR: kept.append((b, s, 0)); continue
            if rule == 'continuation':
                run = min((pr for pb, _, pr in prev if iou(b, pb) >= IOU), default=None)
                if run is not None and run < MAXRUN: kept.append((b, s, run + 1))
        out.append(dict(frame=r['frame'], boxes_xyxy=[b for b, _, _ in kept], confidence=[s for _, s, _ in kept]))
        prev = kept
    return out

def run_video(rule, v, rows):
    stem = f'cap_seal_video_{v}'; rec = next(r for r in man['records'] if stem in r['path'])
    traj = next(x for x in traj_all['videos'] if stem in x['video'])
    det = apply(rule, rows)
    tr = ConveyorGuardedTracker(**proto['tracker'], **proto['tracker_options']); outputs = []
    for d in det:
        xywh, c = tracker_inputs(d['boxes_xyxy'], d['confidence'], lane)
        visible, _ = tr.update(d['frame'], xywh, confidences=c)
        for t in visible: outputs.append(dict(frame=d['frame'], track=asdict(t)))
    dd = dense_detection_score(traj, det); ts = score_tracking(traj, outputs, rec['frames'], {})
    return dict(TP=dd['TP'], FP=dd['FP'], FN=dd['FN'], tTP=ts['track_point_TP'], tFN=ts['track_point_FN'], frag=ts['fragmentations'],
                sw=ts['identity_switches'], mg=ts['identity_merges'], ex=ts['expected_reference_exits'], exm=ts['missed_reference_exits'], ids=len(tr.tracks),
                kept=sum(len(d['boxes_xyxy']) for d in det), raw=sum(len(r['boxes_xyxy']) for r in rows))

if __name__ == '__main__':
    rules = sys.argv[1:] or ['official', 'continuation']
    raw = {v: [json.loads(l) for l in open(H / f'raw/cap_seal_video_{v}.jsonl')] for v in range(1, 6) if (H / f'raw/cap_seal_video_{v}.jsonl').exists()}
    print('videos', sorted(raw))
    print('| rule | precision | recall | track identity coverage | fragmentations | ID switches | merges | reference exits found | camera IDs | boxes kept |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    res = {}
    for rule in rules:
        tot = collections.Counter(); per = {}
        for v, rows in raw.items():
            s = run_video(rule, v, rows); per[v] = s; tot.update(s)
        res[rule] = dict(total=dict(tot), per_video=per)
        print(f"| {rule} | {tot['TP']/(tot['TP']+tot['FP']):.3f} | {tot['TP']/(tot['TP']+tot['FN']):.3f} | {tot['tTP']/(tot['tTP']+tot['tFN']):.3f} | {tot['frag']} | {tot['sw']} | {tot['mg']} | {tot['ex']-tot['exm']}/{tot['ex']} | {tot['ids']} | {tot['kept']} of {tot['raw']} |", flush=True)
    json.dump(res, open(H / 'score_rules.json', 'w'), indent=1)
