"""Replay each model's saved per-frame tracker inputs (the harness packages on the Spark) through the frozen guarded
tracker and the mid-lane re-birth variant, scoring both with the harness's own score_tracking. Detections are
unchanged, so precision/recall are the published ones; only the tracker protocol differs. The frozen-tracker rows must
reproduce data/topdown/harness-report-*.json. Usage: rebirth_tracker_sim.py [--spark DIR] [--out JSON]"""
import argparse
import json, sys, collections
from pathlib import Path
from dataclasses import asdict
ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--spark', default='/home/hackathon5/outputs/topdown-01a17f15-20261029'); ap.add_argument('--out', default=None); a = ap.parse_args()
N = Path(a.spark); ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src')); sys.path.append(str(N / 'dense-real53/source'))  # our fluidmix first; the harness scorer after
from fluidmix.topdown.conveyor_guarded_tracking import ConveyorGuardedTracker
from fluidmix.topdown.conveyor_rebirth_tracking import ConveyorRebirthTracker
from heldout_tracking import score_tracking  # the frozen scorer, on the Spark only
PK = {'train12 YOLO baseline': N / 'real-test12/real-train12', 'dense53': N / 'dense-real53/real-dense53', 'dense60': N / 'dense-real60/real-dense60'}
proto = json.load(open(N / 'dense-real53/protocol.json')); man = json.load(open(N / 'dense-real53/manifest.json'))
traj_all = json.load(open(N / 'dense-real53/trajectories.json'))
def replay(cls, rows, rec, traj):
    tr = cls(**proto['tracker'], **proto['tracker_options']); outputs = []
    for r in rows:
        visible, _ = tr.update(r['frame'], r['tracker_boxes_xywh'], confidences=r['tracker_confidence'])
        for t in visible: outputs.append(dict(frame=r['frame'], track=asdict(t)))
    s = score_tracking(traj, outputs, rec['frames'], {})
    ids = len(getattr(tr, 'confirmed', tr.tracks))
    return dict(tTP=s['track_point_TP'], tFP=s['track_point_FP'], tFN=s['track_point_FN'], frag=s['fragmentations'], sw=s['identity_switches'], mg=s['identity_merges'],
                ex=s['expected_reference_exits'], exm=s['missed_reference_exits'], dup=s['duplicate_reference_exits'], unx=s['unexpected_reference_exits'], ids=ids, born=len(tr.tracks))
res = {}
print('| model | tracker | track identity coverage | track points FP | fragmentations | ID switches | merges | reference exits found | duplicate / unexpected exits | emitted camera IDs |')
print('|---|---|---|---|---|---|---|---|---|---|')
for name, pk in PK.items():
    if not pk.exists(): print('missing', pk); continue
    for label, cls in (('frozen v4 (official)', ConveyorGuardedTracker), ('mid-lane re-birth (declared)', ConveyorRebirthTracker)):
        tot = collections.Counter(); per = {}
        for v in range(1, 6):
            stem = f'cap_seal_video_{v}'; rec = next(r for r in man['records'] if stem in r['path']); traj = next(x for x in traj_all['videos'] if stem in x['video'])
            rows = [json.loads(l) for l in open(pk / stem / 'detections.jsonl')]
            s = replay(cls, rows, rec, traj); per[v] = s; tot.update(s)
        res[f'{name} | {label}'] = dict(total=dict(tot), per_video=per)
        print(f"| {name} | {label} | {tot['tTP']/(tot['tTP']+tot['tFN']):.3f} | {tot['tFP']} | {tot['frag']} | {tot['sw']} | {tot['mg']} | {tot['ex']-tot['exm']}/{tot['ex']} | {tot['dup']} / {tot['unx']} | {tot['ids']} |", flush=True)
if a.out: json.dump(res, open(a.out, 'w'), indent=1)
