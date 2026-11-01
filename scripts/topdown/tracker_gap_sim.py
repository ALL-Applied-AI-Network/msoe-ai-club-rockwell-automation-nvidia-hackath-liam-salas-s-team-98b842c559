"""Tracker-side diagnostic: replay ConveyorGuardedTracker over a package's saved per-frame tracker inputs with wider
gap settings and score exits/fragmentations with the harness's own score_tracking. The frozen harness stays at
max_gap 5 / recovery_gap 8; this only measures what a protocol change would buy. Usage: tracker_gap_sim.py PACKAGE_DIR"""
import json, sys, collections, time
from pathlib import Path
from dataclasses import asdict
N = Path('/home/hackathon5/outputs/topdown-01a17f15-20261029'); root = Path(sys.argv[1])
sys.path.insert(0, str(N / 'dense-real53/source')); sys.path.insert(0, str(N / 'dense-evaluation28/source'))
from fluidmix.conveyor_guarded_tracking import ConveyorGuardedTracker
from heldout_tracking import score_tracking
proto = json.load(open(N / 'dense-real53/protocol.json')); base_t = dict(proto['tracker']); base_o = dict(proto['tracker_options'])
traj_all = json.load(open(N / 'dense-real53/trajectories.json'))
variants = [('official 5+8', 5, 8), ('5+20', 5, 20), ('10+20', 10, 20), ('15+30', 15, 30), ('30+30', 30, 30)]
inputs = {}
for v in range(1, 6):
    rows = [json.loads(l) for l in open(root / f'cap_seal_video_{v}/detections.jsonl')]
    inputs[v] = [(r['frame'], r['tracker_boxes_xywh'], r['tracker_confidence']) for r in rows]
print('| tracker gap (max_gap + recovery_gap) | ID switches | merges | fragmentations | reference exits found | camera IDs |')
print('|---|---|---|---|---|---|')
for name, mg, rg in variants:
    t0 = time.time(); tot = collections.Counter(); ids = 0
    for v in range(1, 6):
        traj = next(x for x in traj_all['videos'] if f'cap_seal_video_{v}' in x['video'])
        tr = ConveyorGuardedTracker(**dict(base_t, max_gap=mg), **dict(base_o, recovery_gap=rg)); outputs = []
        for f, xywh, conf in inputs[v]:
            visible, emitted = tr.update(f, xywh, confidences=conf)
            for track in visible: outputs.append(dict(frame=f, track=asdict(track)))
        s = score_tracking(traj, outputs, traj['frames'], {})
        tot['sw'] += s['identity_switches']; tot['mg'] += s['identity_merges']; tot['frag'] += s['fragmentations']
        tot['ex'] += s['expected_reference_exits']; tot['exm'] += s['missed_reference_exits']; ids += len(tr.tracks)
    print(f"| {name} | {tot['sw']} | {tot['mg']} | {tot['frag']} | {tot['ex']-tot['exm']}/{tot['ex']} | {ids} |", flush=True)
    print(f'   ({time.time()-t0:.0f}s)', file=sys.stderr)
