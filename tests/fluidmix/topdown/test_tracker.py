"""The mid-lane re-birth tracker variant: provisional births, and exact agreement with the guarded tracker when off."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from fluidmix.topdown.conveyor_guarded_tracking import ConveyorGuardedTracker
from fluidmix.topdown.conveyor_rebirth_tracking import ConveyorRebirthTracker

SETTINGS = dict(entry_y=436, exit_y=22, max_gap=5, gate=42, min_hits=3, retire_on_exit=False, recovery_gap=8, ambiguity_margin=0.15)


def moving_cap(start_y, frames, step=-4, x=160):
    return [(f, [(x - 8, start_y + step * f - 8, 16, 16)]) for f in range(frames)]


def test_midlane_birth_is_provisional_until_min_hits():
    tr = ConveyorRebirthTracker(**SETTINGS)
    emitted = [len(tr.update(f, boxes)[0]) for f, boxes in moving_cap(240, 6)]
    assert emitted == [0, 0, 1, 1, 1, 1]  # first emitted on the third accepted observation
    assert len(tr.tracks) == 1 and len(tr.confirmed) == 1


def test_midlane_box_is_never_emitted_by_the_guarded_tracker():
    tr = ConveyorGuardedTracker(**SETTINGS)
    assert all(len(tr.update(f, boxes)[0]) == 0 for f, boxes in moving_cap(240, 6))


def test_entry_band_births_are_unchanged_and_variant_off_matches_guarded():
    seq = moving_cap(460, 40)
    a, b, c = ConveyorGuardedTracker(**SETTINGS), ConveyorRebirthTracker(**SETTINGS), ConveyorRebirthTracker(midlane_births=False, **SETTINGS)
    outs = [[(t.track_id, t.bbox) for t in tr.update(f, boxes)[0]] for tr in (a, b, c) for f, boxes in seq]
    n = len(seq)
    assert outs[:n] == outs[n:2 * n] == outs[2 * n:]
    assert len(a.tracks) == len(b.tracks) == 1


def test_rebirth_after_a_long_dropout_gets_a_new_id_and_can_exit():
    tr = ConveyorRebirthTracker(**SETTINGS)
    seq = moving_cap(460, 30)                       # born in the entry band, moves up to y ~ 344
    seq += [(f, []) for f in range(30, 60)]         # 30 frames of nothing: beyond max_gap + recovery_gap
    seq += [(f, [(152, 200 - 4 * (f - 60) - 8, 16, 16)]) for f in range(60, 110)]  # reappears mid-lane, crosses y=22
    exits = []; ids = set()
    for f, boxes in seq:
        visible, events = tr.update(f, boxes); exits += events; ids.update(t.track_id for t in visible)
    assert len(ids) == 2 and len(exits) == 1 and exits[0]['track_id'] == sorted(ids)[1]
