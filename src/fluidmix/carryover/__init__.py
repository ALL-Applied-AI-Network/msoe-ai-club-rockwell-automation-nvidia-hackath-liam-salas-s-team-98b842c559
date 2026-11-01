"""End-to-end carryover decision for video 5: measure the final mixture colour of each vial,
compare it with the settled colour of the same recipe (leave-one-out), apply a fixed rule and
write an operator message. See measure.py (what is measured), decide.py (the fixed rule) and
explain.py (the message). Everything here is deterministic and nothing is trained. The tau rule
is fixed in decide.py, but its noise term is computed in-sample from the scored vials and its
floor coincides with an older video-5 statistic; see decide.py and results.md section (f).
"""
from fluidmix.carryover.measure import lab, cie76, measure_vials
from fluidmix.carryover.decide import reference_for, frozen_tau, decide
from fluidmix.carryover.explain import explain

__all__ = ["lab", "cie76", "measure_vials", "reference_for", "frozen_tau", "decide", "explain"]
