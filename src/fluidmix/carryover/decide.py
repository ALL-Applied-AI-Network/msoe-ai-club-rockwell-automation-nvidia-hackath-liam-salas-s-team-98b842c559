"""Fixed carryover decision rule (unchanged since the first run; never re-tuned on an outcome).

Provenance, stated honestly (review of 2026-11-01): the floor 2.3 and the multiplier 3 were
chosen by hand. 2.3 is the value commonly quoted as the CIE76 just-noticeable difference, but it
also equals the median position-2 dE of these same video-5 vials in data/video5_vial_colors.csv
and docs/plan.md (committed 2026-10-31), and there is no dated freeze record predating that
file, so the floor cannot be claimed independent of video 5. The noise term is computed
in-sample from the measurement noise of the vials being scored. The author states the rule was
written before the first run of scripts/carryover_video5.py, but no auditable pre-run record of
it exists (it was not committed or recorded in a dated freeze note before the run), so that
ordering rests on the author's statement alone. Any future tau must be set on twin val per
docs/evaluation.md and frozen in a dated, committed note.

Reference colour (measured from the other vials): for a vial of recipe R, the per-channel median
Lab of the OTHER vials of recipe R whose position_in_run >= 2 (leave-one-out: a vial never
contributes to its own reference). No such peer -> insufficient_evidence.

PRIMARY RULE (twin-frozen): tau = 2.0, the CIE76 visibility threshold the team froze on twin renders before any
real measurement; decision = discard if CIE76(measured, reference) > tau else keep.
SECONDARY RULE (kept for the record): tau = max(2.3, 3 * median(measurement_noise_dE over all eligible vials));
it gives the same decisions on video 5.

Baseline: a stand-in for the line's known-colour check (pass if the colour is within tau of ANY
recipe's reference colour; the real Teledyne outputs are not in the dataset). The vial's own
recipe uses the same leave-one-out reference. Such a check cannot tell a wrong-but-known colour
from the right one.
"""
import numpy as np

from fluidmix.carryover.measure import cie76

TAU_FLOOR = 2.3
TAU_NOISE_MULTIPLIER = 3.0
# Primary threshold since the 2026-11-01 review: the colour-visibility boundary the team's twin pipeline froze on twin
# renders before any real measurement (CIE76 <= 2.0 for same-recipe/stage clean-reference agreement and the DP3
# visibility gate; pre-render freeze of 2026-10-30 states that neither the WP1 test nor video 5 may set it). Recorded in
# outputs/pipeline-20261028/PROGRESS.md (lines about "frozen CIE76 <= 2.0") and FINAL-REPORT.md ("frozen 2.0
# threshold"). On video 5 it gives the same decisions as the noise-based rule below, which is kept as the secondary rule.
TAU_TWIN = 2.0
TAU_TWIN_SOURCE = ("CIE76 2.0: the colour-visibility threshold frozen on twin renders by the team's pipeline "
                   "(outputs/pipeline-20261028 PROGRESS.md and FINAL-REPORT.md) before any real-video measurement")


def reference_for(vial, records):
    """Leave-one-out reference for `vial`: (median Lab of steady peers, source text) or (None, text)."""
    me = records[vial]
    peers = [r for v, r in records.items()
             if v != vial and r["recipe_id"] == me["recipe_id"] and r["position_in_run"] >= 2
             and r.get("measured_lab") is not None]
    source = (f"median of recipe {me['recipe_id']} vials at position>=2 excluding this vial "
              f"(vials {', '.join(str(r['vial']) for r in peers) or 'none'})")
    if not peers:
        return None, source
    return [float(x) for x in np.median([r["measured_lab"] for r in peers], 0)], source


def frozen_tau(noises):
    """tau = max(2.3, 3 * median noise); floor and multiplier chosen by hand, noise in-sample."""
    noises = [n for n in noises if n is not None]
    if not noises:
        return TAU_FLOOR
    return max(TAU_FLOOR, TAU_NOISE_MULTIPLIER * float(np.median(noises)))


def _known_colour_check(vial, records, tau):
    """pass if within tau of any recipe's leave-one-out reference colour, else fail."""
    me = records[vial]
    best = None
    for recipe in sorted({r["recipe_id"] for r in records.values()}):
        peers = [r for v, r in records.items()
                 if v != vial and r["recipe_id"] == recipe and r["position_in_run"] >= 2
                 and r.get("measured_lab") is not None]
        if not peers:
            continue
        d = cie76(me["measured_lab"], np.median([r["measured_lab"] for r in peers], 0))
        best = d if best is None else min(best, d)
    if best is None:
        return None
    return "pass" if best <= tau else "fail"


def decide(records, tau=None):
    """Fill reference_lab, reference_source, dE_vs_reference, tau, decision, reason and
    known_colour_check in every record (dict keyed by vial). Returns tau."""
    if tau is None:
        tau = TAU_TWIN
    for vial, r in records.items():
        r["tau"] = round(tau, 3)
        r["reference_lab"], r["reference_source"] = None, None
        r["dE_vs_reference"], r["known_colour_check"] = None, None
        if r.get("measured_lab") is None:
            r["decision"] = "insufficient_evidence"
            r["reason"] = r.get("exclusion_reason") or "no measurement"
            continue
        ref, source = reference_for(vial, records)
        r["reference_source"] = source
        if ref is None:
            r["decision"] = "insufficient_evidence"
            r["reason"] = f"no other recipe {r['recipe_id']} vial at position>=2 to compare with"
            continue
        d = cie76(r["measured_lab"], ref)
        r["reference_lab"] = [round(x, 2) for x in ref]
        r["dE_vs_reference"] = round(d, 2)
        r["decision"] = "discard" if d > tau else "keep"
        r["reason"] = (f"dE_vs_reference {d:.1f} {'>' if d > tau else '<='} tau {tau:.1f}")
        r["known_colour_check"] = _known_colour_check(vial, records, tau)
    return tau


def flagged_count(records, scale):
    """Diagnostic only: how many decided vials would be discarded at tau * scale."""
    return sum(1 for r in records.values()
               if r.get("dE_vs_reference") is not None and r["dE_vs_reference"] > r["tau"] * scale)
