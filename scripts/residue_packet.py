"""Demo of the residue-impact context packet on a SYNTHETIC history (docs/residue-context.md).

    python scripts/residue_packet.py                  # the INN-62 example: 20 mL T100, 10 mL T200, 5 mL T400
    python scripts/residue_packet.py --previous R-ORANGE --json packet.json

The synthetic history below exists only to exercise the protocol. Its dye absorbances are
made-up fixture values that the packet builder never sees: it only gets "measured" colours, as
the pipeline would. Nothing here is plant evidence or real video.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.absorbance import linear_to_srgb, srgb_to_linear, transmit
from fluidmix.residue_context import (Dose, InMemoryHistory, Measurement, Recipe, Switch, build_packet,
                                      check_explanation, render_prompt, template_explanation)

# Fixture only: hidden "truth" used to synthesize measured colours. Never passed to the builder.
_DYE_A = {"T100": np.zeros(3), "T200": np.array([0.05, 0.45, 1.1]), "T400": np.array([1.0, 0.35, 0.05])}
_EMPTY = (200.0, 200.0, 190.0)
_HOLDUP_ML = 20.0
ORDER = ("T200", "T400", "T100")

RECIPES = {
    "R-EXAMPLE": {"T100": 20, "T200": 10, "T400": 5},
    "R-ORANGE": {"T100": 10, "T200": 40},
    "R-BLUE": {"T100": 20, "T400": 50},
    "R-EVEN": {"T100": 20, "T200": 20, "T400": 20},
    "R-PALE": {"T100": 45, "T200": 15, "T400": 15},
}


def recipe(recipe_id):
    ml = RECIPES[recipe_id]
    return Recipe(recipe_id, tuple(Dose(t, float(ml[t])) for t in ORDER if ml.get(t)))


def _absorbance(recipe_id):
    ml = RECIPES[recipe_id]
    total = sum(ml.values())
    return sum(v / total * _DYE_A[t] for t, v in ml.items())


def _rgb(a, rng):
    rgb = linear_to_srgb(transmit(a, srgb_to_linear(_EMPTY))) + rng.normal(0, 0.4, 3)
    return tuple(float(round(c, 2)) for c in np.clip(rgb, 0, 255))


def synthetic_history(sequence=("R-ORANGE", "R-BLUE", "R-EXAMPLE", "R-EVEN", "R-PALE", "R-BLUE", "R-ORANGE"),
                      vials_per_recipe=3, seed=0, role="train"):
    """Measured-looking vials and switches. The first vial after each change carries _HOLDUP_ML of
    the previous vial's mix. Returns (history, programmed list of (cursor, recipe_id))."""
    rng = np.random.default_rng(seed)
    measurements, switches, clean, programmed = [], [], set(), []
    prev_a, prev_m, cursor = None, None, 0
    for rid in sequence:
        a_new = _absorbance(rid)
        total = sum(RECIPES[rid].values())
        vials = []
        for k in range(vials_per_recipe):
            cursor += 1
            a = a_new
            if k == 0 and prev_a is not None:
                f = _HOLDUP_ML / total
                a = (1 - f) * a_new + f * prev_a
            m = Measurement(f"E-{cursor}", cursor, role, f"V-{cursor}", rid, _rgb(a, rng), _EMPTY, f"IMG-{cursor}")
            measurements.append(m)
            vials.append(m)
            programmed.append((cursor, rid))
            if k > 0:
                clean.add(m.evidence_id)
        if prev_m is not None:
            switches.append(Switch(f"S-{vials[0].cursor}", prev_m, vials[0], vials[1], float(total)))
        prev_a, prev_m = a_new, vials[-1]
    return InMemoryHistory(measurements, switches, clean), programmed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", default="R-EXAMPLE", choices=RECIPES)
    ap.add_argument("--previous", default="R-BLUE", choices=RECIPES, help="recipe of the vial just before")
    ap.add_argument("--json", help="also write the packet here")
    a = ap.parse_args()
    seq = ("R-ORANGE", "R-BLUE", "R-EXAMPLE", "R-EVEN", "R-PALE", "R-ORANGE", "R-BLUE", a.previous)
    history, programmed = synthetic_history(seq)
    cursor = programmed[-1][0] + 1
    packet = build_packet(recipe(a.new), recipe(a.previous), cursor, history)
    print(f"target cursor {cursor}: {a.previous} -> {a.new}  (synthetic history, hold-up {_HOLDUP_ML:g} mL)")
    print(json.dumps({k: packet[k] for k in ("coverage", "derived", "decision")}, indent=1))
    explanation = template_explanation(packet)
    print("\ntemplate explanation:\n" + explanation)
    print("\nguard on template:", check_explanation(explanation, packet) or "pass")
    print(f"\nmodel prompt: {len(render_prompt(packet))} characters, images {packet['images']}")
    print("packet_sha256", packet["packet_sha256"])
    if a.json:
        Path(a.json).write_text(json.dumps(packet, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
