"""Synthetic checks for the residue-impact context packet. They test the protocol, not the plant."""
from dataclasses import replace
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from fluidmix.residue_context import (Config, InMemoryHistory, LeakageError, Measurement, Switch, build_packet,
                                      check_explanation, holdup_from_switch, template_explanation)
from residue_packet import recipe, synthetic_history

SEQ = ("R-ORANGE", "R-BLUE", "R-EXAMPLE", "R-EVEN", "R-PALE", "R-ORANGE", "R-BLUE")


def packet_after(previous="R-BLUE", new="R-EXAMPLE", **kw):
    history, programmed = synthetic_history(SEQ[:-1] + (previous,), **kw)
    cursor = programmed[-1][0] + 1
    return build_packet(recipe(new), recipe(previous), cursor, history), history, cursor


def test_recovers_the_holdup_and_flags_a_coloured_switch():
    packet, _, _ = packet_after("R-BLUE")
    lo, mid, hi = packet["derived"]["holdup_ml_quantiles"]
    assert 18 < lo <= mid <= hi < 22                     # synthetic truth: 20 mL
    assert packet["decision"]["risk"] == "likely_impact"


def test_same_recipe_residue_is_unlikely_to_matter():
    packet, _, _ = packet_after("R-EXAMPLE")
    assert packet["derived"]["same_recipe_as_previous"]
    assert packet["decision"]["risk"] == "unlikely"


def test_packet_is_deterministic_and_order_independent():
    a, history, cursor = packet_after()
    shuffled = InMemoryHistory(random.Random(1).sample(history.measurements, len(history.measurements)),
                               random.Random(2).sample(history._switches, len(history._switches)), history.clean_ids)
    b = build_packet(recipe("R-EXAMPLE"), recipe("R-BLUE"), cursor, shuffled)
    assert a["packet_sha256"] == b["packet_sha256"]


def test_future_records_do_not_change_the_packet():
    a, history, cursor = packet_after()
    future = Measurement("E-FUTURE", cursor + 5, "train", "V-F", "R-EXAMPLE", (1.0, 2.0, 3.0), (200.0, 200.0, 190.0))
    history.measurements.append(future)
    history.clean_ids.add("E-FUTURE")
    b = build_packet(recipe("R-EXAMPLE"), recipe("R-BLUE"), cursor, history)
    assert a["packet_sha256"] == b["packet_sha256"]


class LeakySource(InMemoryHistory):
    """Ignores `before` and roles, as a buggy store might."""

    def clean_references(self, recipe_id, before, allowed_roles):
        return [m for m in self.measurements if m.evidence_id in self.clean_ids and m.recipe_id == recipe_id]


def test_builder_rejects_future_or_disallowed_records_from_a_buggy_source():
    _, history, cursor = packet_after()
    leaky = LeakySource(history.measurements, history._switches, history.clean_ids)
    with pytest.raises(LeakageError):
        build_packet(recipe("R-EXAMPLE"), recipe("R-BLUE"), 5, leaky)   # its clean refs are at cursor 8, 9
    test_vials = [replace(m, role="test") if m.recipe_id == "R-EXAMPLE" else m for m in history.measurements]
    with pytest.raises(LeakageError):
        build_packet(recipe("R-EXAMPLE"), recipe("R-BLUE"), cursor, LeakySource(test_vials, [], history.clean_ids))


def test_unseen_recipe_is_insufficient_evidence_and_says_why():
    history, programmed = synthetic_history(("R-ORANGE", "R-BLUE", "R-EVEN", "R-PALE", "R-BLUE"))
    packet = build_packet(recipe("R-EXAMPLE"), recipe("R-BLUE"), programmed[-1][0] + 1, history)
    assert packet["decision"]["risk"] == "insufficient_evidence"
    assert any("clean_references" in r for r in packet["decision"]["reasons"])
    text = template_explanation(packet)
    assert "Missing" in text and check_explanation(text, packet) == []


def test_out_of_range_fraction_is_excluded_not_clipped():
    _, history, _ = packet_after()
    s = history._switches[0]
    no_residue = Switch("S-CLEAN", s.previous, replace(s.first, rgb=s.clean.rgb), s.clean, s.first_total_ml)
    assert holdup_from_switch(no_residue, Config().min_contrast) == (0.0, 0.0, "used")
    darker = Switch("S-DARK", s.previous, replace(s.first, rgb=(0.0, 0.0, 0.0)), s.clean, s.first_total_ml)
    h, f, status = holdup_from_switch(darker, Config().min_contrast)
    assert status == "fraction_out_of_range" and h is None and f is not None and not 0 <= f <= 1


def test_guard_accepts_the_template_and_rejects_invented_content():
    packet, _, _ = packet_after()
    assert check_explanation(template_explanation(packet), packet) == []
    assert "no evidence cited" in check_explanation("Residue will matter.", packet)
    assert any("unknown evidence id" in p for p in check_explanation("See [E-999].", packet))
    assert any("number not in packet" in p for p in check_explanation("Hold-up is 55 mL [E-24].", packet))
    assert any("states risk" in p for p in check_explanation("Risk is unlikely [E-24].", packet))
