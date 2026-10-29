"""Residue-impact context packet: what to pull from history before the next recipe, and the
deterministic call on whether residue from the previous vial will visibly change the first vial.

Proposal for INN-62/INN-63/INN-64, to integrate after the pipeline's clean push. See
docs/residue-context.md. Rules it follows (pipeline run 2026-10-28):
  * Inputs are only the programmed recipe sequence and *measured* camera evidence. No tank-colour
    presets, palettes, residue settings or labels.
  * Code makes the call (likely_impact / unlikely / uncertain / insufficient_evidence). A model may
    only explain the packet, citing its evidence IDs; check_explanation() rejects anything else,
    and template_explanation() is the fallback.
  * History is strictly earlier than the target and from allowed roles. The builder re-checks what
    the source returns and raises LeakageError instead of trusting it.

Model (all in absorbance, A = -log10(I / I0), which mixes linearly by volume fraction):
    A_first ~= (1 - f) * A_clean_new + f * A_residue,   f = V_holdup / V_total_new
A_residue is the previous vial's measured absorbance (an assumption: the shared tube holds the
previous mix; video 5 shows it can hold more than the last dose). Each past switch gives a
closed-form f, hence a hold-up volume V_holdup = f * V_total. The spread of those volumes,
scaled to the new recipe's total, gives a predicted colour shift (CIE76 dE) against the new
recipe's clean reference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Protocol

import numpy as np

from fluidmix.absorbance import absorbance, linear_to_srgb, srgb_to_lab, srgb_to_linear, transmit

SCHEMA_VERSION = "residue-context-v1"


class LeakageError(ValueError):
    """History returned a record at or after the target, or from a role that is not allowed."""


@dataclass(frozen=True)
class Dose:
    tank: str                     # tank ID as programmed, e.g. "T200"
    ml: float


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    doses: tuple[Dose, ...]       # programmed dispense order

    @property
    def total_ml(self) -> float:
        return float(sum(d.ml for d in self.doses))


@dataclass(frozen=True)
class Measurement:
    """One measured vial. rgb is the measured liquid colour; empty_rgb the same region with the
    vial empty (I0), so absorbance is comparable across vials and lighting."""
    evidence_id: str
    cursor: int                   # position in the programmed vial sequence (time order)
    role: str                     # split role, e.g. "train", "validation"
    vial_id: str
    recipe_id: str
    rgb: tuple[float, float, float]
    empty_rgb: tuple[float, float, float] | None
    image_id: str | None = None


@dataclass(frozen=True)
class Switch:
    """A past recipe change: the last vial before it, the first vial after it, and a clean
    (settled) vial of the new recipe. All three are measurements."""
    evidence_id: str
    previous: Measurement
    first: Measurement
    clean: Measurement
    first_total_ml: float


class HistorySource(Protocol):
    """What the builder needs from the evidence store. INN-62 mapping (to confirm on integration):
    clean_references -> reference_bank, previous_vial -> vial_history, switches -> switch_history."""

    def clean_references(self, recipe_id: str, before: int, allowed_roles: frozenset[str]) -> list[Measurement]: ...

    def previous_vial(self, before: int, allowed_roles: frozenset[str]) -> Measurement | None: ...

    def switches(self, before: int, allowed_roles: frozenset[str]) -> list[Switch]: ...


@dataclass(frozen=True)
class Config:
    """Frozen settings. threshold_de76 must be frozen on validation before any test use; the
    default is a placeholder (2.0 = the store's measured-regime tolerance), not a tuned value."""
    config_version: str = "residue-context-config-v1-unfrozen"
    threshold_de76: float = 2.0
    min_clean_references: int = 2
    min_switches: int = 3
    quantiles: tuple[float, float, float] = (0.25, 0.5, 0.75)
    min_contrast: float = 0.02    # |A_residue - A_clean| below this: f is undefined for a switch
    allowed_roles: frozenset[str] = field(default_factory=lambda: frozenset({"train", "validation"}))
    max_images: int = 3


# ------------------------------------------------------------------ measured colour helpers
def absorbance_of(m: Measurement) -> np.ndarray | None:
    if m.empty_rgb is None:
        return None
    return absorbance(srgb_to_linear(m.rgb), srgb_to_linear(m.empty_rgb))


def holdup_from_switch(s: Switch, min_contrast: float):
    """(holdup_ml or None, f or None, reason). f is not clipped: out-of-range values are excluded
    with a reason, never forced into [0, 1]."""
    a_first, a_clean, a_prev = (absorbance_of(m) for m in (s.first, s.clean, s.previous))
    if a_first is None or a_clean is None or a_prev is None:
        return None, None, "missing_empty_reference"
    e = a_prev - a_clean
    if float(np.sqrt(e @ e)) < min_contrast:
        return None, None, "residue_matches_clean"
    f = float((a_first - a_clean) @ e / (e @ e))
    if not 0.0 <= f <= 1.0:
        return None, f, "fraction_out_of_range"
    return f * s.first_total_ml, f, "used"


def predicted_de76(a_clean: np.ndarray, a_residue: np.ndarray, f: float, i0_lin: np.ndarray) -> float:
    """CIE76 dE between the clean colour and the clean colour with residue fraction f mixed in."""
    clean = linear_to_srgb(transmit(a_clean, i0_lin))
    mixed = linear_to_srgb(transmit((1 - f) * a_clean + f * a_residue, i0_lin))
    return float(np.linalg.norm(srgb_to_lab(mixed) - srgb_to_lab(clean)))


# ------------------------------------------------------------------ packet
def _check(records, cursor, roles, what):
    for r in records:
        c = r.cursor if isinstance(r, Measurement) else max(r.previous.cursor, r.first.cursor, r.clean.cursor)
        role_ok = (r.role in roles) if isinstance(r, Measurement) else all(
            m.role in roles for m in (r.previous, r.first, r.clean))
        if c >= cursor or not role_ok:
            raise LeakageError(f"{what} {r.evidence_id}: cursor {c} vs target {cursor}, roles must be {sorted(roles)}")


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _measurement(m: Measurement) -> dict:
    d = asdict(m)
    d["rgb"] = [_r(v, 2) for v in m.rgb]
    d["empty_rgb"] = None if m.empty_rgb is None else [_r(v, 2) for v in m.empty_rgb]
    return d


def build_packet(new: Recipe, previous_programmed: Recipe | None, cursor: int, source: HistorySource,
                 config: Config = Config()) -> dict:
    """Deterministic packet for the vial at `cursor` that starts recipe `new`."""
    roles = config.allowed_roles
    clean = sorted(source.clean_references(new.recipe_id, cursor, roles), key=lambda m: m.evidence_id)
    prev = source.previous_vial(cursor, roles)
    switches = sorted(source.switches(cursor, roles), key=lambda s: s.evidence_id)
    _check(clean, cursor, roles, "clean reference")
    _check([prev] if prev else [], cursor, roles, "previous vial")
    _check(switches, cursor, roles, "switch")
    for m in clean:
        if m.recipe_id != new.recipe_id:
            raise LeakageError(f"clean reference {m.evidence_id} is recipe {m.recipe_id}, not {new.recipe_id}")

    switch_rows, holdups = [], []
    for s in switches:
        h, f, reason = holdup_from_switch(s, config.min_contrast)
        switch_rows.append({"evidence_id": s.evidence_id, "previous": s.previous.evidence_id,
                            "first": s.first.evidence_id, "clean": s.clean.evidence_id,
                            "first_total_ml": _r(s.first_total_ml, 2), "fraction": _r(f),
                            "holdup_ml": _r(h, 2), "status": reason})
        if h is not None:
            holdups.append(h)

    clean_a = [a for a in (absorbance_of(m) for m in clean) if a is not None]
    prev_a = absorbance_of(prev) if prev else None
    reasons = []
    if len(clean_a) < config.min_clean_references:
        reasons.append(f"clean_references {len(clean_a)} < {config.min_clean_references}")
    if prev_a is None:
        reasons.append("no measured previous vial with an empty-vial reference")
    if len(holdups) < config.min_switches:
        reasons.append(f"usable switches {len(holdups)} < {config.min_switches}")

    derived = {"new_total_ml": _r(new.total_ml, 2), "first_dose_ml": _r(new.doses[0].ml, 2) if new.doses else None,
               "same_recipe_as_previous": bool(prev and prev.recipe_id == new.recipe_id)}
    if reasons:
        risk = "insufficient_evidence"
    else:
        a_clean = np.median(clean_a, axis=0)
        i0 = np.median([srgb_to_linear(m.empty_rgb) for m in clean if m.empty_rgb is not None], axis=0)
        hq = [float(np.quantile(holdups, q)) for q in config.quantiles]
        fq = [h / new.total_ml for h in hq]
        derived.update(holdup_ml_quantiles=[_r(h, 2) for h in hq], residue_fraction_quantiles=[_r(f) for f in fq])
        if fq[0] > 1:
            reasons.append("even the low hold-up estimate exceeds the new recipe's volume")
            risk = "likely_impact"
        elif fq[-1] > 1:
            reasons.append("the high hold-up estimate exceeds the new recipe's volume")
            risk = "uncertain"
        else:
            de = [predicted_de76(a_clean, prev_a, f, i0) for f in fq]
            derived["predicted_de76_quantiles"] = [_r(d, 2) for d in de]
            t = config.threshold_de76
            if de[0] > t:
                risk = "likely_impact"
                reasons.append(f"low-quantile predicted dE {de[0]:.2f} > threshold {t:g}")
            elif de[-1] <= t:
                risk = "unlikely"
                reasons.append(f"high-quantile predicted dE {de[-1]:.2f} <= threshold {t:g}")
            else:
                risk = "uncertain"
                reasons.append(f"threshold {t:g} lies inside the predicted dE range")

    images = [m.image_id for m in ([prev] if prev else []) + clean + [s.first for s in switches] if m.image_id]
    packet = {
        "schema_version": SCHEMA_VERSION,
        "config": {"config_version": config.config_version, "threshold_de76": config.threshold_de76,
                   "min_clean_references": config.min_clean_references, "min_switches": config.min_switches,
                   "quantiles": list(config.quantiles), "allowed_roles": sorted(roles)},
        "target": {"cursor": cursor, "new_recipe": asdict(new),
                   "previous_programmed": asdict(previous_programmed) if previous_programmed else None},
        "evidence": {"clean_references": [_measurement(m) for m in clean],
                     "previous_vial": _measurement(prev) if prev else None,
                     "switches": switch_rows},
        "coverage": {"clean_references": len(clean_a), "switches": len(switches), "usable_switches": len(holdups)},
        "derived": derived,
        "decision": {"risk": risk, "reasons": reasons, "decided_by": "deterministic"},
        "images": list(dict.fromkeys(images))[:config.max_images],
    }
    packet["packet_sha256"] = packet_hash(packet)
    return packet


def packet_hash(packet: dict) -> str:
    body = {k: v for k, v in packet.items() if k != "packet_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------ model input and guard
def evidence_ids(packet: dict) -> set[str]:
    ev = packet["evidence"]
    ids = {m["evidence_id"] for m in ev["clean_references"]}
    if ev["previous_vial"]:
        ids.add(ev["previous_vial"]["evidence_id"])
    for s in ev["switches"]:
        ids.update({s["evidence_id"], s["previous"], s["first"], s["clean"]})
    return ids | set(packet["images"])


def render_prompt(packet: dict) -> str:
    """Text for an image-capable model. The images listed in packet['images'] go alongside."""
    return "\n".join([
        "You explain a residue-impact decision that has already been made. Do not change it.",
        "Use only the numbers and evidence IDs in the packet. Cite every claim with its ID in square",
        "brackets, e.g. [E-12]. Do not give process-control advice. If the risk is",
        "insufficient_evidence, say what evidence is missing.",
        f"Decision: {packet['decision']['risk']}.",
        "Packet:",
        json.dumps({k: packet[k] for k in ("target", "evidence", "coverage", "derived", "decision", "images")},
                   sort_keys=True, indent=1),
    ])


def template_explanation(packet: dict) -> str:
    """Fallback explanation built only from packet fields."""
    d, dec, ev = packet["derived"], packet["decision"], packet["evidence"]
    prev = ev["previous_vial"]
    parts = [f"Residue risk for recipe {packet['target']['new_recipe']['recipe_id']}: {dec['risk']}."]
    if prev:
        parts.append(f"Previous vial was recipe {prev['recipe_id']} [{prev['evidence_id']}].")
    if "holdup_ml_quantiles" in d:
        used = [s["evidence_id"] for s in ev["switches"] if s["status"] == "used"]
        parts.append(f"Past switches put the shared-path hold-up at {d['holdup_ml_quantiles'][1]} mL (median) "
                     f"[{', '.join(used)}], a residue share of {d['residue_fraction_quantiles'][1]} of "
                     f"{d['new_total_ml']} mL.")
    if "predicted_de76_quantiles" in d:
        lo, mid, hi = d["predicted_de76_quantiles"]
        refs = ", ".join(m["evidence_id"] for m in ev["clean_references"])
        parts.append(f"Predicted colour shift against the clean reference [{refs}]: dE {lo} to {hi} "
                     f"(median {mid}); threshold {packet['config']['threshold_de76']}.")
    if dec["risk"] == "insufficient_evidence":
        parts.append("Missing: " + "; ".join(dec["reasons"]) + ".")
    return " ".join(parts)


_NUMBER = re.compile(r"(?<![\w-])-?\d+(?:\.\d+)?")


def _packet_numbers(obj, out):
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _packet_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _packet_numbers(v, out)
    elif isinstance(obj, str):
        out.extend(float(x) for x in _NUMBER.findall(obj))
    return out


def check_explanation(text: str, packet: dict, rel_tol: float = 0.01) -> list[str]:
    """Problems with a model explanation; empty means it passes. Rejects unknown or missing
    citations, numbers that are not in the packet, and a stated risk that differs from the decision."""
    problems = []
    cited = set(re.findall(r"\[([^\]]+)\]", text))
    cited_ids = {c.strip() for group in cited for c in group.split(",")}
    known = evidence_ids(packet)
    if not cited_ids:
        problems.append("no evidence cited")
    for c in sorted(cited_ids - known):
        problems.append(f"unknown evidence id: {c}")
    allowed = _packet_numbers(packet, [])
    for raw in _NUMBER.findall(re.sub(r"\[[^\]]*\]", "", text)):
        x = float(raw)
        if not any(abs(x - a) <= max(rel_tol * abs(a), 0.005) for a in allowed):
            problems.append(f"number not in packet: {raw}")
    for risk in ("likely_impact", "unlikely", "uncertain", "insufficient_evidence"):
        if risk in text and risk != packet["decision"]["risk"]:
            problems.append(f"states risk {risk}, decision is {packet['decision']['risk']}")
    return problems


# ------------------------------------------------------------------ reference source for tests and demos
class InMemoryHistory:
    """A HistorySource over lists, with the same strictly-earlier and role filtering the store
    must provide. For tests and demos; production uses an adapter over the INN-62 store."""

    def __init__(self, measurements: list[Measurement], switches: list[Switch], clean_ids: set[str]):
        self.measurements, self._switches, self.clean_ids = measurements, switches, clean_ids

    def clean_references(self, recipe_id, before, allowed_roles):
        return [m for m in self.measurements if m.evidence_id in self.clean_ids and m.recipe_id == recipe_id
                and m.cursor < before and m.role in allowed_roles]

    def previous_vial(self, before, allowed_roles):
        earlier = [m for m in self.measurements if m.cursor < before and m.role in allowed_roles]
        return max(earlier, key=lambda m: m.cursor, default=None)

    def switches(self, before, allowed_roles):
        return [s for s in self._switches
                if max(s.previous.cursor, s.first.cursor, s.clean.cursor) < before
                and all(m.role in allowed_roles for m in (s.previous, s.first, s.clean))]
