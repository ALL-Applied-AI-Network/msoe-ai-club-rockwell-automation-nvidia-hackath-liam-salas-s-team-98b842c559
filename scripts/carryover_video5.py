"""Run the video-5 carryover decision end to end and write records.json and results.md.

    python scripts/carryover_video5.py [--video PATH] [--out DIR]

Inputs (labels, not model outputs): data/video5_vial_recipes.csv (the PLC's programmed recipe),
data/video5_transitions.csv, data/real_check_windows.csv; data/video5_vial_colors.csv is only
compared against. Every number in results.md is computed here.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix.carryover.measure import measure_vials, cie76
from fluidmix.carryover.decide import decide, flagged_count
from fluidmix.carryover.explain import explain

DEFAULT_VIDEO = Path("/home/hackathon5/liamreporoot/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559"
                     "/external/external/csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_5.mp4")


def read_csv(name):
    with (ROOT / "data" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def maybe_int(text):
    return int(text) if text not in ("", "unknown", None) else None


def load_records():
    recipes = {int(r["vial"]): r for r in read_csv("video5_vial_recipes.csv")}
    transitions = {int(r["vial"]): r for r in read_csv("video5_transitions.csv")}
    windows = {int(r["vial"]): r for r in read_csv("real_check_windows.csv")}
    records = {}
    for vial in sorted(recipes):
        rec, tr, w = recipes[vial], transitions[vial], windows[vial]
        eligible = w["eligible"] == "1"
        records[vial] = {
            "vial": vial, "recipe_id": int(rec["recipe_id"]),
            "prev_recipe_id": maybe_int(tr["prev_recipe_id"]),
            "recipe_changed": maybe_int(tr["recipe_changed"]),
            "position_in_run": int(tr["position_in_run"]),
            "fill_start_frame": int(rec["start_frame"]), "fill_end_frame": int(rec["end_frame"]),
            "window": [int(w["start_frame"]), int(w["end_frame"])] if eligible else None,
            "eligible": eligible, "exclusion_reason": w["exclusion_reason"] or None,
            "measured_lab": None, "measurement_noise_dE": None,
            "_window_row": {"vial": vial, "eligible": eligible,
                            "start_frame": int(w["start_frame"]), "end_frame": int(w["end_frame"])},
        }
    return records


def group_of(r):
    if r["recipe_changed"] == 1 and r["position_in_run"] == 1:
        return "position 1 after a recipe change"
    if r["position_in_run"] == 2:
        return "position 2"
    return "position 3 or later"


def fmt_lab(v):
    return "null" if v is None else "[" + ", ".join(f"{x:.1f}" for x in v) + "]"


def write_results(records, tau, out):
    decided = [r for r in records.values() if r["decision"] in ("keep", "discard")]
    lines = ["# Carryover decision, video 5 (measured)", "",
             "All numbers below were computed by scripts/carryover_video5.py from the vial_fill video 5 and",
             "the label CSVs in data/. The recipe schedule is the PLC's programmed recipe (a label stand-in",
             "for the control-system signal), not a model output. This is an in-sample demonstration of a",
             "fixed rule (src/fluidmix/carryover/decide.py), not a validation: the rule was not changed after",
             "the first run, but no independent freeze record exists and its tau floor coincides with an",
             "older video-5 statistic (section (f)).", "",
             f"tau = max(2.3, 3 * median noise) = **{tau:.2f}** "
             f"(median frame-to-frame noise over eligible vials = "
             f"{np.median([r['measurement_noise_dE'] for r in records.values() if r['measured_lab'] is not None]):.3f}).",
             "", "## (a) Per-vial results", "",
             "| vial | recipe | prev | pos | measured Lab | noise dE | reference Lab | dE vs ref | tau | decision | known-colour check |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records.values():
        d = "null" if r["dE_vs_reference"] is None else f"{r['dE_vs_reference']:.1f}"
        n = "null" if r["measurement_noise_dE"] is None else f"{r['measurement_noise_dE']:.2f}"
        lines.append(f"| {r['vial']} | {r['recipe_id']} | {r['prev_recipe_id'] if r['prev_recipe_id'] is not None else 'unknown'} "
                     f"| {r['position_in_run']} | {fmt_lab(r['measured_lab'])} | {n} | {fmt_lab(r['reference_lab'])} "
                     f"| {d} | {r['tau']:.2f} | {r['decision']} | {r['known_colour_check'] or 'null'} |")
    lines += ["", "## (b) Agreement with the process assumption (not scoring)", "",
              "Video 5 has no defect labels (docs/evaluation.md). The grouping below rests on an assumption from",
              "the dispensing-order argument, not on a label: position 1 after a recipe change is assumed to carry",
              "residue of the previous recipe, and positions 2 and 3+ (same recipe again) are assumed clean.",
              "The reference colour of every vial is built from the vials assumed clean, so a position-1 vial is",
              "flagged whenever it differs from its successors for any reason. The position-1 row therefore",
              "shows agreement with the assumption, not detection accuracy, and the position-2 and 3+ rows are",
              "the matching false-alarm rate under the same assumption.", "",
              "| group | discard (fixed rule) / total | known-colour check fail / total |", "|---|---|---|"]
    for g in ("position 1 after a recipe change", "position 2", "position 3 or later"):
        rows = [r for r in decided if group_of(r) == g]
        lines.append(f"| {g} | {sum(r['decision'] == 'discard' for r in rows)} / {len(rows)} "
                     f"| {sum(r['known_colour_check'] == 'fail' for r in rows)} / {len(rows)} |")
    skipped = [f"vial {r['vial']} ({r['reason']})" for r in records.values() if r["decision"] == "insufficient_evidence"]
    lines += ["", f"Not decided (insufficient_evidence): {'; '.join(skipped) or 'none'}.", "",
              "## (c) Sensitivity (diagnostic only, NOT used for selection; tau stays fixed)", "",
              "Discards among decided vials at tau x 0.5 / x 1 / x 2: "
              + " / ".join(f"{flagged_count(records, s)}" for s in (0.5, 1.0, 2.0))
              + f" of {len(decided)} (tau = {tau * 0.5:.2f} / {tau:.2f} / {tau * 2:.2f}).", "",
              "## (d) Sanity check against data/video5_vial_colors.csv (older rough method)", "",
              "The older method sampled three single frames at fixed offsets after the fill end with ffmpeg; this",
              "run uses every frame of the measured final-mixture window with cv2. Same crop and pixel rule.", "",
              "| vial | measured Lab (this run) | old Lab | CIE76 between them |", "|---|---|---|---|"]
    old = {int(r["vial"]): [float(r["L"]), float(r["a"]), float(r["b_lab"])] for r in read_csv("video5_vial_colors.csv")}
    diffs = []
    for r in records.values():
        if r["measured_lab"] is None:
            continue
        d = cie76(r["measured_lab"], old[r["vial"]])
        diffs.append(d)
        lines.append(f"| {r['vial']} | {fmt_lab(r['measured_lab'])} | {fmt_lab(old[r['vial']])} | {d:.1f} |")
    departure = {int(r["vial"]): int(round(float(r["departure_after_end_s"]) * 30)) for r in read_csv("real_check_windows.csv")}
    measured_vials = [r["vial"] for r in records.values() if r["measured_lab"] is not None]
    gap = dict(zip(measured_vials, diffs))
    last_sample = 45  # the old method's last single-frame sample, frames after the fill end
    big = [v for v in measured_vials if gap[v] > 2.0]
    big_late = [v for v in big if departure[v] <= last_sample]
    big_not_late = [v for v in big if departure[v] > last_sample]
    small_late = [v for v in measured_vials if gap[v] <= 2.0 and departure[v] <= last_sample]
    desc = lambda vs: ", ".join(f"vial {v} (gap {gap[v]:.1f}, cart moves {departure[v]} frames after the fill end)" for v in vs) or "none"
    lines += ["", f"Median CIE76 between the two methods: {np.median(diffs):.1f}; max {max(diffs):.1f} "
              f"(vial {measured_vials[int(np.argmax(diffs))]}).",
              f"Gaps above 2.0: {desc(big)}.",
              f"The old method sampled single frames 15, 30 and {last_sample} after the fill end. For {desc(big_late)} the cart",
              f"started moving at or before the last sample, so a moving cart is a plausible cause of the gap (a hypothesis,",
              f"not shown). For {desc(big_not_late)} the cart moved well after the last sample, so the gap is unexplained.",
              f"The cart also moved at or before the last sample for {desc(small_late)}, whose gaps are small, so",
              "cart movement alone does not account for the gap sizes.", "",
              "## Flaws in the fixed rule seen after the run (reported, not corrected)", "",
              "- The noise estimator collapsed. The per-frame colour is the median of integer pixels, so most",
              "  consecutive-frame steps are exactly 0: zero-step fraction per vial, median over eligible vials = "
              f"{np.median([r['diagnostics']['zero_step_fraction'] for r in records.values() if r['measured_lab'] is not None]):.2f}",
              "  (within-window peak-to-peak L range, median over vials = "
              f"{np.median([r['diagnostics']['window_lab_ptp'][0] for r in records.values() if r['measured_lab'] is not None]):.2f}).",
              "  tau therefore fell to the floor 2.3, which measures frame-to-frame quantisation, not how much two",
              "  settled vials of the same recipe differ from each other.",
              "- With three vials per recipe, the leave-one-out reference of a position-2 vial is just the",
              "  position-3 vial and vice versa, so each pair gets the same dE and is flagged or kept together.",
              "  Settled same-recipe pairs differ by " + ", ".join(
                  f"{r['dE_vs_reference']:.1f}" for r in records.values() if r['position_in_run'] == 2 and r['dE_vs_reference'] is not None)
              + " dE (positions 2 vs 3), so a threshold near 2.3 flags half of them; the position-1 vials differ by "
              + ", ".join(f"{r['dE_vs_reference']:.1f}" for r in records.values()
                          if r['recipe_changed'] == 1 and r['dE_vs_reference'] is not None) + ".",
              "- Proposal (not applied): base tau on the between-vial spread of settled same-recipe vials from a",
              "  separate calibration recording, not on frame-to-frame noise, and fix it before scoring.", "",
              "## (e) Limits", "",
              "The recipe schedule and fill times come from hand labels, not from a live PLC signal, and every vial",
              "is from one recording (video 5) with one camera, so this is an in-sample check of the rule, not a",
              "validation on held-out data. The rule and tau formula were not changed after the first run, but",
              "they were not frozen independently of video 5 (section (f)), and the measurement region (crop and",
              "top-0.7 % saturation rule) was chosen by eye earlier in the project on this same video.",
              "References are leave-one-out medians of at most two peers per recipe, so a",
              "single odd peer moves the reference; the known-colour check uses the same tau and references and is",
              "only a baseline. CIE76 on camera sRGB is not a calibrated colour difference.", ""]
    lines += provenance_section(records, tau)
    (out / "results.md").write_text("\n".join(lines))


def provenance_section(records, tau):
    """Section (f): where tau comes from, with the coincidence the reviewers found computed here."""
    old = {int(r["vial"]): float(r["deltaE_vs_last_vial_of_recipe"]) for r in read_csv("video5_vial_colors.csv")}
    old_pos2 = sorted(old[v] for v, r in records.items() if r["position_in_run"] == 2)
    old_pos1 = sorted(old[v] for v, r in records.items() if r["recipe_changed"] == 1 and r["position_in_run"] == 1)
    noises = [r["measurement_noise_dE"] for r in records.values() if r["measured_lab"] is not None]
    return ["## (f) Provenance of the rule and tau (review response, 2026-11-01)", "",
            "- tau = max(2.3, 3 * median noise) was written in decide.py before scripts/carryover_video5.py was first",
            "  run (file mtimes 2026-11-01 00:37 vs records.json 00:39) and has not been changed since. That is the",
            "  only ordering evidence: the rule was not committed or recorded in a dated freeze note before the run,",
            "  so the ordering cannot be audited from the repository (docs/feasibility-protocol.md asks for a",
            "  timestamped freeze). Earlier wording that the rule was 'frozen before any measurement was seen' and",
            "  that 'nothing was tuned on video 5' has been withdrawn.",
            f"- The noise term is computed in-sample: 3 * median(measurement_noise_dE) over the {len(noises)} scored vials",
            f"  = {3 * float(np.median(noises)):.3f}, so the floor alone decided every vial (tau = {tau:.2f}).",
            "- The floor 2.3 and multiplier 3 were chosen by hand. 2.3 is the value commonly quoted as the CIE76",
            "  just-noticeable difference, but it is also exactly the median position-2 dE of these same vials under",
            "  the older method (data/video5_vial_colors.csv, column deltaE_vs_last_vial_of_recipe; docs/plan.md",
            "  committed 2026-10-31): position-2 values " + ", ".join(f"{x:.1f}" for x in old_pos2)
            + f", median {np.median(old_pos2):.1f}; position-1-after-change values "
            + ", ".join(f"{x:.1f}" for x in old_pos1) + f", median {np.median(old_pos1):.1f}.",
            "  With no freeze record predating that file, the floor cannot be claimed independent of video 5.",
            "  This run is therefore an in-sample demonstration whose threshold coincides with a prior video-5",
            "  statistic. The rule was deliberately left unchanged here rather than re-tuned on the outcome;",
            "  any future tau must be set on twin val per docs/evaluation.md and frozen in a dated, committed note.", ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/carryover-video5")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_records()
    measured = measure_vials(args.video, [r.pop("_window_row") for r in records.values()])
    for vial, m in measured.items():
        records[vial]["measured_lab"] = m["measured_lab"]
        records[vial]["measurement_noise_dE"] = m["measurement_noise_dE"]
        records[vial]["diagnostics"] = m["diagnostics"]
    tau = decide(records)
    for r in records.values():
        r["explanation"] = explain(r)
        r["evidence"] = {"fill_frames": [r["fill_start_frame"], r["fill_end_frame"]],
                         "window_frames": measured.get(r["vial"], {}).get("window_frames", [])}
    (args.out / "records.json").write_text(json.dumps(
        {"video_id": 5, "video": str(args.video), "tau": round(tau, 3), "vials": list(records.values())},
        indent=1, allow_nan=False))
    write_results(records, tau, args.out)
    for r in records.values():
        print(r["vial"], r["recipe_id"], r["decision"], r["dE_vs_reference"], r["known_colour_check"])
    print("tau", round(tau, 3), "->", args.out)


if __name__ == "__main__":
    main()
