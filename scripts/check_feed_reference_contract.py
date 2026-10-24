"""Emit a deterministic synthetic-only audit of the feed-reference contract."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.feed_reference import (  # noqa: E402
    CONTRACT_VERSION,
    FeedDerivedReferenceCandidate,
    FeedEvent,
    compare_feed_reference,
    measure_feed_regions,
    summarize_reference_coverage,
    validate_event_ledger,
)


def region(rgb, count=16):
    return np.repeat(np.asarray([rgb], dtype=float), count, axis=0)


def split_region(low, high):
    return np.vstack([region(low, 8), region(high, 8)])


def event(position, *, suffix=None, **changes):
    suffix = str(position) if suffix is None else suffix
    values = dict(
        record_id=f"event-{suffix}",
        schedule_record_id=f"schedule-{suffix}",
        sequence_id="synthetic-sequence-1",
        transition_id=f"transition-{suffix}",
        recipe_id="synthetic-recipe-7",
        recipe_occurrence_index=position,
        vial_ordinal=1,
        scheduled_position=position,
        observed_position=position,
        frame_ids=(f"frame-{suffix}",),
        window_id=f"window-{suffix}",
        timing_source="constructed-metadata-fixture",
        timing_uncertainty_s=0.01,
        timing_tolerance_s=0.05,
        mapping_status="confirmed",
        phase="final_mixture",
        appearance_domain_id="synthetic-continuous-run-1",
    )
    values.update(changes)
    return FeedEvent(**values)


def measurement(liquid, patch, suffix):
    return measure_feed_regions(
        region(liquid),
        region(patch),
        record_id=f"measurement-{suffix}",
        frame_id=f"frame-{suffix}",
        liquid_roi_id="synthetic-liquid-roi-v1",
        patch_id="synthetic-fixed-patch-v1",
    )


def candidate(reference_event, reference_measurement):
    return FeedDerivedReferenceCandidate(
        record_id=f"candidate-{reference_event.record_id}",
        event_record_id=reference_event.record_id,
        measurement_record_id=reference_measurement.record_id,
        selection_rule_id="externally-supplied-synthetic-rule-v1",
        source_frame_ids=(reference_measurement.frame_id,),
    )


def compare(reference_event, current_event, reference_measurement, current_measurement):
    return compare_feed_reference(
        record_id=f"comparison-{current_event.record_id}",
        current_event=current_event,
        current_measurement=current_measurement,
        event_ledger=(reference_event, current_event),
        reference_candidate=candidate(reference_event, reference_measurement),
        reference_event=reference_event,
        reference_measurement=reference_measurement,
    )


def build_report():
    cases = []

    def add(case_id, expectation, passed, observed):
        if isinstance(observed, np.generic):
            observed = observed.item()
        cases.append(
            {
                "id": case_id,
                "expectation": expectation,
                "observed": observed,
                "passed": bool(passed),
            }
        )

    liquid = np.array([0.2, 0.3, 0.4])
    patch = np.array([0.5, 0.5, 0.5])
    base = measurement(liquid, patch, "base")
    scalar = measurement(liquid * 1.5, patch * 1.5, "scalar")
    channel_gain = np.array([1.2, 0.8, 0.6])
    channel = measurement(liquid * channel_gain, patch * channel_gain, "channel")

    def descriptor_distance(left, right):
        return float(np.linalg.norm(np.asarray(left.log_ratio) - np.asarray(right.log_ratio)))

    normalized_expected = np.array([2 / 9, 3 / 9, 4 / 9])
    normalized_error = float(
        np.max(np.abs(np.asarray(base.normalized_rgb) - normalized_expected))
    )
    add(
        "normalized_rgb_no_patch_baseline",
        "liquid median divided by its channel sum",
        normalized_error < 1e-12,
        {"max_abs_error": normalized_error},
    )
    scalar_delta = descriptor_distance(base, scalar)
    add(
        "common_scalar_gain",
        "descriptor delta below 2e-6 away from epsilon and clipping",
        scalar_delta < 2e-6,
        {"descriptor_distance": scalar_delta},
    )
    channel_delta = descriptor_distance(base, channel)
    add(
        "common_channel_gain",
        "descriptor delta below 2e-6 away from epsilon and clipping",
        channel_delta < 2e-6,
        {"descriptor_distance": channel_delta},
    )

    liquid_change = measurement([0.2, 0.3, 0.2], patch, "liquid-change")
    liquid_change_distance = descriptor_distance(base, liquid_change)
    add(
        "liquid_only_change",
        "descriptor changes without a defect interpretation",
        liquid_change_distance > 0.6,
        {"descriptor_distance": liquid_change_distance},
    )
    independent_gain = measurement(liquid * 1.2, patch, "independent-gain")
    independent_distance = descriptor_distance(base, independent_gain)
    add(
        "independent_region_gain",
        "conditional invariance fails visibly",
        independent_distance > 0.3,
        {"descriptor_distance": independent_distance},
    )
    gamma = measurement(liquid**1.8, patch**1.8, "gamma")
    gamma_distance = descriptor_distance(base, gamma)
    add(
        "nonlinear_gamma",
        "nonlinear processing remains visible",
        gamma_distance > 0.2,
        {"descriptor_distance": gamma_distance},
    )
    quantized = measurement(
        np.round(liquid * 255) / 255,
        np.round(patch * 255) / 255,
        "quantized",
    )
    quantized_distance = descriptor_distance(base, quantized)
    add(
        "eight_bit_quantization",
        "small nonzero descriptor movement is reported",
        0 < quantized_distance < 0.02,
        {"descriptor_distance": quantized_distance},
    )
    focus_mixed = measure_feed_regions(
        split_region(liquid, patch),
        region(patch),
        record_id="measurement-focus-mixed",
        frame_id="frame-focus-mixed",
        liquid_roi_id="synthetic-liquid-roi-v1",
        patch_id="synthetic-fixed-patch-v1",
    )
    focus_distance = descriptor_distance(base, focus_mixed)
    add(
        "focus_or_edge_mixing",
        "region mixing changes the descriptor",
        focus_distance > 0.1,
        {"descriptor_distance": focus_distance},
    )
    dirty_patch = measure_feed_regions(
        region(liquid),
        split_region([0.1, 0.1, 0.1], patch),
        record_id="measurement-dirty-patch",
        frame_id="frame-dirty-patch",
        liquid_roi_id="synthetic-liquid-roi-v1",
        patch_id="synthetic-fixed-patch-v1",
    )
    add(
        "patch_occlusion_or_dirt",
        "spatially unstable patch abstains",
        dirty_patch.log_ratio is None and "patch_spatially_unstable" in dirty_patch.reason_codes,
        {"validity": dirty_patch.validity, "reason_codes": dirty_patch.reason_codes},
    )
    saturated = measurement([0.2, 0.3, 1.0], patch, "saturated")
    add(
        "saturation",
        "clipped region abstains",
        saturated.log_ratio is None and "liquid_saturated" in saturated.reason_codes,
        {"validity": saturated.validity, "reason_codes": saturated.reason_codes},
    )
    underexposed = measurement([1e-5, 0.3, 0.4], patch, "underexposed")
    add(
        "near_black",
        "underexposed channel abstains",
        underexposed.log_ratio is None and "liquid_underexposed" in underexposed.reason_codes,
        {"validity": underexposed.validity, "reason_codes": underexposed.reason_codes},
    )
    nonfinite_pixels = region(liquid)
    nonfinite_pixels[0, 0] = np.nan
    nonfinite = measure_feed_regions(
        nonfinite_pixels,
        region(patch),
        record_id="measurement-nonfinite",
        frame_id="frame-nonfinite",
        liquid_roi_id="synthetic-liquid-roi-v1",
        patch_id="synthetic-fixed-patch-v1",
    )
    add(
        "nonfinite_input",
        "nonfinite region is invalid and has no descriptor",
        nonfinite.log_ratio is None and nonfinite.validity == "invalid",
        {"validity": nonfinite.validity, "reason_codes": nonfinite.reason_codes},
    )

    reference_event = event(10)
    current_event = event(11)
    shifted_reference = measurement([0.35, 0.3, 0.4], patch, "10")
    later_unshifted = measurement(liquid, patch, "11")
    later_matching_shift = measurement([0.35, 0.3, 0.4], patch, "11")
    false_disagreement = compare(
        reference_event,
        current_event,
        shifted_reference,
        later_unshifted,
    )
    masked_shift = compare(
        reference_event,
        current_event,
        shifted_reference,
        later_matching_shift,
    )
    add(
        "contaminated_reference_false_disagreement",
        "shifted reference disagrees with later unshifted appearance",
        false_disagreement.appearance_distance > 0.5,
        {"appearance_distance": false_disagreement.appearance_distance},
    )
    add(
        "contaminated_reference_masking",
        "matching shifted appearances can have zero distance",
        abs(masked_shift.appearance_distance) < 1e-12,
        {"appearance_distance": masked_shift.appearance_distance},
    )

    first_event = event(20, recipe_occurrence_index=1)
    first_measurement = measurement(liquid, patch, "20")
    first = compare_feed_reference(
        record_id="comparison-first",
        current_event=first_event,
        current_measurement=first_measurement,
        event_ledger=(first_event,),
    )
    add(
        "first_occurrence",
        "no prior reference has its own unavailable bucket",
        first.coverage_bucket == "first_occurrence" and not first.comparison_available,
        {"coverage_bucket": first.coverage_bucket},
    )
    repeat_without_reference_event = event(21, recipe_occurrence_index=2)
    repeat_without_reference = compare_feed_reference(
        record_id="comparison-no-prior",
        current_event=repeat_without_reference_event,
        current_measurement=measurement(liquid, patch, "21"),
        event_ledger=(repeat_without_reference_event,),
    )
    add(
        "repeat_without_reference",
        "missing prior is not counted as a comparison",
        repeat_without_reference.coverage_bucket == "reference_unavailable"
        and not repeat_without_reference.comparison_available,
        {"coverage_bucket": repeat_without_reference.coverage_bucket},
    )

    valid_reference = measurement(liquid, patch, "10")
    valid_current = measurement([0.2, 0.3, 0.5], patch, "11")
    ambiguous_event = replace(current_event, mapping_status="ambiguous")
    ambiguous = compare_feed_reference(
        record_id="comparison-ambiguous",
        current_event=ambiguous_event,
        current_measurement=valid_current,
        event_ledger=(reference_event, ambiguous_event),
        reference_candidate=candidate(reference_event, valid_reference),
        reference_event=reference_event,
        reference_measurement=valid_reference,
    )
    add(
        "ambiguous_boundary",
        "ambiguous mapping has no numeric comparison",
        not ambiguous.comparison_available and "mapping_not_confirmed" in ambiguous.reason_codes,
        {"reason_codes": ambiguous.reason_codes},
    )
    off_by_one = validate_event_ledger(
        (reference_event, replace(current_event, observed_position=12))
    )
    add(
        "off_by_one_mapping",
        "scheduled/observed mismatch invalidates the ledger",
        "schedule_observation_mismatch" in off_by_one.reason_codes,
        {"reason_codes": off_by_one.reason_codes},
    )
    missing_vial = validate_event_ledger((reference_event, event(12)))
    add(
        "missing_vial_mapping",
        "position gap invalidates the ledger",
        "schedule_gap" in missing_vial.reason_codes,
        {"reason_codes": missing_vial.reason_codes},
    )
    extra_vial = validate_event_ledger(
        (reference_event, replace(current_event, scheduled_position=10))
    )
    add(
        "extra_or_duplicate_vial_mapping",
        "duplicate/nonmonotone schedule position invalidates the ledger",
        "schedule_nonmonotone" in extra_vial.reason_codes,
        {"reason_codes": extra_vial.reason_codes},
    )
    clock_drift = validate_event_ledger(
        (replace(reference_event, timing_uncertainty_s=0.06, timing_tolerance_s=0.05),)
    )
    add(
        "clock_drift_beyond_tolerance",
        "excess timing uncertainty blocks comparison",
        clock_drift.validity == "uncertain"
        and "timing_uncertainty_exceeds_tolerance" in clock_drift.reason_codes,
        {"validity": clock_drift.validity, "reason_codes": clock_drift.reason_codes},
    )
    stale_current_event = replace(current_event, appearance_domain_id="synthetic-run-2")
    stale = compare(
        reference_event,
        stale_current_event,
        valid_reference,
        valid_current,
    )
    add(
        "same_recipe_changed_appearance_domain",
        "changed/unknown tank appearance is stale, not a carryover label",
        not stale.comparison_available and "reference_stale" in stale.reason_codes,
        {"reason_codes": stale.reason_codes},
    )
    invalid_phase_event = replace(current_event, phase="pre_final_drop")
    invalid_phase = compare_feed_reference(
        record_id="comparison-invalid-phase",
        current_event=invalid_phase_event,
        current_measurement=valid_current,
        event_ledger=(reference_event, invalid_phase_event),
        reference_candidate=candidate(reference_event, valid_reference),
        reference_event=reference_event,
        reference_measurement=valid_reference,
    )
    add(
        "invalid_frame_phase",
        "pre-final-drop frame has no numeric comparison",
        not invalid_phase.comparison_available and "invalid_phase" in invalid_phase.reason_codes,
        {"reason_codes": invalid_phase.reason_codes},
    )
    shared_reference_event = replace(reference_event, frame_ids=("shared-frame",))
    shared_current_event = replace(current_event, frame_ids=("shared-frame",))
    shared_reference_measurement = replace(valid_reference, frame_id="shared-frame")
    shared_current_measurement = replace(valid_current, frame_id="shared-frame")
    overlap = compare(
        shared_reference_event,
        shared_current_event,
        shared_reference_measurement,
        shared_current_measurement,
    )
    add(
        "reference_current_frame_overlap",
        "same source frame cannot compare against itself",
        not overlap.comparison_available and "duplicate_frame_id" in overlap.reason_codes,
        {"reason_codes": overlap.reason_codes},
    )
    invalid_current = compare(
        reference_event,
        current_event,
        valid_reference,
        replace(saturated, frame_id="frame-11", record_id="measurement-11-invalid"),
    )
    add(
        "invalid_current_measurement",
        "quality-invalid current evidence has no comparison",
        invalid_current.coverage_bucket == "current_invalid"
        and not invalid_current.comparison_available,
        {"coverage_bucket": invalid_current.coverage_bucket},
    )
    compared = compare(
        reference_event,
        current_event,
        valid_reference,
        valid_current,
    )
    add(
        "valid_structural_pair_is_still_unverified",
        "distance is available but cleanliness remains uncertain",
        compared.comparison_available
        and compared.validity == "uncertain"
        and compared.reason_codes == ("reference_cleanliness_unverified",),
        {"validity": compared.validity, "reason_codes": compared.reason_codes},
    )

    coverage_first_event = event(
        100,
        suffix="coverage-first",
        recipe_occurrence_index=1,
    )
    coverage_first = compare_feed_reference(
        record_id="coverage-first",
        current_event=coverage_first_event,
        current_measurement=measurement(liquid, patch, "coverage-first"),
        event_ledger=(coverage_first_event,),
    )
    coverage_unavailable_event = event(
        101,
        suffix="coverage-unavailable",
        recipe_occurrence_index=2,
    )
    coverage_unavailable = compare_feed_reference(
        record_id="coverage-unavailable",
        current_event=coverage_unavailable_event,
        current_measurement=measurement(liquid, patch, "coverage-unavailable"),
        event_ledger=(coverage_unavailable_event,),
    )

    coverage_stale_reference_event = event(110, suffix="coverage-stale-reference")
    coverage_stale_current_event = event(
        111,
        suffix="coverage-stale-current",
        appearance_domain_id="synthetic-run-2",
    )
    coverage_stale = compare(
        coverage_stale_reference_event,
        coverage_stale_current_event,
        measurement(liquid, patch, "coverage-stale-reference"),
        measurement(liquid, patch, "coverage-stale-current"),
    )

    coverage_invalid_reference_event = event(120, suffix="coverage-invalid-reference")
    coverage_invalid_current_event = event(121, suffix="coverage-invalid-current")
    coverage_current_invalid = compare(
        coverage_invalid_reference_event,
        coverage_invalid_current_event,
        measurement(liquid, patch, "coverage-invalid-reference"),
        measurement([0.2, 0.3, 1.0], patch, "coverage-invalid-current"),
    )

    coverage_mapping_reference_event = event(130, suffix="coverage-mapping-reference")
    coverage_mapping_current_event = event(
        131,
        suffix="coverage-mapping-current",
        mapping_status="ambiguous",
    )
    coverage_mapping_invalid = compare(
        coverage_mapping_reference_event,
        coverage_mapping_current_event,
        measurement(liquid, patch, "coverage-mapping-reference"),
        measurement(liquid, patch, "coverage-mapping-current"),
    )

    coverage_compared_reference_event = event(140, suffix="coverage-compared-reference")
    coverage_compared_current_event = event(141, suffix="coverage-compared-current")
    coverage_compared = compare(
        coverage_compared_reference_event,
        coverage_compared_current_event,
        measurement(liquid, patch, "coverage-compared-reference"),
        measurement([0.2, 0.3, 0.5], patch, "coverage-compared-current"),
    )
    coverage = summarize_reference_coverage(
        (
            coverage_first,
            coverage_unavailable,
            coverage_stale,
            coverage_current_invalid,
            coverage_mapping_invalid,
            coverage_compared,
        )
    )
    pass_count = sum(case["passed"] for case in cases)
    return {
        "schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "input_kind": "constructed_arrays_and_metadata_only",
        "case_count": len(cases),
        "pass_count": pass_count,
        "fail_count": len(cases) - pass_count,
        "cases": cases,
        "coverage": asdict(coverage),
        "limits": [
            "synthetic contract behavior only",
            "no video or image input",
            "no clean-colour, physical, camera-transfer, detector, carryover, or cause evidence",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    if report["fail_count"]:
        raise SystemExit(f"{report['fail_count']} synthetic contract checks failed")


if __name__ == "__main__":
    main()
