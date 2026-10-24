import os
import sys
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidmix.feed_reference import (
    DEFAULT_MEASUREMENT_CONFIG,
    FeedDerivedReferenceCandidate,
    FeedEvent,
    FeedRegionMeasurement,
    compare_feed_reference,
    measure_feed_regions,
    summarize_reference_coverage,
    validate_event_ledger,
)


ROOT = Path(__file__).resolve().parents[1]


def _region(rgb, n=16):
    return np.repeat(np.asarray([rgb], dtype=float), n, axis=0)


def _event(position, *, suffix=None, **changes):
    suffix = str(position) if suffix is None else suffix
    values = dict(
        record_id=f"event-{suffix}",
        schedule_record_id=f"schedule-{position}",
        sequence_id="sequence-1",
        transition_id=f"transition-{position}",
        recipe_id="recipe-7",
        recipe_occurrence_index=position,
        vial_ordinal=1,
        scheduled_position=position,
        observed_position=position,
        frame_ids=(f"frame-{suffix}",),
        window_id=f"window-{suffix}",
        timing_source="synthetic-fixture",
        timing_uncertainty_s=0.01,
        timing_tolerance_s=0.05,
        mapping_status="confirmed",
        phase="final_mixture",
        appearance_domain_id="continuous-run-1",
    )
    values.update(changes)
    return FeedEvent(**values)


def test_measurement_uses_declared_log_ratio_and_retains_no_patch_baseline():
    result = measure_feed_regions(
        _region([0.2, 0.4, 0.6]),
        _region([0.5, 0.5, 0.5]),
        record_id="measurement-1",
        frame_id="frame-1",
        liquid_roi_id="vial-liquid-v1",
        patch_id="fixed-patch-v1",
    )

    expected_log_ratio = np.log(
        (np.array([0.2, 0.4, 0.6]) + 1e-6)
        / (np.array([0.5, 0.5, 0.5]) + 1e-6)
    )
    expected_normalized_rgb = np.array([1 / 6, 1 / 3, 1 / 2])
    assert result.validity == "valid"
    assert result.reason_codes == ()
    assert np.allclose(result.log_ratio, expected_log_ratio)
    assert np.allclose(result.normalized_rgb, expected_normalized_rgb)
    assert result.liquid_median == (0.2, 0.4, 0.6)
    assert result.patch_median == (0.5, 0.5, 0.5)
    assert result.liquid_area == result.patch_area == 16


def test_nonfinite_region_is_invalid_without_descriptor():
    liquid = _region([0.2, 0.4, 0.6])
    liquid[3, 1] = np.nan
    result = measure_feed_regions(
        liquid,
        _region([0.5, 0.5, 0.5]),
        record_id="measurement-nonfinite",
        frame_id="frame-2",
        liquid_roi_id="vial-liquid-v1",
        patch_id="fixed-patch-v1",
    )

    assert result.validity == "invalid"
    assert result.reason_codes == ("nonfinite_region",)
    assert result.log_ratio is None
    assert result.normalized_rgb is None


@pytest.mark.parametrize(
    ("liquid", "patch", "reason"),
    [
        (0.2, _region([0.5, 0.5, 0.5]), "invalid_region_shape"),
        (np.full((16, 4), 0.4), _region([0.5, 0.5, 0.5]), "invalid_region_shape"),
        (_region([0.2, 0.4, 0.6], 15), _region([0.5, 0.5, 0.5]), "insufficient_region_area"),
        (_region([0.2, 0.4, 1.1]), _region([0.5, 0.5, 0.5]), "region_out_of_range"),
        (np.ma.array(_region([0.2, 0.4, 0.6]), mask=False), _region([0.5, 0.5, 0.5]), "masked_region_unsupported"),
        ([['not-a-number', 0.4, 0.6]] * 16, _region([0.5, 0.5, 0.5]), "invalid_region_type"),
    ],
)
def test_structurally_invalid_regions_do_not_produce_descriptors(liquid, patch, reason):
    result = measure_feed_regions(
        liquid,
        patch,
        record_id=f"measurement-{reason}",
        frame_id="frame-invalid",
        liquid_roi_id="vial-liquid-v1",
        patch_id="fixed-patch-v1",
    )

    assert result.validity == "invalid"
    assert result.reason_codes == (reason,)
    assert result.log_ratio is None


@pytest.mark.parametrize(
    ("liquid", "metadata", "config", "reason"),
    [
        (np.full((16, 3), True, dtype=bool), {}, DEFAULT_MEASUREMENT_CONFIG, "invalid_region_type"),
        ([['0.2', '0.4', '0.6']] * 16, {}, DEFAULT_MEASUREMENT_CONFIG, "invalid_region_type"),
        (
            _region([0.2, 0.4, 0.6]),
            {"record_id": ""},
            DEFAULT_MEASUREMENT_CONFIG,
            "invalid_measurement_metadata",
        ),
        (
            _region([0.2, 0.4, 0.6]),
            {},
            replace(DEFAULT_MEASUREMENT_CONFIG, epsilon=0.01),
            "unsupported_measurement_config",
        ),
    ],
)
def test_measurement_rejects_silent_type_config_and_metadata_changes(
    liquid, metadata, config, reason
):
    ids = dict(
        record_id="measurement-integrity",
        frame_id="frame-integrity",
        liquid_roi_id="vial-liquid-v1",
        patch_id="fixed-patch-v1",
    )
    ids.update(metadata)
    result = measure_feed_regions(
        liquid,
        _region([0.5, 0.5, 0.5]),
        config=config,
        **ids,
    )

    assert result.validity == "invalid"
    assert result.reason_codes == (reason,)
    assert result.log_ratio is None


def _split_region(low, high):
    return np.vstack([_region(low, 8), _region(high, 8)])


@pytest.mark.parametrize(
    ("liquid", "patch", "reason"),
    [
        (_region([1e-5, 0.4, 0.6]), _region([0.5, 0.5, 0.5]), "liquid_underexposed"),
        (_region([0.2, 0.4, 0.6]), _region([0.5, 1e-5, 0.5]), "patch_underexposed"),
        (_region([0.2, 0.4, 1.0]), _region([0.5, 0.5, 0.5]), "liquid_saturated"),
        (_region([0.2, 0.4, 0.6]), _region([1.0, 0.5, 0.5]), "patch_saturated"),
        (
            _region([0.2, 0.4, 0.6]),
            _split_region([0.3, 0.3, 0.3], [0.7, 0.7, 0.7]),
            "patch_spatially_unstable",
        ),
    ],
)
def test_quality_limit_failures_are_uncertain_and_not_comparable(liquid, patch, reason):
    result = measure_feed_regions(
        liquid,
        patch,
        record_id=f"measurement-{reason}",
        frame_id="frame-quality",
        liquid_roi_id="vial-liquid-v1",
        patch_id="fixed-patch-v1",
    )

    assert result.validity == "uncertain"
    assert result.reason_codes == (reason,)
    assert result.log_ratio is None
    assert result.normalized_rgb is None
    assert result.liquid_median is not None
    assert result.patch_median is not None


def _measurement(liquid, patch, suffix):
    return measure_feed_regions(
        _region(liquid),
        _region(patch),
        record_id=f"measurement-{suffix}",
        frame_id=f"frame-{suffix}",
        liquid_roi_id="vial-liquid-v1",
        patch_id="fixed-patch-v1",
    )


def test_common_scalar_and_channel_gain_are_nearly_invariant_away_from_epsilon():
    base_liquid = np.array([0.2, 0.3, 0.4])
    base_patch = np.array([0.5, 0.5, 0.5])
    base = _measurement(base_liquid, base_patch, "base")
    scalar = _measurement(base_liquid * 1.5, base_patch * 1.5, "scalar-gain")
    channel_gain = np.array([1.2, 0.8, 0.6])
    channel = _measurement(
        base_liquid * channel_gain,
        base_patch * channel_gain,
        "channel-gain",
    )

    assert base.validity == scalar.validity == channel.validity == "valid"
    assert np.max(np.abs(np.array(base.log_ratio) - scalar.log_ratio)) < 2e-6
    assert np.max(np.abs(np.array(base.log_ratio) - channel.log_ratio)) < 2e-6


def test_noncommon_and_nonlinear_nuisances_remain_visible_or_abstain():
    liquid = np.array([0.2, 0.3, 0.4])
    patch = np.array([0.5, 0.5, 0.5])
    base = _measurement(liquid, patch, "nuisance-base")
    liquid_only_gain = _measurement(liquid * 1.2, patch, "liquid-only-gain")
    gamma = _measurement(liquid**1.8, patch**1.8, "gamma")
    quantized = _measurement(
        np.round(liquid * 255) / 255,
        np.round(patch * 255) / 255,
        "quantized",
    )
    focus_mixed = measure_feed_regions(
        _split_region(liquid, patch),
        _region(patch),
        record_id="measurement-focus-mixed",
        frame_id="frame-focus-mixed",
        liquid_roi_id="vial-liquid-v1",
        patch_id="fixed-patch-v1",
    )
    dirty_patch = measure_feed_regions(
        _region(liquid),
        _split_region([0.1, 0.1, 0.1], patch),
        record_id="measurement-dirty-patch",
        frame_id="frame-dirty-patch",
        liquid_roi_id="vial-liquid-v1",
        patch_id="fixed-patch-v1",
    )

    def distance(other):
        return np.linalg.norm(np.array(base.log_ratio) - other.log_ratio)

    assert distance(liquid_only_gain) > 0.3
    assert distance(gamma) > 0.2
    assert 0 < distance(quantized) < 0.02
    assert distance(focus_mixed) > 0.1
    assert dirty_patch.validity == "uncertain"
    assert "patch_spatially_unstable" in dirty_patch.reason_codes
    assert dirty_patch.log_ratio is None


def test_valid_event_ledger_preserves_explicit_order_and_ids():
    events = (_event(10), _event(11))

    result = validate_event_ledger(events)

    assert result.validity == "valid"
    assert result.reason_codes == ()
    assert result.event_ids == ("event-10", "event-11")


@pytest.mark.parametrize(
    ("events", "validity", "reason"),
    [
        ((), "invalid", "empty_event_ledger"),
        ((_event(10), _event(11, record_id="event-10")), "invalid", "duplicate_event_id"),
        ((_event(10), _event(12)), "invalid", "schedule_gap"),
        ((_event(10), _event(11, observed_position=12)), "invalid", "schedule_observation_mismatch"),
        ((_event(10), _event(11, frame_ids=("frame-10",))), "invalid", "duplicate_frame_id"),
        ((_event(10, frame_ids=()),), "invalid", "missing_frame_id"),
        ((_event(10, mapping_status="ambiguous"),), "uncertain", "mapping_not_confirmed"),
        (
            (_event(10, timing_uncertainty_s=0.06, timing_tolerance_s=0.05),),
            "uncertain",
            "timing_uncertainty_exceeds_tolerance",
        ),
        ((_event(10, timing_uncertainty_s=np.nan),), "invalid", "invalid_timing"),
        ((_event(10, phase="pre_final_drop"),), "invalid", "invalid_phase"),
        ((_event(10), _event(11, sequence_id="sequence-2")), "invalid", "sequence_mismatch"),
    ],
)
def test_event_ledger_fails_closed_on_mapping_and_provenance_errors(events, validity, reason):
    result = validate_event_ledger(events)

    assert result.validity == validity
    assert reason in result.reason_codes


@pytest.mark.parametrize(
    ("events", "reason"),
    [
        ((_event(10), _event(11, scheduled_position="11")), "invalid_event_position"),
        ((_event(10, frame_ids=None),), "missing_frame_id"),
        ((replace(_event(10), record_id=[]),), "missing_event_metadata"),
        (("not-an-event",), "invalid_event_type"),
    ],
)
def test_malformed_ledger_fields_return_invalid_instead_of_raising(events, reason):
    result = validate_event_ledger(events)

    assert result.validity == "invalid"
    assert reason in result.reason_codes


def _candidate(event, measurement):
    return FeedDerivedReferenceCandidate(
        record_id="reference-candidate-1",
        event_record_id=event.record_id,
        measurement_record_id=measurement.record_id,
        selection_rule_id="externally-selected-rule-v1",
        source_frame_ids=(measurement.frame_id,),
    )


def test_explicit_valid_pair_yields_only_an_uncertain_descriptive_distance():
    reference_event = _event(10)
    current_event = _event(11)
    reference_measurement = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "10")
    current_measurement = _measurement([0.2, 0.3, 0.5], [0.5, 0.5, 0.5], "11")

    result = compare_feed_reference(
        record_id="comparison-1",
        current_event=current_event,
        current_measurement=current_measurement,
        event_ledger=(reference_event, current_event),
        reference_candidate=_candidate(reference_event, reference_measurement),
        reference_event=reference_event,
        reference_measurement=reference_measurement,
    )

    reference_z = np.log((np.array([0.2, 0.3, 0.4]) + 1e-6) / (0.5 + 1e-6))
    current_z = np.log((np.array([0.2, 0.3, 0.5]) + 1e-6) / (0.5 + 1e-6))
    reference_chroma = np.array([2 / 9, 3 / 9, 4 / 9])
    current_chroma = np.array([0.2, 0.3, 0.5])
    assert result.reference_available is True
    assert result.measurement_valid is True
    assert result.comparison_available is True
    assert result.validity == "uncertain"
    assert result.reason_codes == ("reference_cleanliness_unverified",)
    assert result.coverage_bucket == "compared_repeat_occurrence"
    assert result.appearance_distance == pytest.approx(np.linalg.norm(current_z - reference_z))
    assert result.normalized_rgb_distance == pytest.approx(
        np.linalg.norm(current_chroma - reference_chroma)
    )


@pytest.mark.parametrize(
    ("occurrence", "bucket", "reason"),
    [
        (1, "first_occurrence", "first_occurrence_no_reference"),
        (2, "reference_unavailable", "reference_unavailable"),
    ],
)
def test_missing_reference_has_an_explicit_noncomparison_bucket(occurrence, bucket, reason):
    current_event = _event(10, recipe_occurrence_index=occurrence)
    current_measurement = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "10")

    result = compare_feed_reference(
        record_id=f"comparison-no-reference-{occurrence}",
        current_event=current_event,
        current_measurement=current_measurement,
        event_ledger=(current_event,),
    )

    assert result.reference_available is False
    assert result.measurement_valid is True
    assert result.comparison_available is False
    assert result.validity == "invalid"
    assert result.reason_codes == (reason,)
    assert result.coverage_bucket == bucket
    assert result.appearance_distance is None
    assert result.normalized_rgb_distance is None


def test_missing_reference_does_not_hide_invalid_current_evidence():
    ambiguous_event = _event(20, recipe_occurrence_index=1, mapping_status="ambiguous")
    valid_measurement = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "20")
    invalid_measurement = _measurement([0.2, 0.3, 1.0], [0.5, 0.5, 0.5], "20")
    valid_event = _event(20, recipe_occurrence_index=1)

    ambiguous = compare_feed_reference(
        record_id="comparison-ambiguous-without-reference",
        current_event=ambiguous_event,
        current_measurement=valid_measurement,
        event_ledger=(ambiguous_event,),
    )
    invalid_current = compare_feed_reference(
        record_id="comparison-invalid-current-without-reference",
        current_event=valid_event,
        current_measurement=invalid_measurement,
        event_ledger=(valid_event,),
    )
    wrong_frame = compare_feed_reference(
        record_id="comparison-wrong-frame-without-reference",
        current_event=valid_event,
        current_measurement=replace(valid_measurement, frame_id="wrong-frame"),
        event_ledger=(valid_event,),
    )

    assert ambiguous.coverage_bucket == "event_mapping_invalid"
    assert "mapping_not_confirmed" in ambiguous.reason_codes
    assert invalid_current.coverage_bucket == "current_invalid"
    assert "liquid_saturated" in invalid_current.reason_codes
    assert wrong_frame.coverage_bucket == "event_mapping_invalid"
    assert "current_measurement_event_mismatch" in wrong_frame.reason_codes


def test_ambiguous_event_ledger_blocks_numeric_comparison():
    reference_event = _event(10)
    current_event = _event(11, mapping_status="ambiguous")
    reference_measurement = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "10")
    current_measurement = _measurement([0.2, 0.3, 0.5], [0.5, 0.5, 0.5], "11")

    result = compare_feed_reference(
        record_id="comparison-ambiguous-ledger",
        current_event=current_event,
        current_measurement=current_measurement,
        event_ledger=(reference_event, current_event),
        reference_candidate=_candidate(reference_event, reference_measurement),
        reference_event=reference_event,
        reference_measurement=reference_measurement,
    )

    assert result.comparison_available is False
    assert "mapping_not_confirmed" in result.reason_codes
    assert result.coverage_bucket == "event_mapping_invalid"
    assert result.appearance_distance is None


def test_invalid_current_and_reference_measurements_have_separate_buckets():
    reference_event = _event(10)
    current_event = _event(11)
    valid_reference = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "10")
    valid_current = _measurement([0.2, 0.3, 0.5], [0.5, 0.5, 0.5], "11")
    invalid_reference = _measurement([0.2, 0.3, 1.0], [0.5, 0.5, 0.5], "10")
    invalid_current = _measurement([0.2, 0.3, 1.0], [0.5, 0.5, 0.5], "11")

    current_result = compare_feed_reference(
        record_id="comparison-invalid-current",
        current_event=current_event,
        current_measurement=invalid_current,
        event_ledger=(reference_event, current_event),
        reference_candidate=_candidate(reference_event, valid_reference),
        reference_event=reference_event,
        reference_measurement=valid_reference,
    )
    reference_result = compare_feed_reference(
        record_id="comparison-invalid-reference",
        current_event=current_event,
        current_measurement=valid_current,
        event_ledger=(reference_event, current_event),
        reference_candidate=_candidate(reference_event, invalid_reference),
        reference_event=reference_event,
        reference_measurement=invalid_reference,
    )

    assert current_result.measurement_valid is False
    assert current_result.coverage_bucket == "current_invalid"
    assert current_result.reason_codes == ("current_measurement_uncertain", "liquid_saturated")
    assert current_result.appearance_distance is None
    assert reference_result.measurement_valid is True
    assert reference_result.coverage_bucket == "reference_stale_or_invalid"
    assert reference_result.reason_codes == (
        "reference_measurement_uncertain",
        "liquid_saturated",
    )
    assert reference_result.appearance_distance is None


def _compare_pair(reference_event, current_event, reference_measurement, current_measurement, candidate=None):
    return compare_feed_reference(
        record_id=f"comparison-{current_event.record_id}",
        current_event=current_event,
        current_measurement=current_measurement,
        event_ledger=(reference_event, current_event),
        reference_candidate=candidate or _candidate(reference_event, reference_measurement),
        reference_event=reference_event,
        reference_measurement=reference_measurement,
    )


def test_reference_provenance_state_and_geometry_fail_closed():
    reference_event = _event(10)
    current_event = _event(11)
    reference_measurement = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "10")
    current_measurement = _measurement([0.2, 0.3, 0.5], [0.5, 0.5, 0.5], "11")
    candidate = _candidate(reference_event, reference_measurement)

    cases = [
        (
            "unknown-state",
            replace(reference_event, appearance_domain_id=None),
            replace(current_event, appearance_domain_id=None),
            reference_measurement,
            current_measurement,
            candidate,
            "reference_state_unknown",
        ),
        (
            "changed-state",
            reference_event,
            replace(current_event, appearance_domain_id="continuous-run-2"),
            reference_measurement,
            current_measurement,
            candidate,
            "reference_stale",
        ),
        (
            "recipe-mismatch",
            reference_event,
            replace(current_event, recipe_id="recipe-8"),
            reference_measurement,
            current_measurement,
            candidate,
            "reference_recipe_mismatch",
        ),
        (
            "not-prior",
            replace(reference_event, recipe_occurrence_index=2),
            replace(current_event, recipe_occurrence_index=1),
            reference_measurement,
            current_measurement,
            candidate,
            "reference_not_prior",
        ),
        (
            "geometry",
            reference_event,
            current_event,
            reference_measurement,
            replace(current_measurement, liquid_roi_id="different-liquid-roi"),
            candidate,
            "measurement_geometry_mismatch",
        ),
        (
            "source-frame",
            reference_event,
            current_event,
            reference_measurement,
            current_measurement,
            replace(candidate, source_frame_ids=("wrong-frame",)),
            "reference_source_mismatch",
        ),
        (
            "claimed-clean",
            reference_event,
            current_event,
            reference_measurement,
            current_measurement,
            replace(candidate, cleanliness_status="verified_clean"),
            "invalid_reference_cleanliness_status",
        ),
        (
            "current-frame",
            reference_event,
            current_event,
            reference_measurement,
            replace(current_measurement, frame_id="wrong-current-frame"),
            candidate,
            "current_measurement_event_mismatch",
        ),
    ]

    for _, ref_event, cur_event, ref_measurement, cur_measurement, ref_candidate, reason in cases:
        result = _compare_pair(
            ref_event,
            cur_event,
            ref_measurement,
            cur_measurement,
            ref_candidate,
        )
        assert result.comparison_available is False
        assert reason in result.reason_codes
        expected_bucket = (
            "event_mapping_invalid"
            if reason == "current_measurement_event_mismatch"
            else "reference_stale_or_invalid"
        )
        assert result.coverage_bucket == expected_bucket
        assert result.appearance_distance is None


def test_reference_and_current_frames_cannot_overlap():
    reference_event = _event(10, frame_ids=("shared-frame",))
    current_event = _event(11, frame_ids=("shared-frame",))
    reference_measurement = replace(
        _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "10"),
        frame_id="shared-frame",
    )
    current_measurement = replace(
        _measurement([0.2, 0.3, 0.5], [0.5, 0.5, 0.5], "11"),
        frame_id="shared-frame",
    )

    result = _compare_pair(
        reference_event,
        current_event,
        reference_measurement,
        current_measurement,
    )

    assert result.comparison_available is False
    assert "duplicate_frame_id" in result.reason_codes
    assert result.appearance_distance is None


def test_comparison_rejects_forged_valid_measurements_and_substituted_ledger_records():
    reference_event = _event(10)
    current_event = _event(11)
    reference_measurement = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "10")
    current_measurement = _measurement([0.2, 0.3, 0.5], [0.5, 0.5, 0.5], "11")
    forged_current = replace(current_measurement, log_ratio=(np.nan, 0.0, 0.0))
    forged_reference = replace(reference_measurement, normalized_rgb=(0.2, 0.2, 0.2))
    forged_finite_log = replace(reference_measurement, log_ratio=(99.0, 99.0, 99.0))
    forged_finite_normalized = replace(
        reference_measurement,
        normalized_rgb=(0.2, 0.2, 0.6),
    )

    current_result = _compare_pair(
        reference_event,
        current_event,
        reference_measurement,
        forged_current,
    )
    reference_result = _compare_pair(
        reference_event,
        current_event,
        forged_reference,
        current_measurement,
    )
    finite_log_result = _compare_pair(
        reference_event,
        current_event,
        forged_finite_log,
        current_measurement,
    )
    finite_normalized_result = _compare_pair(
        reference_event,
        current_event,
        forged_finite_normalized,
        current_measurement,
    )
    substituted_event = replace(current_event, recipe_id="substituted-recipe")
    substituted_result = compare_feed_reference(
        record_id="comparison-substituted-ledger-record",
        current_event=current_event,
        current_measurement=current_measurement,
        event_ledger=(reference_event, substituted_event),
        reference_candidate=_candidate(reference_event, reference_measurement),
        reference_event=reference_event,
        reference_measurement=reference_measurement,
    )

    assert current_result.measurement_valid is False
    assert current_result.coverage_bucket == "current_invalid"
    assert current_result.reason_codes == ("invalid_current_measurement_record",)
    assert current_result.appearance_distance is None
    assert reference_result.coverage_bucket == "reference_stale_or_invalid"
    assert reference_result.reason_codes == ("invalid_reference_measurement_record",)
    assert reference_result.appearance_distance is None
    assert finite_log_result.reason_codes == ("invalid_reference_measurement_record",)
    assert finite_log_result.appearance_distance is None
    assert finite_normalized_result.reason_codes == ("invalid_reference_measurement_record",)
    assert finite_normalized_result.normalized_rgb_distance is None
    assert substituted_result.coverage_bucket == "event_mapping_invalid"
    assert substituted_result.reason_codes == ("current_event_record_mismatch",)


def test_unverified_shifted_reference_exposes_false_disagreement_and_masking():
    reference_event = _event(10)
    current_event = _event(11)
    shifted_reference = _measurement([0.35, 0.3, 0.4], [0.5, 0.5, 0.5], "10")
    later_unshifted = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "11")
    later_matching_shift = _measurement([0.35, 0.3, 0.4], [0.5, 0.5, 0.5], "11")

    false_disagreement = _compare_pair(
        reference_event,
        current_event,
        shifted_reference,
        later_unshifted,
    )
    masked_shift = _compare_pair(
        reference_event,
        current_event,
        shifted_reference,
        later_matching_shift,
    )

    assert false_disagreement.comparison_available is True
    assert false_disagreement.appearance_distance > 0.5
    assert masked_shift.comparison_available is True
    assert masked_shift.appearance_distance == pytest.approx(0.0)
    assert false_disagreement.reason_codes == masked_shift.reason_codes == (
        "reference_cleanliness_unverified",
    )


def test_coverage_summary_uses_one_bucket_per_requested_event():
    liquid = [0.2, 0.3, 0.4]
    patch = [0.5, 0.5, 0.5]
    first = compare_feed_reference(
        record_id="coverage-first",
        current_event=_event(100, suffix="coverage-first", recipe_occurrence_index=1),
        current_measurement=_measurement(liquid, patch, "coverage-first"),
        event_ledger=(_event(100, suffix="coverage-first", recipe_occurrence_index=1),),
    )
    unavailable_event = _event(
        101,
        suffix="coverage-unavailable",
        recipe_occurrence_index=2,
    )
    unavailable = compare_feed_reference(
        record_id="coverage-unavailable",
        current_event=unavailable_event,
        current_measurement=_measurement(liquid, patch, "coverage-unavailable"),
        event_ledger=(unavailable_event,),
    )
    stale_reference_event = _event(110, suffix="coverage-stale-reference")
    stale_current_event = _event(
        111,
        suffix="coverage-stale-current",
        appearance_domain_id="changed-domain",
    )
    stale = _compare_pair(
        stale_reference_event,
        stale_current_event,
        _measurement(liquid, patch, "coverage-stale-reference"),
        _measurement(liquid, patch, "coverage-stale-current"),
    )
    invalid_reference_event = _event(120, suffix="coverage-invalid-reference")
    invalid_current_event = _event(121, suffix="coverage-invalid-current")
    current_invalid = _compare_pair(
        invalid_reference_event,
        invalid_current_event,
        _measurement(liquid, patch, "coverage-invalid-reference"),
        _measurement([0.2, 0.3, 1.0], patch, "coverage-invalid-current"),
    )
    mapping_reference_event = _event(130, suffix="coverage-mapping-reference")
    mapping_current_event = _event(
        131,
        suffix="coverage-mapping-current",
        mapping_status="ambiguous",
    )
    mapping_invalid = _compare_pair(
        mapping_reference_event,
        mapping_current_event,
        _measurement(liquid, patch, "coverage-mapping-reference"),
        _measurement(liquid, patch, "coverage-mapping-current"),
    )
    compared_reference_event = _event(140, suffix="coverage-compared-reference")
    compared_current_event = _event(141, suffix="coverage-compared-current")
    compared = _compare_pair(
        compared_reference_event,
        compared_current_event,
        _measurement(liquid, patch, "coverage-compared-reference"),
        _measurement([0.2, 0.3, 0.5], patch, "coverage-compared-current"),
    )

    summary = summarize_reference_coverage(
        (first, unavailable, stale, current_invalid, mapping_invalid, compared)
    )

    assert summary.total_requested == 6
    assert summary.first_occurrence == 1
    assert summary.reference_unavailable == 1
    assert summary.reference_stale_or_invalid == 1
    assert summary.current_invalid == 1
    assert summary.event_mapping_invalid == 1
    assert summary.compared_repeat_occurrence == 1


def test_coverage_summary_rejects_duplicate_current_event_denominators():
    current_event = _event(20, recipe_occurrence_index=1)
    current_measurement = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "20")
    first = compare_feed_reference(
        record_id="coverage-first",
        current_event=current_event,
        current_measurement=current_measurement,
        event_ledger=(current_event,),
    )

    with pytest.raises(ValueError, match="duplicate current event"):
        summarize_reference_coverage((first, replace(first, record_id="coverage-replay")))


def test_coverage_summary_rejects_a_relabelled_noncomparison_bucket():
    current_event = _event(20, recipe_occurrence_index=1)
    current_measurement = _measurement([0.2, 0.3, 0.4], [0.5, 0.5, 0.5], "20")
    first = compare_feed_reference(
        record_id="coverage-first",
        current_event=current_event,
        current_measurement=current_measurement,
        event_ledger=(current_event,),
    )
    relabelled = replace(first, coverage_bucket="compared_repeat_occurrence")

    with pytest.raises(ValueError, match="coverage bucket conflicts"):
        summarize_reference_coverage((relabelled,))


def test_contract_audit_reports_all_predeclared_properties_and_exact_coverage():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_feed_reference_contract.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["schema_version"] == 1
    assert report["contract_version"] == "feed-reference-contract-v1"
    assert report["input_kind"] == "constructed_arrays_and_metadata_only"
    assert report["case_count"] >= 20
    assert report["pass_count"] == report["case_count"]
    assert report["fail_count"] == 0
    assert all(case["passed"] is True for case in report["cases"])
    assert report["coverage"] == {
        "total_requested": 6,
        "first_occurrence": 1,
        "reference_unavailable": 1,
        "reference_stale_or_invalid": 1,
        "current_invalid": 1,
        "event_mapping_invalid": 1,
        "compared_repeat_occurrence": 1,
    }
    forbidden_result_keys = {"threshold", "decision", "defect", "cause", "residue_fraction"}
    assert not forbidden_result_keys.intersection(report)
    assert all(not forbidden_result_keys.intersection(case) for case in report["cases"])
