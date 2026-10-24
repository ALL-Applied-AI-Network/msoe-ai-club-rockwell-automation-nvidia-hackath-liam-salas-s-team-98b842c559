"""Synthetic-only feed-derived appearance/reference contract.

This module does not select a reference, detect a defect, estimate carryover, or infer a cause.
See docs/feed-reference-contract.md for the frozen scope and assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
import numbers

import numpy as np


CONTRACT_VERSION = "feed-reference-contract-v1"


@dataclass(frozen=True)
class MeasurementConfig:
    """Frozen numerical limits for the v1 synthetic contract, not deployment settings."""

    version: str = CONTRACT_VERSION
    epsilon: float = 1e-6
    min_region_pixels: int = 16
    underexposed_median: float = 1e-4
    saturation_value: float = 0.99
    max_saturated_fraction: float = 0.05
    max_patch_mad: float = 0.08


DEFAULT_MEASUREMENT_CONFIG = MeasurementConfig()


@dataclass(frozen=True)
class FeedRegionMeasurement:
    """A descriptive same-frame appearance record with no threshold or decision."""

    record_id: str
    frame_id: str
    liquid_roi_id: str
    patch_id: str
    config_version: str
    validity: str
    reason_codes: tuple[str, ...]
    liquid_area: int
    patch_area: int
    liquid_median: tuple[float, float, float] | None
    patch_median: tuple[float, float, float] | None
    liquid_saturated_fraction: tuple[float, float, float] | None
    patch_saturated_fraction: tuple[float, float, float] | None
    patch_mad: tuple[float, float, float] | None
    log_ratio: tuple[float, float, float] | None
    normalized_rgb: tuple[float, float, float] | None


@dataclass(frozen=True)
class FeedEvent:
    """Explicit schedule-to-frame mapping supplied by an upstream provenance owner."""

    record_id: str
    schedule_record_id: str
    sequence_id: str
    transition_id: str
    recipe_id: str
    recipe_occurrence_index: int
    vial_ordinal: int
    scheduled_position: int
    observed_position: int
    frame_ids: tuple[str, ...]
    window_id: str
    timing_source: str
    timing_uncertainty_s: float
    timing_tolerance_s: float
    mapping_status: str
    phase: str
    appearance_domain_id: str | None


@dataclass(frozen=True)
class LedgerValidation:
    validity: str
    reason_codes: tuple[str, ...]
    event_ids: tuple[str, ...]


@dataclass(frozen=True)
class FeedDerivedReferenceCandidate:
    """An externally selected prior appearance; cleanliness is intentionally unverified."""

    record_id: str
    event_record_id: str
    measurement_record_id: str
    selection_rule_id: str
    source_frame_ids: tuple[str, ...]
    cleanliness_status: str = "unverified"


@dataclass(frozen=True)
class ReferenceComparison:
    """Descriptive comparison availability and distance, never a defect decision."""

    record_id: str
    current_event_id: str
    current_measurement_id: str
    reference_candidate_id: str | None
    reference_available: bool
    measurement_valid: bool
    comparison_available: bool
    validity: str
    reason_codes: tuple[str, ...]
    coverage_bucket: str
    appearance_distance: float | None
    normalized_rgb_distance: float | None


COVERAGE_BUCKETS = (
    "first_occurrence",
    "reference_unavailable",
    "reference_stale_or_invalid",
    "current_invalid",
    "event_mapping_invalid",
    "compared_repeat_occurrence",
)


@dataclass(frozen=True)
class CoverageSummary:
    total_requested: int
    first_occurrence: int
    reference_unavailable: int
    reference_stale_or_invalid: int
    current_invalid: int
    event_mapping_invalid: int
    compared_repeat_occurrence: int


def summarize_reference_coverage(comparisons):
    """Count one mutually exclusive availability bucket per requested current event."""
    comparisons = tuple(comparisons)
    counts = {bucket: 0 for bucket in COVERAGE_BUCKETS}
    seen_record_ids = set()
    seen_current_event_ids = set()
    for comparison in comparisons:
        if not isinstance(comparison, ReferenceComparison):
            raise TypeError("coverage inputs must be ReferenceComparison records")
        if not isinstance(comparison.record_id, str) or not comparison.record_id:
            raise ValueError("coverage comparison record ID must be nonempty")
        if not isinstance(comparison.current_event_id, str) or not comparison.current_event_id:
            raise ValueError("coverage current event ID must be nonempty")
        if comparison.record_id in seen_record_ids:
            raise ValueError(f"duplicate comparison record: {comparison.record_id}")
        if comparison.current_event_id in seen_current_event_ids:
            raise ValueError(f"duplicate current event: {comparison.current_event_id}")
        if comparison.coverage_bucket not in counts:
            raise ValueError(f"unknown coverage bucket: {comparison.coverage_bucket}")
        expected_bucket = _derive_coverage_bucket(comparison)
        if comparison.coverage_bucket != expected_bucket:
            raise ValueError(
                "coverage bucket conflicts with comparison state: "
                f"{comparison.coverage_bucket} != {expected_bucket}"
            )
        seen_record_ids.add(comparison.record_id)
        seen_current_event_ids.add(comparison.current_event_id)
        counts[comparison.coverage_bucket] += 1
    return CoverageSummary(total_requested=len(comparisons), **counts)


_EVENT_MAPPING_REASONS = {
    "empty_event_ledger",
    "invalid_event_type",
    "missing_event_metadata",
    "invalid_event_position",
    "schedule_observation_mismatch",
    "missing_frame_id",
    "invalid_timing",
    "timing_uncertainty_exceeds_tolerance",
    "mapping_not_confirmed",
    "invalid_phase",
    "duplicate_event_id",
    "duplicate_schedule_record_id",
    "duplicate_frame_id",
    "sequence_mismatch",
    "schedule_nonmonotone",
    "schedule_gap",
    "observed_position_nonmonotone",
    "observed_position_gap",
    "current_event_not_in_ledger",
    "current_event_record_mismatch",
    "reference_event_not_in_ledger",
    "reference_event_record_mismatch",
    "current_measurement_event_mismatch",
    "reference_measurement_event_mismatch",
}


def _derive_coverage_bucket(comparison):
    if comparison.comparison_available:
        distances = (comparison.appearance_distance, comparison.normalized_rgb_distance)
        if (
            not comparison.reference_available
            or not comparison.measurement_valid
            or comparison.validity != "uncertain"
            or comparison.reason_codes != ("reference_cleanliness_unverified",)
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, numbers.Real)
                or not np.isfinite(value)
                or value < 0
                for value in distances
            )
        ):
            raise ValueError("available comparison has inconsistent evidence state")
        return "compared_repeat_occurrence"
    if comparison.appearance_distance is not None or comparison.normalized_rgb_distance is not None:
        raise ValueError("unavailable comparison cannot carry numeric distances")
    if comparison.validity != "invalid" or not comparison.reason_codes:
        raise ValueError("unavailable comparison has inconsistent validity state")
    if comparison.reason_codes == ("first_occurrence_no_reference",):
        return "first_occurrence"
    if comparison.reason_codes == ("reference_unavailable",):
        return "reference_unavailable"
    if not comparison.measurement_valid and (
        comparison.reason_codes[0] == "invalid_current_measurement_record"
        or comparison.reason_codes[0].startswith("current_measurement_")
    ):
        return "current_invalid"
    if any(reason in _EVENT_MAPPING_REASONS for reason in comparison.reason_codes):
        return "event_mapping_invalid"
    return "reference_stale_or_invalid"


def _finite_triplet(value):
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and all(
            not isinstance(item, (bool, np.bool_))
            and isinstance(item, numbers.Real)
            and np.isfinite(item)
            for item in value
        )
    )


def _measurement_record_is_well_formed(measurement):
    if not isinstance(measurement, FeedRegionMeasurement):
        return False
    metadata = (
        measurement.record_id,
        measurement.frame_id,
        measurement.liquid_roi_id,
        measurement.patch_id,
    )
    if any(not isinstance(value, str) or not value for value in metadata):
        return False
    if measurement.config_version != CONTRACT_VERSION:
        return False
    if measurement.validity not in {"valid", "uncertain", "invalid"}:
        return False
    if (
        not isinstance(measurement.reason_codes, tuple)
        or any(not isinstance(reason, str) or not reason for reason in measurement.reason_codes)
    ):
        return False
    areas = (measurement.liquid_area, measurement.patch_area)
    if any(
        isinstance(area, (bool, np.bool_))
        or not isinstance(area, numbers.Integral)
        or area < 0
        for area in areas
    ):
        return False
    if measurement.validity != "valid":
        return (
            bool(measurement.reason_codes)
            and measurement.log_ratio is None
            and measurement.normalized_rgb is None
        )
    if measurement.reason_codes or min(areas) < DEFAULT_MEASUREMENT_CONFIG.min_region_pixels:
        return False
    triplets = (
        measurement.liquid_median,
        measurement.patch_median,
        measurement.liquid_saturated_fraction,
        measurement.patch_saturated_fraction,
        measurement.patch_mad,
        measurement.log_ratio,
        measurement.normalized_rgb,
    )
    if not all(_finite_triplet(value) for value in triplets):
        return False
    if any(value < 0 or value > 1 for value in measurement.liquid_median):
        return False
    if any(value < 0 or value > 1 for value in measurement.patch_median):
        return False
    if any(
        value < 0 or value > 1
        for fractions in (
            measurement.liquid_saturated_fraction,
            measurement.patch_saturated_fraction,
        )
        for value in fractions
    ):
        return False
    if any(value < 0 for value in measurement.patch_mad):
        return False
    if any(value < 0 or value > 1 for value in measurement.normalized_rgb):
        return False
    liquid_median = np.asarray(measurement.liquid_median)
    patch_median = np.asarray(measurement.patch_median)
    expected_log_ratio = np.log(
        (liquid_median + DEFAULT_MEASUREMENT_CONFIG.epsilon)
        / (patch_median + DEFAULT_MEASUREMENT_CONFIG.epsilon)
    )
    expected_normalized = liquid_median / liquid_median.sum()
    return (
        np.allclose(
            measurement.log_ratio,
            expected_log_ratio,
            rtol=1e-12,
            atol=1e-12,
        )
        and np.allclose(
            measurement.normalized_rgb,
            expected_normalized,
            rtol=1e-12,
            atol=1e-12,
        )
    )


def compare_feed_reference(
    *,
    record_id: str,
    current_event: FeedEvent,
    current_measurement: FeedRegionMeasurement,
    event_ledger,
    reference_candidate: FeedDerivedReferenceCandidate | None = None,
    reference_event: FeedEvent | None = None,
    reference_measurement: FeedRegionMeasurement | None = None,
):
    """Compare explicit current/prior records without selecting either record."""
    def unavailable(
        reason_codes,
        bucket,
        *,
        reference_available,
        measurement_valid_override=None,
    ):
        return ReferenceComparison(
            record_id=record_id,
            current_event_id=current_event.record_id,
            current_measurement_id=current_measurement.record_id,
            reference_candidate_id=(
                reference_candidate.record_id if reference_candidate is not None else None
            ),
            reference_available=reference_available,
            measurement_valid=(
                current_measurement.validity == "valid"
                if measurement_valid_override is None
                else measurement_valid_override
            ),
            comparison_available=False,
            validity="invalid",
            reason_codes=tuple(reason_codes),
            coverage_bucket=bucket,
            appearance_distance=None,
            normalized_rgb_distance=None,
        )

    ledger = tuple(event_ledger)
    ledger_validation = validate_event_ledger(ledger)
    membership_reasons = []
    current_records = [
        event
        for event in ledger
        if isinstance(event, FeedEvent) and event.record_id == current_event.record_id
    ]
    if not current_records:
        membership_reasons.append("current_event_not_in_ledger")
    elif current_records[0] != current_event:
        membership_reasons.append("current_event_record_mismatch")
    if ledger_validation.validity != "valid" or membership_reasons:
        return unavailable(
            tuple(membership_reasons) + ledger_validation.reason_codes,
            "event_mapping_invalid",
            reference_available=reference_candidate is not None,
        )
    if current_measurement.frame_id not in current_event.frame_ids:
        return unavailable(
            ("current_measurement_event_mismatch",),
            "event_mapping_invalid",
            reference_available=reference_candidate is not None,
        )
    if not _measurement_record_is_well_formed(current_measurement):
        return unavailable(
            ("invalid_current_measurement_record",),
            "current_invalid",
            reference_available=reference_candidate is not None,
            measurement_valid_override=False,
        )
    if current_measurement.validity != "valid":
        return unavailable(
            (f"current_measurement_{current_measurement.validity}",)
            + current_measurement.reason_codes,
            "current_invalid",
            reference_available=reference_candidate is not None,
        )
    if reference_candidate is None and reference_event is None and reference_measurement is None:
        if current_event.recipe_occurrence_index == 1:
            return unavailable(
                ("first_occurrence_no_reference",),
                "first_occurrence",
                reference_available=False,
            )
        return unavailable(
            ("reference_unavailable",),
            "reference_unavailable",
            reference_available=False,
        )
    if reference_candidate is None or reference_event is None or reference_measurement is None:
        return unavailable(
            ("incomplete_reference_record",),
            "reference_stale_or_invalid",
            reference_available=reference_candidate is not None,
        )
    reference_records = [
        event
        for event in ledger
        if isinstance(event, FeedEvent) and event.record_id == reference_event.record_id
    ]
    if not reference_records:
        return unavailable(
            ("reference_event_not_in_ledger",),
            "event_mapping_invalid",
            reference_available=True,
        )
    if reference_records[0] != reference_event:
        return unavailable(
            ("reference_event_record_mismatch",),
            "event_mapping_invalid",
            reference_available=True,
        )
    if not _measurement_record_is_well_formed(reference_measurement):
        return unavailable(
            ("invalid_reference_measurement_record",),
            "reference_stale_or_invalid",
            reference_available=True,
        )
    if reference_measurement.validity != "valid":
        return unavailable(
            (f"reference_measurement_{reference_measurement.validity}",)
            + reference_measurement.reason_codes,
            "reference_stale_or_invalid",
            reference_available=True,
        )
    provenance_reasons = []

    def add_provenance(reason):
        if reason not in provenance_reasons:
            provenance_reasons.append(reason)

    if (
        not reference_candidate.record_id
        or not reference_candidate.selection_rule_id
        or not reference_candidate.source_frame_ids
    ):
        add_provenance("invalid_reference_metadata")
    if reference_candidate.cleanliness_status != "unverified":
        add_provenance("invalid_reference_cleanliness_status")
    if (
        reference_candidate.event_record_id != reference_event.record_id
        or reference_candidate.measurement_record_id != reference_measurement.record_id
    ):
        add_provenance("reference_record_mismatch")
    if reference_candidate.source_frame_ids != (reference_measurement.frame_id,):
        add_provenance("reference_source_mismatch")
    if current_measurement.frame_id not in current_event.frame_ids:
        add_provenance("current_measurement_event_mismatch")
    if reference_measurement.frame_id not in reference_event.frame_ids:
        add_provenance("reference_measurement_event_mismatch")
    if set(current_event.frame_ids) & set(reference_candidate.source_frame_ids):
        add_provenance("reference_current_frame_overlap")
    if current_event.recipe_id != reference_event.recipe_id:
        add_provenance("reference_recipe_mismatch")
    if (
        reference_event.scheduled_position >= current_event.scheduled_position
        or reference_event.recipe_occurrence_index >= current_event.recipe_occurrence_index
    ):
        add_provenance("reference_not_prior")
    if (
        current_measurement.liquid_roi_id != reference_measurement.liquid_roi_id
        or current_measurement.patch_id != reference_measurement.patch_id
        or current_measurement.config_version != reference_measurement.config_version
    ):
        add_provenance("measurement_geometry_mismatch")
    if (
        reference_event.appearance_domain_id is None
        or current_event.appearance_domain_id is None
    ):
        add_provenance("reference_state_unknown")
    elif reference_event.appearance_domain_id != current_event.appearance_domain_id:
        add_provenance("reference_stale")
    if provenance_reasons:
        return unavailable(
            tuple(provenance_reasons),
            "reference_stale_or_invalid",
            reference_available=True,
        )
    appearance_distance = float(
        np.linalg.norm(
            np.asarray(current_measurement.log_ratio)
            - np.asarray(reference_measurement.log_ratio)
        )
    )
    normalized_rgb_distance = float(
        np.linalg.norm(
            np.asarray(current_measurement.normalized_rgb)
            - np.asarray(reference_measurement.normalized_rgb)
        )
    )
    return ReferenceComparison(
        record_id=record_id,
        current_event_id=current_event.record_id,
        current_measurement_id=current_measurement.record_id,
        reference_candidate_id=reference_candidate.record_id,
        reference_available=True,
        measurement_valid=True,
        comparison_available=True,
        validity="uncertain",
        reason_codes=("reference_cleanliness_unverified",),
        coverage_bucket="compared_repeat_occurrence",
        appearance_distance=appearance_distance,
        normalized_rgb_distance=normalized_rgb_distance,
    )


def validate_event_ledger(events):
    """Validate an explicitly ordered, contiguous schedule-to-frame ledger."""
    events = tuple(events)
    if not events:
        return LedgerValidation("invalid", ("empty_event_ledger",), ())

    invalid_reasons = []
    uncertain_reasons = []

    def add(target, reason):
        if reason not in target:
            target.append(reason)

    required_text_fields = (
        "record_id",
        "schedule_record_id",
        "sequence_id",
        "transition_id",
        "recipe_id",
        "window_id",
        "timing_source",
    )
    for event in events:
        if not isinstance(event, FeedEvent):
            add(invalid_reasons, "invalid_event_type")
            continue
        if any(
            not isinstance(getattr(event, field), str) or not getattr(event, field)
            for field in required_text_fields
        ):
            add(invalid_reasons, "missing_event_metadata")
        integer_fields = (
            event.recipe_occurrence_index,
            event.vial_ordinal,
            event.scheduled_position,
            event.observed_position,
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Integral)
            or value < 1
            for value in integer_fields
        ):
            add(invalid_reasons, "invalid_event_position")
        if event.scheduled_position != event.observed_position:
            add(invalid_reasons, "schedule_observation_mismatch")
        if (
            not isinstance(event.frame_ids, tuple)
            or not event.frame_ids
            or any(not isinstance(frame, str) or not frame for frame in event.frame_ids)
        ):
            add(invalid_reasons, "missing_frame_id")
        timing_values = (event.timing_uncertainty_s, event.timing_tolerance_s)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Real)
            or not np.isfinite(value)
            for value in timing_values
        ) or event.timing_uncertainty_s < 0 or event.timing_tolerance_s <= 0:
            add(invalid_reasons, "invalid_timing")
        elif event.timing_uncertainty_s > event.timing_tolerance_s:
            add(uncertain_reasons, "timing_uncertainty_exceeds_tolerance")
        if event.mapping_status != "confirmed":
            add(uncertain_reasons, "mapping_not_confirmed")
        if event.phase != "final_mixture":
            add(invalid_reasons, "invalid_phase")

    typed_events = tuple(event for event in events if isinstance(event, FeedEvent))
    event_ids = [event.record_id for event in typed_events if isinstance(event.record_id, str)]
    if len(event_ids) != len(set(event_ids)):
        add(invalid_reasons, "duplicate_event_id")
    schedule_ids = [
        event.schedule_record_id
        for event in typed_events
        if isinstance(event.schedule_record_id, str)
    ]
    if len(schedule_ids) != len(set(schedule_ids)):
        add(invalid_reasons, "duplicate_schedule_record_id")
    frame_ids = [
        frame
        for event in typed_events
        if isinstance(event.frame_ids, tuple)
        for frame in event.frame_ids
        if isinstance(frame, str)
    ]
    if len(frame_ids) != len(set(frame_ids)):
        add(invalid_reasons, "duplicate_frame_id")
    sequence_ids = {
        event.sequence_id for event in typed_events if isinstance(event.sequence_id, str)
    }
    if len(sequence_ids) > 1:
        add(invalid_reasons, "sequence_mismatch")

    def valid_position(value):
        return (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, numbers.Integral)
            and value >= 1
        )

    for previous, current in zip(typed_events, typed_events[1:]):
        if not all(
            valid_position(value)
            for value in (
                previous.scheduled_position,
                previous.observed_position,
                current.scheduled_position,
                current.observed_position,
            )
        ):
            continue
        schedule_delta = current.scheduled_position - previous.scheduled_position
        observed_delta = current.observed_position - previous.observed_position
        if schedule_delta <= 0:
            add(invalid_reasons, "schedule_nonmonotone")
        elif schedule_delta > 1:
            add(invalid_reasons, "schedule_gap")
        if observed_delta <= 0:
            add(invalid_reasons, "observed_position_nonmonotone")
        elif observed_delta > 1:
            add(invalid_reasons, "observed_position_gap")

    if invalid_reasons:
        validity = "invalid"
    elif uncertain_reasons:
        validity = "uncertain"
    else:
        validity = "valid"
    return LedgerValidation(
        validity=validity,
        reason_codes=tuple(invalid_reasons + uncertain_reasons),
        event_ids=tuple(event_ids),
    )


def _measurement_without_descriptor(
    *,
    record_id: str,
    frame_id: str,
    liquid_roi_id: str,
    patch_id: str,
    config: MeasurementConfig,
    validity: str,
    reason_codes: tuple[str, ...],
    liquid_area: int,
    patch_area: int,
    liquid_median=None,
    patch_median=None,
    liquid_saturated_fraction=None,
    patch_saturated_fraction=None,
    patch_mad=None,
):
    return FeedRegionMeasurement(
        record_id=record_id,
        frame_id=frame_id,
        liquid_roi_id=liquid_roi_id,
        patch_id=patch_id,
        config_version=config.version,
        validity=validity,
        reason_codes=reason_codes,
        liquid_area=liquid_area,
        patch_area=patch_area,
        liquid_median=liquid_median,
        patch_median=patch_median,
        liquid_saturated_fraction=liquid_saturated_fraction,
        patch_saturated_fraction=patch_saturated_fraction,
        patch_mad=patch_mad,
        log_ratio=None,
        normalized_rgb=None,
    )


def measure_feed_regions(
    liquid_rgb,
    patch_rgb,
    *,
    record_id: str,
    frame_id: str,
    liquid_roi_id: str,
    patch_id: str,
    config: MeasurementConfig = DEFAULT_MEASUREMENT_CONFIG,
):
    """Summarize named same-frame regions without making a decision."""
    effective_config = (
        config if isinstance(config, MeasurementConfig) else DEFAULT_MEASUREMENT_CONFIG
    )

    def raw_area(value):
        try:
            return len(value)
        except TypeError:
            return 0

    def fail(reason, liquid_area=None, patch_area=None):
        return _measurement_without_descriptor(
            record_id=record_id,
            frame_id=frame_id,
            liquid_roi_id=liquid_roi_id,
            patch_id=patch_id,
            config=effective_config,
            validity="invalid",
            reason_codes=(reason,),
            liquid_area=raw_area(liquid_rgb) if liquid_area is None else liquid_area,
            patch_area=raw_area(patch_rgb) if patch_area is None else patch_area,
        )

    metadata = (record_id, frame_id, liquid_roi_id, patch_id)
    if any(not isinstance(value, str) or not value for value in metadata):
        return fail("invalid_measurement_metadata")
    if config != DEFAULT_MEASUREMENT_CONFIG:
        return fail("unsupported_measurement_config")
    if np.ma.isMaskedArray(liquid_rgb) or np.ma.isMaskedArray(patch_rgb):
        return fail("masked_region_unsupported")
    try:
        raw_liquid = np.asarray(liquid_rgb)
        raw_patch = np.asarray(patch_rgb)
    except (TypeError, ValueError):
        return fail("invalid_region_type")
    if raw_liquid.dtype.kind not in "iuf" or raw_patch.dtype.kind not in "iuf":
        return fail("invalid_region_type")
    liquid = raw_liquid.astype(float, copy=False)
    patch = raw_patch.astype(float, copy=False)
    if liquid.ndim != 2 or patch.ndim != 2 or liquid.shape[1:] != (3,) or patch.shape[1:] != (3,):
        return fail("invalid_region_shape")
    if len(liquid) < config.min_region_pixels or len(patch) < config.min_region_pixels:
        return fail("insufficient_region_area", len(liquid), len(patch))
    if not np.isfinite(liquid).all() or not np.isfinite(patch).all():
        return fail("nonfinite_region", len(liquid), len(patch))
    if (liquid < 0).any() or (liquid > 1).any() or (patch < 0).any() or (patch > 1).any():
        return fail("region_out_of_range", len(liquid), len(patch))
    liquid_median = np.median(liquid, axis=0)
    patch_median = np.median(patch, axis=0)
    liquid_saturated = np.mean(liquid >= config.saturation_value, axis=0)
    patch_saturated = np.mean(patch >= config.saturation_value, axis=0)
    patch_mad = np.median(np.abs(patch - patch_median), axis=0)
    quality_reasons = []
    if np.any(liquid_median <= config.underexposed_median):
        quality_reasons.append("liquid_underexposed")
    if np.any(patch_median <= config.underexposed_median):
        quality_reasons.append("patch_underexposed")
    if np.any(liquid_saturated > config.max_saturated_fraction):
        quality_reasons.append("liquid_saturated")
    if np.any(patch_saturated > config.max_saturated_fraction):
        quality_reasons.append("patch_saturated")
    if np.any(patch_mad > config.max_patch_mad):
        quality_reasons.append("patch_spatially_unstable")
    summaries = dict(
        liquid_median=tuple(float(value) for value in liquid_median),
        patch_median=tuple(float(value) for value in patch_median),
        liquid_saturated_fraction=tuple(float(value) for value in liquid_saturated),
        patch_saturated_fraction=tuple(float(value) for value in patch_saturated),
        patch_mad=tuple(float(value) for value in patch_mad),
    )
    if quality_reasons:
        return _measurement_without_descriptor(
            record_id=record_id,
            frame_id=frame_id,
            liquid_roi_id=liquid_roi_id,
            patch_id=patch_id,
            config=config,
            validity="uncertain",
            reason_codes=tuple(quality_reasons),
            liquid_area=len(liquid),
            patch_area=len(patch),
            **summaries,
        )
    log_ratio = np.log(
        (liquid_median + config.epsilon) / (patch_median + config.epsilon)
    )
    normalized_rgb = liquid_median / liquid_median.sum()
    return FeedRegionMeasurement(
        record_id=record_id,
        frame_id=frame_id,
        liquid_roi_id=liquid_roi_id,
        patch_id=patch_id,
        config_version=config.version,
        validity="valid",
        reason_codes=(),
        liquid_area=len(liquid),
        patch_area=len(patch),
        **summaries,
        log_ratio=tuple(float(value) for value in log_ratio),
        normalized_rgb=tuple(float(value) for value in normalized_rgb),
    )
