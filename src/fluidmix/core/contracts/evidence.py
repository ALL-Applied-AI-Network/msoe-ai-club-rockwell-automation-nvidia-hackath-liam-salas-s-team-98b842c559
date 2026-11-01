"""Validation and artifact boundaries for the flat evidence-record-v1 contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping, Sequence

from ._common import (
    bounded_control_free_text,
    contained_relative_path,
    exact_lower_sha256,
    load_schema,
    pretty_json_bytes,
    publish_bytes_no_clobber,
    reject_nonfinite,
    relative_path_below_root,
    safe_posix_component,
    sha256_bytes,
    stable_hash_relative_file,
    stable_read_path,
    strict_json_bytes,
    validate_schema,
)


_SCHEMA = load_schema("evidence-record-v1.schema.json")
_PHASES = {"measured", "decided", "explained"}
_MODES = {"storage", "production"}
_TANKS = ("t100", "t200", "t400")
_DERIVATION_FIELDS = {
    "schema_version",
    "predecessor_phase",
    "derived_phase",
    "predecessor_sha256",
    "derived_sha256",
}
_ADJACENT_PHASES = {("measured", "decided"), ("decided", "explained")}
_DECISION_FIELDS = {"decision_id", "disposition", "limit_codes", "history_refs"}
_EXPLANATION_FIELDS = {
    "explanation_id",
    "model_revision",
    "prompt_sha256",
    "image_ids_sent",
    "guard_result",
}
_ID_FIELDS = (
    "run_id",
    "case_id",
    "block_recipe_id",
    "recipe_id",
    "split_version",
    "stage_id",
    "measurement_id",
    "evidence_id",
)
_NULLABLE_ID_FIELDS = (
    "previous_recipe_id",
    "colour_regime_id",
    "decision_id",
    "explanation_id",
    "model_revision",
)
_HASH_FIELDS = ("clip_sha256", "rgb_sha256", "mask_sha256")


def _validate_evidence_lexical_fields(value: dict[str, object]) -> None:
    """Apply full-string and Unicode-category rules JSON Schema cannot express."""

    for field in _ID_FIELDS:
        bounded_control_free_text(value[field], field)
    for field in _NULLABLE_ID_FIELDS:
        if value[field] is not None:
            bounded_control_free_text(value[field], field)
    for field in _HASH_FIELDS:
        exact_lower_sha256(value[field], field)
    if value["prompt_sha256"] is not None:
        exact_lower_sha256(value["prompt_sha256"], "prompt_sha256")
    safe_posix_component(value["image_id"], "image_id")

    pairs = value["pairwise_delta_e76"]
    history = value["history_refs"]
    limit_codes = value["limit_codes"]
    image_ids = value["image_ids_sent"]
    assert isinstance(pairs, list)
    assert isinstance(history, list)
    assert isinstance(limit_codes, list)
    assert isinstance(image_ids, list)
    for index, row in enumerate(pairs):
        assert isinstance(row, dict)
        bounded_control_free_text(
            row["other_evidence_id"],
            f"pairwise_delta_e76[{index}].other_evidence_id",
        )
    for index, code in enumerate(limit_codes):
        bounded_control_free_text(code, f"limit_codes[{index}]")
    for index, row in enumerate(history):
        assert isinstance(row, dict)
        bounded_control_free_text(
            row["evidence_id"], f"history_refs[{index}].evidence_id"
        )
        exact_lower_sha256(
            row["query_sha256"], f"history_refs[{index}].query_sha256"
        )
    for index, image_id in enumerate(image_ids):
        safe_posix_component(image_id, f"image_ids_sent[{index}]")


def _frame_range(value: object, role: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{role} must be a two-item frame range")
    start, end = value
    if type(start) is not int or type(end) is not int or start < 0 or start >= end:
        raise ValueError(f"{role} must satisfy 0 <= start < end")
    return start, end


def _sorted_unique_text(values: object, role: str) -> None:
    if not isinstance(values, list):
        raise TypeError(f"{role} must be a list")
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError(f"{role} must be sorted and unique")


def validate_evidence_record_v1(
    record: object,
    *,
    phase: str,
    mode: str = "storage",
    verified_colour_regime_binding: object | None = None,
) -> dict[str, object]:
    """Validate one exact per-stage record and return a defensive copy."""

    if phase not in _PHASES:
        raise ValueError("phase must be measured, decided, or explained")
    if mode not in _MODES:
        raise ValueError("mode must be storage or production")
    if verified_colour_regime_binding is not None:
        raise ValueError("colour regime bindings are unavailable until WP5")
    reject_nonfinite(record, "evidence record")
    validate_schema(record, _SCHEMA, "evidence record")
    assert isinstance(record, dict)
    value = deepcopy(record)
    _validate_evidence_lexical_fields(value)
    contained_relative_path(value["fill_clip_path"], "fill_clip_path")

    volumes = value["recipe_ml"]
    assert isinstance(volumes, dict)
    if sum(float(volumes[tank]) for tank in _TANKS) <= 0.0:
        raise ValueError("recipe_ml must have a positive total")
    active = {tank for tank in _TANKS if float(volumes[tank]) > 0.0}
    order = value["dispense_order"]
    assert isinstance(order, list)
    if set(order) != active or len(order) != len(active):
        raise ValueError("dispense_order must equal the positive-volume tanks")

    fill_start, fill_end = _frame_range(value["fill_frame_range"], "fill_frame_range")
    transitions = value["transition_frame_ranges"]
    assert isinstance(transitions, dict)
    entry = transitions["entry"]
    if entry is not None:
        _, entry_end = _frame_range(entry, "transition_frame_ranges.entry")
        if entry_end > fill_start:
            raise ValueError("entry transition overlaps the fill range")
    exit_range = transitions["exit"]
    if exit_range is not None:
        exit_start, _ = _frame_range(exit_range, "transition_frame_ranges.exit")
        if fill_end > exit_start:
            raise ValueError("exit transition overlaps the fill range")

    pairs = value["pairwise_delta_e76"]
    assert isinstance(pairs, list)
    pair_ids = [row["other_evidence_id"] for row in pairs]
    if pair_ids != sorted(pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pairwise_delta_e76 must be sorted and unique")
    if value["evidence_id"] in pair_ids:
        raise ValueError("pairwise_delta_e76 cannot contain a self-pair")
    _sorted_unique_text(value["limit_codes"], "limit_codes")
    history = value["history_refs"]
    assert isinstance(history, list)
    history_ids = [row["evidence_id"] for row in history]
    if history_ids != sorted(history_ids) or len(history_ids) != len(set(history_ids)):
        raise ValueError("history_refs must be sorted and unique by evidence_id")
    if value["colour_regime_id"] is not None:
        raise ValueError("colour_regime_id must remain null until WP5")
    if mode == "production" and value["mask_source"] == "oracle":
        raise ValueError("production evidence rejects oracle masks")

    decision_fields = (value["decision_id"], value["disposition"])
    explanation_fields = (
        value["explanation_id"],
        value["model_revision"],
        value["prompt_sha256"],
        value["guard_result"],
    )
    images = value["image_ids_sent"]
    assert isinstance(images, list)
    if phase == "measured":
        if any(item is not None for item in decision_fields + explanation_fields):
            raise ValueError("measured records cannot contain decision or explanation values")
        if value["limit_codes"] or history or images:
            raise ValueError("measured records require empty limit/history/image lists")
    elif phase == "decided":
        if any(item is None for item in decision_fields):
            raise ValueError("decided records require decision_id and disposition")
        if any(item is not None for item in explanation_fields) or images:
            raise ValueError("decided records cannot contain explanation values")
    else:
        if any(item is None for item in decision_fields + explanation_fields):
            raise ValueError("explained records require decision and explanation values")
        revision = value["model_revision"]
        assert isinstance(revision, str)
        template = revision.startswith("deterministic-template:")
        if template:
            if not revision.removeprefix("deterministic-template:") or images:
                raise ValueError("deterministic-template explanations require a revision and no images")
        elif len(images) not in {4, 6}:
            raise ValueError("visual explanations require exactly 4 or 6 delivered image IDs")
    return value


def validate_evidence_sequence_v1(
    records: object,
    *,
    phase: str,
    mode: str = "storage",
    verified_colour_regime_bindings: object | None = None,
) -> list[dict[str, object]]:
    """Validate one ordered 2--3-stage vial sequence."""

    if verified_colour_regime_bindings is not None:
        raise ValueError("colour regime bindings are unavailable until WP5")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("evidence sequence must be a sequence")
    if len(records) not in {2, 3}:
        raise ValueError("evidence sequence must contain exactly 2 or 3 records")
    checked = [
        validate_evidence_record_v1(record, phase=phase, mode=mode)
        for record in records
    ]
    shared_fields = (
        "run_id",
        "case_id",
        "vial_ordinal",
        "block_recipe_id",
        "block_position",
        "recipe_id",
        "previous_recipe_id",
        "recipe_ml",
        "dispense_order",
        "split_role",
        "split_version",
        "fill_clip_path",
        "fill_frame_range",
        "transition_frame_ranges",
        "clip_sha256",
        "decision_id",
        "disposition",
        "limit_codes",
        "history_refs",
        "explanation_id",
        "model_revision",
        "prompt_sha256",
        "image_ids_sent",
        "guard_result",
    )
    first = checked[0]
    for index, record in enumerate(checked[1:], start=2):
        for field in shared_fields:
            if record[field] != first[field]:
                raise ValueError(
                    f"evidence sequence record {index} disagrees on shared {field}"
                )
    unique_fields = ("stage_id", "image_id", "measurement_id", "evidence_id")
    for field in unique_fields:
        values = [record[field] for record in checked]
        if len(set(values)) != len(values):
            raise ValueError(f"evidence sequence requires unique {field} values")
    if [record["stage_ordinal"] for record in checked] != list(
        range(1, len(checked) + 1)
    ):
        raise ValueError("evidence sequence stage ordinals must be contiguous from one")
    order = first["dispense_order"]
    assert isinstance(order, list)
    if len(checked) != len(order) or [record["tank_id"] for record in checked] != order:
        raise ValueError("evidence sequence tanks must exactly follow dispense_order")
    return checked


def verify_evidence_artifacts(
    record: object,
    artifact_root: str | Path,
    *,
    predicted_mask_registry: object | None = None,
) -> dict[str, object]:
    """Verify exact clip/RGB/mask bytes for storage; this does not trust lineage."""

    reject_nonfinite(record, "evidence record")
    validate_schema(record, _SCHEMA, "evidence record")
    assert isinstance(record, dict)
    value = deepcopy(record)
    _validate_evidence_lexical_fields(value)
    clip_relative = contained_relative_path(value["fill_clip_path"], "fill_clip_path")
    image_id = value["image_id"]
    assert isinstance(image_id, str)
    if value["mask_source"] == "oracle":
        if predicted_mask_registry is not None:
            raise ValueError("oracle evidence cannot use a predicted-mask registry")
        rgb_relative = contained_relative_path(
            f"images/{image_id}.png", "oracle RGB path"
        )
        mask_relative = contained_relative_path(
            f"masks/oracle/{image_id}.png", "oracle mask path"
        )
    else:
        if not isinstance(predicted_mask_registry, Mapping):
            raise ValueError("predicted evidence requires the normalized storage registry")
        for registry_id, raw_locator in predicted_mask_registry.items():
            if not isinstance(registry_id, str) or not isinstance(raw_locator, Mapping):
                raise ValueError("predicted-mask registry entries are invalid")
            safe_posix_component(registry_id, "predicted-mask registry image_id")
            if set(raw_locator) != {
                "rgb_path",
                "rgb_sha256",
                "mask_path",
                "mask_sha256",
            }:
                raise ValueError("predicted-mask registry locators have invalid fields")
            exact_lower_sha256(
                raw_locator["rgb_sha256"], "predicted-mask registry rgb_sha256"
            )
            exact_lower_sha256(
                raw_locator["mask_sha256"], "predicted-mask registry mask_sha256"
            )
        locator = predicted_mask_registry.get(image_id)
        if not isinstance(locator, Mapping):
            raise ValueError("predicted-mask registry is missing image_id")
        if (
            locator["rgb_sha256"] != value["rgb_sha256"]
            or locator["mask_sha256"] != value["mask_sha256"]
        ):
            raise ValueError("predicted-mask registry hashes disagree with evidence")
        rgb_relative = relative_path_below_root(
            artifact_root, locator["rgb_path"], "predicted RGB path"
        )
        mask_relative = relative_path_below_root(
            artifact_root, locator["mask_path"], "predicted mask path"
        )
    for relative, expected, role in (
        (clip_relative, value["clip_sha256"], "fill clip"),
        (rgb_relative, value["rgb_sha256"], "RGB image"),
        (mask_relative, value["mask_sha256"], "mask image"),
    ):
        observed = stable_hash_relative_file(artifact_root, relative, role)
        if observed["sha256"] != expected:
            raise ValueError(f"{role} SHA-256 mismatch")
    return value


def publish_evidence_derivation_v1(
    predecessor_path: str | Path,
    derived_path: str | Path,
    receipt_path: str | Path,
    derived_record: object,
    *,
    predecessor_phase: str,
    derived_phase: str,
    mode: str = "storage",
) -> tuple[Path, Path]:
    """Publish one immutable adjacent-phase record and exact binding receipt."""

    predecessor = Path(predecessor_path)
    derived = Path(derived_path)
    receipt = Path(receipt_path)
    if len({predecessor.absolute(), derived.absolute(), receipt.absolute()}) != 3:
        raise ValueError("derivation paths must be distinct")
    _require_adjacent_phases(predecessor_phase, derived_phase)
    predecessor_raw = stable_read_path(predecessor, "predecessor evidence record")
    predecessor_value = strict_json_bytes(
        predecessor_raw, "predecessor evidence record"
    )
    checked_predecessor = validate_evidence_record_v1(
        predecessor_value, phase=predecessor_phase, mode=mode
    )
    checked_derived = validate_evidence_record_v1(
        derived_record, phase=derived_phase, mode=mode
    )
    _verify_phase_immutability(
        checked_predecessor,
        checked_derived,
        predecessor_phase=predecessor_phase,
        derived_phase=derived_phase,
    )
    derived_raw = pretty_json_bytes(checked_derived)
    receipt_value = {
        "schema_version": "evidence-record-derivation-v1",
        "predecessor_phase": predecessor_phase,
        "derived_phase": derived_phase,
        "predecessor_sha256": sha256_bytes(predecessor_raw),
        "derived_sha256": sha256_bytes(derived_raw),
    }
    receipt_raw = pretty_json_bytes(receipt_value)
    publish_bytes_no_clobber(derived, derived_raw, "derived evidence record")
    publish_bytes_no_clobber(receipt, receipt_raw, "evidence derivation receipt")
    verify_evidence_derivation_v1(
        predecessor,
        derived,
        receipt,
        predecessor_phase=predecessor_phase,
        derived_phase=derived_phase,
        mode=mode,
    )
    return derived, receipt


def verify_evidence_derivation_v1(
    predecessor_path: str | Path,
    derived_path: str | Path,
    receipt_path: str | Path,
    *,
    predecessor_phase: str,
    derived_phase: str,
    mode: str = "storage",
) -> None:
    """Stable-read and verify an exact adjacent-phase derivation."""

    _require_adjacent_phases(predecessor_phase, derived_phase)
    predecessor_raw = stable_read_path(
        predecessor_path, "predecessor evidence record"
    )
    derived_raw = stable_read_path(derived_path, "derived evidence record")
    receipt_raw = stable_read_path(receipt_path, "evidence derivation receipt")
    predecessor_value = strict_json_bytes(
        predecessor_raw, "predecessor evidence record"
    )
    derived_value = strict_json_bytes(derived_raw, "derived evidence record")
    receipt_value = strict_json_bytes(receipt_raw, "evidence derivation receipt")
    if not isinstance(receipt_value, dict) or set(receipt_value) != _DERIVATION_FIELDS:
        raise ValueError("evidence derivation receipt fields are invalid")
    expected = {
        "schema_version": "evidence-record-derivation-v1",
        "predecessor_phase": predecessor_phase,
        "derived_phase": derived_phase,
        "predecessor_sha256": sha256_bytes(predecessor_raw),
        "derived_sha256": sha256_bytes(derived_raw),
    }
    if receipt_value != expected:
        raise ValueError("evidence derivation receipt binding mismatch")
    checked_predecessor = validate_evidence_record_v1(
        predecessor_value, phase=predecessor_phase, mode=mode
    )
    checked_derived = validate_evidence_record_v1(
        derived_value, phase=derived_phase, mode=mode
    )
    _verify_phase_immutability(
        checked_predecessor,
        checked_derived,
        predecessor_phase=predecessor_phase,
        derived_phase=derived_phase,
    )


def _require_adjacent_phases(predecessor_phase: str, derived_phase: str) -> None:
    if (predecessor_phase, derived_phase) not in _ADJACENT_PHASES:
        raise ValueError("evidence derivation phases must be adjacent")


def _verify_phase_immutability(
    predecessor: Mapping[str, object],
    derived: Mapping[str, object],
    *,
    predecessor_phase: str,
    derived_phase: str,
) -> None:
    allowed = (
        _DECISION_FIELDS
        if (predecessor_phase, derived_phase) == ("measured", "decided")
        else _EXPLANATION_FIELDS
    )
    changed = {
        key for key in predecessor if predecessor.get(key) != derived.get(key)
    }
    if not changed or not changed.issubset(allowed):
        raise ValueError("derived evidence changed fields outside its owning phase")
