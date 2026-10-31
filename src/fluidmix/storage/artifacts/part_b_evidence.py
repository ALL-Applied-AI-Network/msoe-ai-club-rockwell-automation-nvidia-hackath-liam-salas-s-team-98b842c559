"""Production-only projection for trusted flat Part B evidence records."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import hashlib
import json

from ...core.contracts.evidence import (
    validate_evidence_record_v1,
    validate_evidence_sequence_v1,
)
from ...core.policy import validate_deploy_input_fields
from ...perception.masks import (
    is_trusted_predicted_mask_registry_v1,
    is_trusted_real_rgb_mask_registry_v1,
)
from ...measurement.vision_measurement import (
    is_trusted_predicted_measurement_binding_v1,
    trusted_measurement_bindings_from_result_v1,
    trusted_prediction_registry_from_result_v1,
)


class PartBInsufficientEvidence(ValueError):
    """A production Part B input could not establish trusted usable evidence."""

    disposition = "insufficient_evidence"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"insufficient_evidence: {reason}")


def _insufficient(reason: str) -> PartBInsufficientEvidence:
    return PartBInsufficientEvidence(reason)


def prohibit_legacy_diagnostic_active_path(entry_point: str) -> None:
    """Fail closed when an active path tries to use frozen diagnostic serializers."""

    raise RuntimeError(
        "legacy_diagnostic_only: "
        f"{entry_point} is deactivated; use project_part_b_model_evidence_v1"
    )


def bind_part_b_measurements_from_adapter_v1(
    record_templates: object,
    trusted_adapter_result: object,
) -> list[dict[str, object]]:
    """Fill measurement-owned fields from a live public adapter result."""

    try:
        bindings = trusted_measurement_bindings_from_result_v1(trusted_adapter_result)
        registry = trusted_prediction_registry_from_result_v1(trusted_adapter_result)
    except TypeError as error:
        raise _insufficient("live_trusted_adapter_result_missing") from error
    if isinstance(record_templates, (str, bytes)) or not isinstance(
        record_templates, Sequence
    ):
        raise _insufficient("part_b_record_templates_invalid")
    records = deepcopy(list(record_templates))
    by_image = {binding.image_id: binding for binding in bindings}
    if len(by_image) != len(bindings):
        raise _insufficient("duplicate_measurement_binding")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("image_id"), str):
            raise _insufficient("part_b_record_template_invalid")
        binding = by_image.get(record["image_id"])
        if binding is None:
            raise _insufficient("trusted_measurement_binding_missing")
        if binding.validity != "valid":
            reason = binding.limit_codes[0] if binding.limit_codes else "mask_unusable"
            raise _insufficient(f"predicted_measurement_unusable:{reason}")
        entry = registry.entry(binding.image_id)
        record.update(
            {
                "rgb_sha256": entry.rgb_sha256,
                "mask_sha256": entry.mask_sha256,
                "mask_source": "predicted",
                "measurement_id": binding.measurement_id,
                "evidence_id": binding.evidence_id,
                "median_lab": list(binding.median_lab),
                "mad_lab": list(binding.mad_lab),
                "iqr_lab": list(binding.iqr_lab),
            }
        )
    return records


def project_part_b_model_evidence_v1(
    records: object,
    *,
    phase: str,
    predicted_mask_registry: object,
    measurement_bindings: object,
    verified_history_bindings: object | None = None,
) -> str:
    """Return canonical JSON for one trusted production vial sequence.

    Locator registries and measurement/artifact receipts are verifier inputs only;
    they are intentionally absent from the emitted flat records.
    """

    try:
        checked = validate_evidence_sequence_v1(
            records,
            phase=phase,
            mode="production",
        )
    except (TypeError, ValueError) as error:
        raise _insufficient(f"production_record_invalid:{error}") from error
    try:
        for record in checked:
            validate_deploy_input_fields(
                record,
                envelope="part_b_evidence_v1",
            )
    except ValueError as error:
        raise _insufficient(f"production_envelope_invalid:{error}") from error

    try:
        predicted_mask_registry = trusted_prediction_registry_from_result_v1(
            predicted_mask_registry
        )
    except TypeError:
        pass
    if not (
        is_trusted_predicted_mask_registry_v1(predicted_mask_registry)
        or is_trusted_real_rgb_mask_registry_v1(predicted_mask_registry)
    ):
        raise _insufficient("trusted_predicted_mask_registry_missing")
    adapter_result = False
    try:
        measurement_bindings = trusted_measurement_bindings_from_result_v1(
            measurement_bindings
        )
        adapter_result = True
    except TypeError:
        pass
    if isinstance(measurement_bindings, (str, bytes)) or not isinstance(
        measurement_bindings, Sequence
    ):
        raise _insufficient("trusted_measurement_bindings_missing")
    if not adapter_result and len(measurement_bindings) != len(checked):
        raise _insufficient("trusted_measurement_binding_count_mismatch")
    if any(record["history_refs"] for record in checked):
        # Retrieval has no reviewed trusted constructor in INN-58. A caller dict
        # cannot mint history provenance; later WP5/WP7 work must supply it.
        raise _insufficient("trusted_history_binding_unavailable")
    if verified_history_bindings is not None:
        raise _insufficient("untrusted_history_binding")

    bindings_by_image = {}
    for binding in measurement_bindings:
        if not is_trusted_predicted_measurement_binding_v1(binding):
            raise _insufficient("untrusted_measurement_binding")
        if binding.image_id in bindings_by_image:
            raise _insufficient("duplicate_measurement_binding")
        bindings_by_image[binding.image_id] = binding

    for record in checked:
        image_id = record["image_id"]
        if record["mask_source"] != "predicted":
            raise _insufficient("oracle_mask_forbidden")
        try:
            entry = predicted_mask_registry.entry(image_id)
        except ValueError as error:
            raise _insufficient("trusted_predicted_mask_entry_missing") from error
        binding = bindings_by_image.get(image_id)
        if binding is None:
            raise _insufficient("trusted_measurement_binding_missing")
        if binding.validity != "valid":
            reason = binding.limit_codes[0] if binding.limit_codes else "mask_unusable"
            raise _insufficient(f"predicted_measurement_unusable:{reason}")
        if (
            record["rgb_sha256"] != entry.rgb_sha256
            or record["mask_sha256"] != entry.mask_sha256
            or binding.rgb_sha256 != entry.rgb_sha256
            or binding.mask_sha256 != entry.mask_sha256
        ):
            raise _insufficient("predicted_artifact_hash_mismatch")
        if (
            binding.image_id != image_id
            or binding.measurement_id != record["measurement_id"]
            or binding.evidence_id != record["evidence_id"]
            or binding.entry_sha256 != entry.entry_sha256
            or binding.registry_sha256 != predicted_mask_registry.registry_sha256
        ):
            raise _insufficient("predicted_registry_evidence_binding_mismatch")
        if (
            binding.median_lab is None
            or binding.mad_lab is None
            or binding.iqr_lab is None
            or record["median_lab"] != list(binding.median_lab)
            or record["mad_lab"] != list(binding.mad_lab)
            or record["iqr_lab"] != list(binding.iqr_lab)
        ):
            raise _insufficient("measurement_content_mismatch")
        measurement_payload = {
            "image_id": image_id,
            "measurement_id": record["measurement_id"],
            "evidence_id": record["evidence_id"],
            "rgb_sha256": record["rgb_sha256"],
            "mask_sha256": record["mask_sha256"],
            "validity": binding.validity,
            "limit_codes": list(binding.limit_codes),
            "median_lab": record["median_lab"],
            "mad_lab": record["mad_lab"],
            "iqr_lab": record["iqr_lab"],
        }
        observed_measurement_sha256 = hashlib.sha256(
            json.dumps(
                measurement_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if observed_measurement_sha256 != binding.measurement_sha256:
            raise _insufficient("measurement_content_mismatch")

    # This is intentionally adjacent to serialization: no unvalidated object is
    # formatted into a production prompt after the sequence/trust gates above.
    try:
        serializable = [
            validate_evidence_record_v1(record, phase=phase, mode="production")
            for record in checked
        ]
        for record in serializable:
            validate_deploy_input_fields(
                record,
                envelope="part_b_evidence_v1",
            )
        return json.dumps(
            serializable,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise _insufficient(f"production_prompt_record_invalid:{error}") from error
