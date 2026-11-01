"""CPU-only diagnostic bundle for the Varied V3 evidence path."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil

from fluidmix.core.policy import validate_deploy_input_fields
from fluidmix.explanation.template.varied_explanation import (
    parse_varied_response,
    structure_varied_response,
    varied_reference_response,
)


class VariedDemoError(ValueError):
    """A Varied V3 demo input failed the frozen diagnostic boundary."""


_INPUT_FORBIDDEN = {
    "expected_disposition",
    "label",
    "labels",
    "oracle",
    "palette",
    "residual_volume",
    "tank_palette",
}
_INPUT_FORBIDDEN_FRAGMENTS = ("label", "oracle", "palette", "residual")
_LOCATOR_FIELDS = {
    "batch_id",
    "mask_path",
    "mask_sha256",
    "recipe_id",
    "rgb_path",
    "rgb_sha256",
    "run_id",
    "source_manifest_sha256",
    "split_role",
    "split_version",
    "stage_id",
    "stage_ordinal",
    "tank_id",
    "vial_id",
}
_HASH_FIELDS = ("rgb_sha256", "mask_sha256", "source_manifest_sha256")
_MASK_LIMIT = "rgb_only_predicted_liquid_mask_missing"


def _reject_constant(value: str) -> object:
    raise VariedDemoError(f"nonfinite_json_constant:{value}")


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VariedDemoError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _load(
    path_like: str | os.PathLike[str], role: str
) -> tuple[dict[str, object], Path, str]:
    try:
        path = Path(path_like).expanduser().resolve(strict=True)
        if not path.is_file():
            raise OSError("not a regular file")
        data = path.read_bytes()
        value = json.loads(
            data,
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, VariedDemoError) as error:
        if isinstance(error, VariedDemoError):
            raise
        raise VariedDemoError(f"{role}_invalid:{error}") from error
    if not isinstance(value, dict):
        raise VariedDemoError(f"{role}_invalid")
    return value, path, hashlib.sha256(data).hexdigest()


def _reject_hidden_fields(value: object) -> None:
    try:
        validate_deploy_input_fields(value, require_allowlist=False)
    except ValueError as error:
        raise VariedDemoError(f"forbidden_input_field:{error}") from error

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = key.lower() if isinstance(key, str) else ""
                if (
                    not isinstance(key, str)
                    or normalized in _INPUT_FORBIDDEN
                    or any(
                        fragment in normalized
                        for fragment in _INPUT_FORBIDDEN_FRAGMENTS
                    )
                ):
                    raise VariedDemoError(f"forbidden_input_field:{key}")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)


def _header(
    value: Mapping[str, object], expected: Mapping[str, object], role: str
) -> None:
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise VariedDemoError(f"{role}_header_invalid")


def _hex(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise VariedDemoError(f"{role}_invalid")
    return value


def _decision_map(
    value: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    _header(
        value,
        {
            "schema_version": 1,
            "artifact_type": "f2_varied_triad_decisions",
            "status": "assessed",
            "data_origin": "twin",
            "real_data_used": False,
            "split_version": "whole-run-varied-v3",
        },
        "decision",
    )
    rows = value.get("decision_records")
    denominators = value.get("denominators")
    if (
        not isinstance(rows, list)
        or len(rows) != 18
        or not isinstance(denominators, Mapping)
    ):
        raise VariedDemoError("decision_denominator_invalid")
    expected_denominators = {
        "runs": 6,
        "vial_decisions": 18,
        "keep": 0,
        "discard": 0,
        "insufficient_evidence": 18,
    }
    if any(
        denominators.get(key) != expected
        for key, expected in expected_denominators.items()
    ):
        raise VariedDemoError("decision_denominator_invalid")
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise VariedDemoError("decision_record_invalid")
        try:
            varied_reference_response(row)
        except (TypeError, ValueError) as error:
            raise VariedDemoError("decision_record_invalid") from error
        identity = row.get("decision_id")
        if not isinstance(identity, str) or not identity or identity in result:
            raise VariedDemoError("decision_identity_invalid")
        if row.get("disposition") != "insufficient_evidence":
            raise VariedDemoError("non_abstaining_decision_forbidden")
        result[identity] = row
    return result


def _expected_images(
    decisions: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    expected: dict[str, dict[str, object]] = {}
    for decision in decisions.values():
        stages = decision.get("stages")
        assert isinstance(stages, list)
        for stage in stages:
            assert isinstance(stage, Mapping)
            image_id = stage.get("target_image_id")
            if not isinstance(image_id, str) or image_id in expected:
                raise VariedDemoError("target_image_identity_invalid")
            expected[image_id] = {
                "batch_id": decision["batch_id"],
                "run_id": decision["run_id"],
                "recipe_id": decision["recipe_id"],
                "split_role": decision["split_role"],
                "split_version": decision["split_version"],
                "vial_id": decision["vial_id"],
                "stage_id": stage["stage_id"],
                "stage_ordinal": stage["stage_ordinal"],
                "tank_id": stage["tank_id"],
            }
    if len(expected) != 42:
        raise VariedDemoError("image_denominator_invalid")
    return expected


def _verify_png(locator: Mapping[str, object], role: str) -> None:
    from PIL import Image, UnidentifiedImageError
    import numpy as np

    path_text = locator.get(f"{role}_path")
    digest = _hex(locator.get(f"{role}_sha256"), f"{role}_sha256")
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise VariedDemoError(f"{role}_path_invalid")
    path = Path(path_text)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise VariedDemoError(f"{role}_missing") from error
    if hashlib.sha256(data).hexdigest() != digest:
        raise VariedDemoError(f"{role}_hash_mismatch")
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.size != (640, 480):
                raise VariedDemoError(f"{role}_png_invalid")
            if role == "rgb":
                if image.mode != "RGB":
                    raise VariedDemoError("rgb_png_invalid")
            else:
                if image.mode not in {"L", "1"}:
                    raise VariedDemoError("mask_png_invalid")
                values = set(
                    int(item)
                    for item in np.unique(np.asarray(image.convert("L")))
                )
                if not values.issubset({0, 255}) or 255 not in values:
                    raise VariedDemoError("mask_png_invalid")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        if isinstance(error, VariedDemoError):
            raise
        raise VariedDemoError(f"{role}_png_invalid") from error


def _locator_map(
    value: Mapping[str, object],
    expected: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    _header(
        value,
        {
            "schema_version": 1,
            "artifact_type": "f2_varied_image_locators",
            "status": "published",
            "data_origin": "twin",
            "real_data_used": False,
            "split_version": "whole-run-varied-v3",
        },
        "locator",
    )
    images = value.get("images")
    denominators = value.get("denominators")
    if (
        not isinstance(images, Mapping)
        or set(images) != set(expected)
        or not isinstance(denominators, Mapping)
        or denominators.get("images") != 42
    ):
        raise VariedDemoError("image_denominator_invalid")
    result: dict[str, Mapping[str, object]] = {}
    for image_id in sorted(expected):
        locator = images[image_id]
        if not isinstance(locator, Mapping) or set(locator) != _LOCATOR_FIELDS:
            raise VariedDemoError("image_locator_invalid")
        if any(
            locator.get(field) != wanted
            for field, wanted in expected[image_id].items()
        ):
            raise VariedDemoError("image_locator_context_drift")
        for field in _HASH_FIELDS:
            _hex(locator.get(field), field)
        _verify_png(locator, "rgb")
        _verify_png(locator, "mask")
        result[image_id] = locator
    return result


def _template_records(
    value: Mapping[str, object],
    decisions: Mapping[str, Mapping[str, object]],
    decision_sha256: str,
) -> dict[str, Mapping[str, object]]:
    _header(
        value,
        {
            "schema_version": 1,
            "artifact_type": "f2_varied_cosmos_nano_score",
            "status": "scored",
        },
        "score",
    )
    if value.get("decision_artifact_sha256") != decision_sha256:
        raise VariedDemoError("score_decision_hash_drift")
    score = value.get("score")
    if not isinstance(score, Mapping):
        raise VariedDemoError("score_invalid")
    metrics = score.get("metrics")
    if not isinstance(metrics, Mapping):
        raise VariedDemoError("score_invalid")
    for metric in (
        "lenient_parse_control",
        "template_control_content_policy",
        "required_abstention_control",
    ):
        if metrics.get(metric) != {"passed": 18, "total": 18}:
            raise VariedDemoError("template_metric_not_passing")
    rows = score.get("structured_records")
    if not isinstance(rows, list):
        raise VariedDemoError("template_record_invalid")
    if any(
        isinstance(row, Mapping) and row.get("kind") == "model" for row in rows
    ):
        raise VariedDemoError("model_structured_record_forbidden")
    if len(rows) != 18:
        raise VariedDemoError("template_record_denominator_invalid")
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("condition_id") != "deterministic_template"
            or row.get("kind") != "control"
        ):
            raise VariedDemoError("template_record_invalid")
        decision_id = row.get("decision_id")
        if (
            not isinstance(decision_id, str)
            or decision_id not in decisions
            or decision_id in result
        ):
            raise VariedDemoError("template_record_identity_invalid")
        raw = varied_reference_response(decisions[decision_id])
        expected = structure_varied_response(
            raw,
            parse_varied_response(raw),
            decisions[decision_id],
        )
        if row.get("record") != expected:
            raise VariedDemoError("template_record_drift")
        result[decision_id] = row
    if set(result) != set(decisions):
        raise VariedDemoError("template_record_identity_invalid")
    return result


def _operator_artifact(
    decisions: Mapping[str, Mapping[str, object]],
    templates: Mapping[str, Mapping[str, object]],
    source_artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for decision_id in sorted(decisions):
        decision = decisions[decision_id]
        raw = varied_reference_response(decision)
        rows.append(
            {
                "decision_id": decision_id,
                "context": {
                    field: copy.deepcopy(decision[field])
                    for field in (
                        "batch_id",
                        "run_id",
                        "recipe_id",
                        "split_role",
                        "split_version",
                        "vial_id",
                        "decision_timing",
                    )
                },
                "disposition": "insufficient_evidence",
                "relative_assessment": decision["relative_assessment"],
                "limit_codes": [*decision["limit_codes"], _MASK_LIMIT],
                "stage_evidence": copy.deepcopy(decision["stages"]),
                "deterministic_explanation": {
                    "response_raw": raw,
                    "structured_record": copy.deepcopy(
                        templates[decision_id]["record"]
                    ),
                },
                "deployment_eligible": False,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "f2_varied_diagnostic_operator_records",
        "status": "diagnostic_only",
        "data_origin": "twin",
        "real_data_used": False,
        "deployment_eligible": False,
        "mask_source": "renderer_semantic_mask_oracle_diagnostic",
        "limit_codes": [_MASK_LIMIT],
        "source_artifacts": copy.deepcopy(source_artifacts),
        "operator_records": rows,
        "denominators": {
            "decisions": 18,
            "images": 42,
            "insufficient_evidence": 18,
        },
        "claim_boundary": (
            "Synthetic FillCam review only. Renderer semantic masks are oracle diagnostics, "
            "not RGB-only predictions. No keep/discard detector, physical carryover, cause, "
            "plant action, or sim-to-real claim is authorized."
        ),
    }


def _review_markdown(
    operator: Mapping[str, object],
    locators: Mapping[str, Mapping[str, object]],
) -> str:
    lines = [
        "# Varied V3 synthetic evidence review",
        "",
        "Deployment eligible: **no**.",
        "",
        "All 18 vials return `insufficient_evidence`. The liquid masks shown here are aligned "
        "Isaac renderer semantic masks, not RGB-only model predictions.",
        "",
    ]
    records = operator["operator_records"]
    assert isinstance(records, list)
    for row in records:
        assert isinstance(row, Mapping)
        context = row["context"]
        assert isinstance(context, Mapping)
        lines.extend(
            [
                f"## {row['decision_id']}",
                "",
                f"Run `{context['run_id']}`, batch `{context['batch_id']}`, vial "
                f"`{context['vial_id']}`, split `{context['split_role']}`.",
                "",
                f"Disposition: **{row['disposition']}**. Relative diagnostic: "
                f"`{row['relative_assessment']}`.",
                "",
                "| Stage | Target vs peer mode dE76 | Peer-pair dE76 | "
                "Threshold dE76 | Assessment |",
                "|---|---:|---:|---:|---|",
            ]
        )
        stages = row["stage_evidence"]
        assert isinstance(stages, list)
        for stage in stages:
            assert isinstance(stage, Mapping)
            lines.append(
                f"| {stage['stage_ordinal']} / {stage['tank_id']} | "
                f"{float(stage['target_delta_e76']):.6f} | "
                f"{float(stage['peer_pair_delta_e76']):.6f} | "
                f"{float(stage['threshold_delta_e76']):.6f} | "
                f"`{stage['relative_status']}` |"
            )
        lines.append("")
        for stage in stages:
            assert isinstance(stage, Mapping)
            image_ids = [stage["target_image_id"], *stage["peer_image_ids"]]
            captions = ["target", "peer 1", "peer 2"]
            lines.append(f"Stage {stage['stage_ordinal']} (`{stage['tank_id']}`):")
            lines.append("")
            for caption, image_id in zip(captions, image_ids, strict=True):
                locator = locators[str(image_id)]
                lines.append(f"- {caption}: `{image_id}`")
                lines.append(
                    f"  ![{caption} {image_id}]({locator['rgb_path']})"
                )
            lines.append("")
        explanation = row["deterministic_explanation"]
        assert isinstance(explanation, Mapping)
        lines.extend(
            [
                "Deterministic explanation:",
                "",
                "```text",
                str(explanation["response_raw"]),
                "```",
                "",
            ]
        )
    lines.extend(["## Claim boundary", "", str(operator["claim_boundary"]), ""])
    return "\n".join(lines)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_file(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a raced destination."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise VariedDemoError("atomic_no_clobber_unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise VariedDemoError("destination_exists")
    raise OSError(error_number, os.strerror(error_number), destination)


def build_varied_demo_bundle(
    decisions_path: str | os.PathLike[str],
    image_locators_path: str | os.PathLike[str],
    score_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
) -> dict[str, object]:
    """Validate frozen evidence and atomically publish one diagnostic review bundle."""

    output = Path(output_directory).expanduser().absolute()
    if os.path.lexists(output):
        raise VariedDemoError("destination_exists")
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as error:
        raise VariedDemoError("destination_parent_invalid") from error
    if not parent.is_dir():
        raise VariedDemoError("destination_parent_invalid")

    decisions_value, decisions_file, decisions_sha = _load(
        decisions_path, "decisions"
    )
    locators_value, locators_file, locators_sha = _load(
        image_locators_path, "locators"
    )
    score_value, score_file, score_sha = _load(score_path, "score")
    for value in (decisions_value, locators_value, score_value):
        _reject_hidden_fields(value)
    decisions = _decision_map(decisions_value)
    expected_images = _expected_images(decisions)
    locators = _locator_map(locators_value, expected_images)
    templates = _template_records(score_value, decisions, decisions_sha)
    sources = {
        "decisions": {"path": str(decisions_file), "sha256": decisions_sha},
        "image_locators": {
            "path": str(locators_file),
            "sha256": locators_sha,
        },
        "cosmos_score": {"path": str(score_file), "sha256": score_sha},
    }
    operator = _operator_artifact(decisions, templates, sources)
    review = _review_markdown(operator, locators)
    operator_bytes = _json_bytes(operator)
    review_bytes = review.encode("utf-8")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "f2_varied_diagnostic_demo_receipt",
        "status": "pass",
        "deployment_eligible": False,
        "source_artifacts": sources,
        "outputs": {
            "operator-records.json": hashlib.sha256(operator_bytes).hexdigest(),
            "REVIEW.md": hashlib.sha256(review_bytes).hexdigest(),
        },
        "denominators": {"decisions": 18, "images": 42},
        "checks": {
            "source_hashes_verified": 3,
            "rgb_mask_pairs_rehashed": 42,
            "deterministic_template_records_recomputed": 18,
            "all_decisions_abstained": True,
            "renderer_semantic_oracle_declared": True,
        },
        "claim_boundary": operator["claim_boundary"],
    }
    receipt_bytes = _json_bytes(receipt)
    temporary = parent / f".{output.name}.tmp-{secrets.token_hex(8)}"
    try:
        os.mkdir(temporary, 0o700)
        _write_file(temporary / "operator-records.json", operator_bytes)
        _write_file(temporary / "REVIEW.md", review_bytes)
        _write_file(temporary / "receipt.json", receipt_bytes)
        descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _rename_directory_no_replace(temporary, output)
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(error, VariedDemoError):
            raise
        raise VariedDemoError(f"publication_failed:{error}") from error
    return receipt
