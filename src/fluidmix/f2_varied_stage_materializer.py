"""Materialize label-free Varied V3 stage-publication requests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from fluidmix.deploy_input_policy import validate_deploy_input_fields
from fluidmix.f2_receipt_manifest_publisher import (
    F2ReceiptPublicationError,
    _varied_previous_ml,
    _varied_programmed_recipe,
)
from fluidmix.render_process import hash_regular_file, load_strict_json_snapshot
from fluidmix.twin_video import _atomic_rename_noreplace, _write_private_json


_RUNTIME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CASE_ID = re.compile(r"CASE_[0-9a-f]{16}\Z")
_SPLIT_VERSION = "whole-run-varied-v3"
_SPLITS = frozenset(("train", "validation", "evaluation"))
_TOP_KEYS = {
    "artifact_type",
    "claim_boundary",
    "data_origin",
    "denominators",
    "model_input_allowlist",
    "real_images_used",
    "schema_version",
    "split_version",
    "status",
    "vial_records",
}
_VIAL_KEYS = {
    "batch_id",
    "case_id",
    "current_programmed_ml",
    "current_recipe_id",
    "data_origin",
    "previous_programmed_ml",
    "previous_recipe_id",
    "programmed_dispense_order",
    "quantitative_evidence_ids",
    "run_id",
    "split_role",
    "split_version",
    "stage_image_evidence_ids",
    "vial_position",
}


class F2VariedStageMaterializationError(ValueError):
    """A frozen Varied V3 input plan could not be materialized safely."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise F2VariedStageMaterializationError(f"{field}_invalid")
    return value


def _load_plan(path: str | os.PathLike[str]) -> tuple[list[dict[str, object]], str]:
    try:
        payload, snapshot = load_strict_json_snapshot(Path(path).expanduser().absolute())
    except (OSError, ValueError) as error:
        raise F2VariedStageMaterializationError("model_input_plan_invalid") from error
    if not isinstance(payload, Mapping) or set(payload) != _TOP_KEYS:
        raise F2VariedStageMaterializationError("model_input_plan_schema_invalid")
    if (
        payload.get("schema_version") != 3
        or payload.get("artifact_type") != "synthetic_varied_run_input_plan_v3"
        or payload.get("data_origin") != "twin"
        or payload.get("real_images_used") is not False
        or payload.get("split_version") != _SPLIT_VERSION
        or payload.get("status") != "unrendered"
    ):
        raise F2VariedStageMaterializationError("model_input_plan_schema_invalid")
    allowlist = payload.get("model_input_allowlist")
    denominators = payload.get("denominators")
    records = payload.get("vial_records")
    if (
        not isinstance(allowlist, list)
        or len(allowlist) != len(_VIAL_KEYS)
        or set(allowlist) != _VIAL_KEYS
        or not isinstance(denominators, Mapping)
        or denominators.get("vials") != 18
        or denominators.get("runs") != 6
        or not isinstance(records, list)
        or len(records) != 18
    ):
        raise F2VariedStageMaterializationError("model_input_plan_denominator_invalid")

    normalized: list[dict[str, object]] = []
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != _VIAL_KEYS:
            raise F2VariedStageMaterializationError("vial_record_schema_invalid")
        try:
            validate_deploy_input_fields(raw)
        except ValueError as error:
            raise F2VariedStageMaterializationError(
                "vial_record_forbidden_field"
            ) from error
        position = raw.get("vial_position")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position not in {1, 2, 3}
            or raw.get("data_origin") != "twin"
            or raw.get("split_role") not in _SPLITS
            or raw.get("split_version") != _SPLIT_VERSION
            or raw.get("quantitative_evidence_ids") != []
            or raw.get("stage_image_evidence_ids") != []
        ):
            raise F2VariedStageMaterializationError("vial_record_schema_invalid")
        try:
            current = _varied_programmed_recipe(
                {
                    "dispense_order": raw.get("programmed_dispense_order"),
                    "current_ml_by_tank": raw.get("current_programmed_ml"),
                }
            )
            previous = _varied_previous_ml(raw.get("previous_programmed_ml"))
        except F2ReceiptPublicationError as error:
            raise F2VariedStageMaterializationError(
                "vial_record_recipe_invalid"
            ) from error
        case_id = _text(raw.get("case_id"), "case_id")
        if _CASE_ID.fullmatch(case_id) is None:
            raise F2VariedStageMaterializationError("case_id_invalid")
        normalized.append(
            {
                "batch_id": _text(raw.get("batch_id"), "batch_id"),
                "case_id": case_id,
                "current_programmed_ml": current["current_ml_by_tank"],
                "current_recipe_id": _text(
                    raw.get("current_recipe_id"), "current_recipe_id"
                ),
                "previous_programmed_ml": previous,
                "previous_recipe_id": _text(
                    raw.get("previous_recipe_id"), "previous_recipe_id"
                ),
                "programmed_dispense_order": current["dispense_order"],
                "run_id": _text(raw.get("run_id"), "run_id"),
                "split_role": str(raw["split_role"]),
                "vial_position": position,
            }
        )
    case_ids = [str(row["case_id"]) for row in normalized]
    if len(set(case_ids)) != len(case_ids):
        raise F2VariedStageMaterializationError("case_id_duplicate")
    return normalized, str(snapshot["sha256"])


def _cohort_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path).expanduser().absolute()
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise F2VariedStageMaterializationError("cohort_root_missing") from error
    if resolved != root or not root.is_dir():
        raise F2VariedStageMaterializationError("cohort_root_invalid")
    return root


def _receipt(root: Path, case_id: str) -> tuple[str, str]:
    if _CASE_ID.fullmatch(case_id) is None:
        raise F2VariedStageMaterializationError("case_id_invalid")
    cases_root = (root / "cases").absolute()
    path = (cases_root / case_id / "app-output" / "receipt.json").absolute()
    if path.parents[2] != cases_root:
        raise F2VariedStageMaterializationError(f"case_receipt_invalid:{case_id}")
    try:
        resolved_cases = cases_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        snapshot = hash_regular_file(path)
    except (OSError, ValueError) as error:
        raise F2VariedStageMaterializationError(
            f"case_receipt_missing:{case_id}"
        ) from error
    if resolved_cases != cases_root or resolved != path or not path.is_file():
        raise F2VariedStageMaterializationError(f"case_receipt_invalid:{case_id}")
    return str(path), str(snapshot["sha256"])


def materialize_varied_publish_requests(
    *,
    model_input_plan: str | os.PathLike[str],
    cohort_root: str | os.PathLike[str],
    runtime_id: str,
    output_dir: str | os.PathLike[str],
) -> Path:
    """Write six hash-bound V3 publication requests from the safe input plan."""

    if not isinstance(runtime_id, str) or _RUNTIME_ID.fullmatch(runtime_id) is None:
        raise F2VariedStageMaterializationError("runtime_id_invalid")
    records, plan_sha = _load_plan(model_input_plan)
    root = _cohort_root(cohort_root)
    groups: dict[str, list[dict[str, object]]] = {}
    for row in records:
        groups.setdefault(str(row["batch_id"]), []).append(row)
    if len(groups) != 6:
        raise F2VariedStageMaterializationError("batch_denominator_invalid")

    requests: dict[str, dict[str, object]] = {}
    request_rows: list[dict[str, object]] = []
    run_ids: set[str] = set()
    split_vial_counts: Counter[str] = Counter()
    stage_pair_count = 0
    seen_receipt_paths: set[str] = set()
    seen_receipt_hashes: set[str] = set()
    for batch_id in sorted(groups):
        rows = sorted(groups[batch_id], key=lambda row: int(row["vial_position"]))
        if [row["vial_position"] for row in rows] != [1, 2, 3]:
            raise F2VariedStageMaterializationError("batch_vial_positions_invalid")
        shared = (
            "run_id",
            "current_recipe_id",
            "current_programmed_ml",
            "programmed_dispense_order",
            "split_role",
        )
        if any(
            any(row[field] != rows[0][field] for field in shared)
            for row in rows[1:]
        ):
            raise F2VariedStageMaterializationError("batch_context_mismatch")
        run_id = str(rows[0]["run_id"])
        if run_id in run_ids:
            raise F2VariedStageMaterializationError("run_reused_across_batches")
        run_ids.add(run_id)
        split_role = str(rows[0]["split_role"])
        split_vial_counts[split_role] += 3
        stage_count = len(rows[0]["programmed_dispense_order"])
        stage_pair_count += 3 * stage_count
        vials: list[dict[str, object]] = []
        for row in rows:
            receipt_path, receipt_sha = _receipt(root, str(row["case_id"]))
            if receipt_path in seen_receipt_paths or receipt_sha in seen_receipt_hashes:
                raise F2VariedStageMaterializationError("case_receipt_reused")
            seen_receipt_paths.add(receipt_path)
            seen_receipt_hashes.add(receipt_sha)
            position = int(row["vial_position"])
            vials.append(
                {
                    "case_id": row["case_id"],
                    "vial_id": f"V{position}",
                    "vial_position": position,
                    "previous_recipe_id": row["previous_recipe_id"],
                    "previous_ml_by_tank": row["previous_programmed_ml"],
                    "receipt_path": receipt_path,
                    "receipt_sha256": receipt_sha,
                }
            )
        request = {
            "schema_version": 3,
            "artifact_type": "f2_varied_stage_batch_publish_request",
            "data_origin": "twin",
            "provider": "brev",
            "runtime_name": "NVIDIA Isaac Sim",
            "runtime_id": runtime_id,
            "batch_id": batch_id,
            "run_id": run_id,
            "cohort": "scored_target",
            "current_recipe_id": rows[0]["current_recipe_id"],
            "split_role": split_role,
            "split_version": _SPLIT_VERSION,
            "programmed_recipe": {
                "dispense_order": rows[0]["programmed_dispense_order"],
                "current_ml_by_tank": rows[0]["current_programmed_ml"],
            },
            "vials": vials,
        }
        try:
            validate_deploy_input_fields(request, require_allowlist=False)
        except ValueError as error:
            raise F2VariedStageMaterializationError(
                "publish_request_forbidden_field"
            ) from error
        name = f"publish-request-{batch_id}.json"
        requests[name] = request
        request_rows.append(
            {
                "path": name,
                "batch_id": batch_id,
                "run_id": run_id,
                "split_role": split_role,
                "active_stage_count": stage_count,
                "stage_pair_count": 3 * stage_count,
            }
        )
    if (
        len(run_ids) != 6
        or dict(sorted(split_vial_counts.items()))
        != {"evaluation": 3, "train": 12, "validation": 3}
        or stage_pair_count != 42
    ):
        raise F2VariedStageMaterializationError("cohort_denominator_invalid")

    output = Path(output_dir).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    if os.path.lexists(output):
        raise FileExistsError(output)
    partial: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=parent)
    )
    os.chmod(partial, 0o700)
    try:
        for row in request_rows:
            name = str(row["path"])
            _write_private_json(partial / name, requests[name])
            row.update(hash_regular_file(partial / name))
        bundle = {
            "schema_version": 3,
            "artifact_type": "f2_varied_stage_publish_request_bundle",
            "status": "materialized",
            "source_model_input_plan_sha256": plan_sha,
            "runtime_id": runtime_id,
            "batch_count": 6,
            "vial_receipt_count": 18,
            "stage_pair_count": 42,
            "split_vial_counts": dict(sorted(split_vial_counts.items())),
            "requests": request_rows,
            "claim_boundary": (
                "synthetic artifact routing only; particle-height warnings are not "
                "particle-physics passes and no sim-to-real claim is made"
            ),
        }
        receipt_name = "request-materialization-receipt.json"
        _write_private_json(partial / receipt_name, bundle)
        directory_fd = os.open(partial, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _atomic_rename_noreplace(partial, output)
        partial = None
        return output / receipt_name
    finally:
        if partial is not None and partial.exists():
            metadata = os.lstat(partial)
            if stat.S_ISDIR(metadata.st_mode) and partial.parent == parent:
                shutil.rmtree(partial)
