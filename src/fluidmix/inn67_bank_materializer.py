"""CPU-only INN-67 post-render reference-bank materialization.

This boundary reads JSON receipts and an already-populated evidence-store export. It
does not decode images, execute a predictor, launch a renderer, or open test/video-5
artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil


PLAN_SHA256 = "f956b6d1a4b3f14fd57df1004594c330b5322f8ff5c32afe00c1aef895719b79"
PLAN_INVENTORY_SHA256 = "0b373388c040d0d01d6b8eb45f353b8cbfdeba5b5608f02a9ec5eda09f5e7f95"
PROFILE_SHA256 = "47bc0b6f483a121a28871b3eb74ec27efb79996b83d03d8c45d6f5b0b9cdeb7c"
FREEZE_PROTOCOL_SHA256 = "8b875c93f2d9d61871dd03a187ada847239b3c2e608a18f167ceb2d1646431d6"
LEAKAGE_SHA256 = "26b850654e326d46d46ec65bfe74e4356270e7a84608b4576b86e4bc9bdc4c0e"
NO_ACCESS_SHA256 = "0ccd7d31632463ceb9c63d6103616846ad28c94d1aeff98af88bceb5b4db3731"
AMENDMENT_FILE_SHA256 = "230ca2c7dbfef1023d9e314fc40b5da70b6b1bdd59e184d1642540213f55d9a8"
AMENDMENT_RECEIPT_SHA256 = "3cc4212f84a6a011b0d3dd579cb6cf817285d3a946598908756f6c95c9f349db"
MATERIALIZATION_AMENDMENT_SHA256 = "15604a5251ae920c140ef5068f7057dc2ab4f1e7f1e2b7946408eef05e09d61c"
ACCEPTED_OCCLUSION_VALIDATOR_SHA256 = "3780a83579cf807cf8f4041bdfc931f8b98c374ae8f7eaea386be04ed6ac2315"

DONOR_BATCHES = ("INN67-DEV-RB-B01", "INN67-DEV-RB-B02")
CONFIGURATION_BATCH = "INN67-DEV-RB-B03"
TRIAD_CONFIG_ID = "v3-relative-diagnostic-inherited-stage-v1"
TIGHT_PEER_PAIR_MAX_DELTA_E76 = 2.0
TARGET_OUTLIER_MARGIN_DELTA_E76 = 1.0

_VALIDATION_RECEIPT_FIELDS = {
    "schema_version", "artifact_type", "status", "failures", "case_id",
    "source_render_receipt_sha256", "source_render_plan_sha256",
    "known_good_profile_sha256", "validator_implementation_sha256",
    "validator_result",
}
_VALIDATOR_RESULT_FIELDS = {
    "status", "verified_artifact_count", "failures", "warnings",
    "offline_revalidations",
}

ACCEPTED_CNN_PREDICTOR = {
    "producer_id": "frozen-wp5-six-channel-cnn",
    "producer_kind": "rgb_only_liquid_segmenter",
    "model_id": "resnet18-unet-current-empty-rgb",
    "model_sha256": "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba",
    "config_id": "wp5-cnn-six-channel-frozen-v1",
    "config_sha256": "7be72579f0cb0dc94f6bbe2cf894f46e63f24b9254f131106a0ed7805e96c64f",
    "code_commit": "a5f29ea90272a3424d8c7de09dbf435509d28105",
    "threshold_source": "twin_validation_frozen",
    "input_fields": ["rgb"],
    "training_data_origin": "twin",
}

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_EXPORT_FIELDS = {"schema_version", "artifact_type", "status", "predictor", "store", "records"}
_STORE_FIELDS = {
    "database_path", "database_sha256", "population_receipt_path",
    "population_receipt_sha256",
}
_RECORD_FIELDS = {
    "evidence_id", "event_cursor", "batch_id", "run_id", "case_id", "recipe_id",
    "split_role", "stage_id", "stage_ordinal", "tank_id", "rgb_sha256",
    "mask_sha256", "mask_source", "median_lab", "source_render_receipt_sha256",
    "source_validation_receipt_sha256", "source_render_plan_sha256",
    "known_good_profile_sha256",
}


class MaterializationError(ValueError):
    """The INN-67 materialization contract failed closed."""


def _fail(code: str, error: BaseException | None = None) -> MaterializationError:
    result = MaterializationError(code)
    if error is not None:
        result.__cause__ = error
    return result


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _hash_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    try:
        body = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _fail("noncanonical_input", error)
    return _hash_bytes(body)


def _absolute_regular(path_value: str | os.PathLike[str], role: str) -> Path:
    path = Path(path_value).expanduser().absolute()
    try:
        if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
            raise OSError("not an exact regular file")
    except OSError as error:
        raise _fail(f"{role}_file_invalid", error)
    return path


def _load_json(
    path_value: str | os.PathLike[str], role: str, expected_sha: str | None = None
) -> tuple[dict[str, object], Path, str]:
    path = _absolute_regular(path_value, role)
    try:
        body = path.read_bytes()
        value = json.loads(
            body,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise _fail(f"{role}_json_invalid", error)
    observed = _hash_bytes(body)
    if expected_sha is not None and observed != expected_sha:
        raise _fail(f"{role}_hash_mismatch")
    if not isinstance(value, dict):
        raise _fail(f"{role}_json_invalid")
    return value, path, observed


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise _fail(f"{role}_sha256_invalid")
    return value


def _exact(value: object, fields: set[str], role: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise _fail(f"{role}_fields_invalid")
    return value


def _sequence(value: object, role: str, count: int | None = None) -> list[object]:
    if not isinstance(value, list) or (count is not None and len(value) != count):
        raise _fail(f"{role}_denominator_invalid")
    return value


def _bound_file(path_value: object, sha_value: object, role: str) -> tuple[Path, str]:
    if not isinstance(path_value, str):
        raise _fail(f"{role}_path_invalid")
    expected = _sha(sha_value, role)
    path = _absolute_regular(path_value, role)
    try:
        observed = _hash_file(path)
    except OSError as error:
        raise _fail(f"{role}_file_invalid", error)
    if observed != expected:
        raise _fail(f"{role}_hash_mismatch")
    return path, observed


def _check_no_sealed_locator(value: object) -> None:
    """Reject explicit test/video-5 identities before opening bound artifacts."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in {"partition", "split", "split_role"} and isinstance(child, str):
                if child.lower() in {"test", "evaluation"}:
                    raise _fail("forbidden_test_or_video5_locator")
            if isinstance(child, str):
                lower = child.lower()
                if any(token in lower for token in ("video5", "video-5", "video_5", "wp1")):
                    raise _fail("forbidden_test_or_video5_locator")
            _check_no_sealed_locator(child)
    elif isinstance(value, list):
        for child in value:
            _check_no_sealed_locator(child)


def _validate_frozen_inputs(plan_root: Path, amendment_path: Path) -> dict[str, object]:
    root = plan_root.expanduser().absolute()
    try:
        if root.resolve(strict=True) != root or root.is_symlink() or not root.is_dir():
            raise OSError("plan root is not a canonical directory")
    except OSError as error:
        raise _fail("plan_root_invalid", error)
    expected = {
        "artifact-hashes.json": PLAN_INVENTORY_SHA256,
        "reference-bank-plan.json": PLAN_SHA256,
        "known-good-profile.json": PROFILE_SHA256,
        "delta-e-freeze-config.json": FREEZE_PROTOCOL_SHA256,
        "leakage-audit.json": LEAKAGE_SHA256,
        "no-access-attestation.json": NO_ACCESS_SHA256,
    }
    values = {
        relative: _load_json(root / relative, relative.replace(".json", ""), digest)[0]
        for relative, digest in expected.items()
    }
    inventory = values["artifact-hashes.json"]
    files = inventory.get("files")
    if (
        inventory.get("schema_version") != 1
        or inventory.get("artifact_type") != "inn67_reference_bank_plan_hash_inventory"
        or not isinstance(files, dict)
    ):
        raise _fail("plan_inventory_invalid")
    for relative, digest in expected.items():
        if relative != "artifact-hashes.json" and files.get(relative) != digest:
            raise _fail("plan_inventory_binding_mismatch")
    plan = values["reference-bank-plan.json"]
    profile = values["known-good-profile.json"]
    freeze = values["delta-e-freeze-config.json"]
    if (
        plan.get("artifact_type") != "inn67_development_reference_bank_plan"
        or plan.get("status") != "frozen_unrendered"
        or plan.get("denominators") != {
            "development_batches": 3, "palette_subruns": 6, "fill_cases": 18,
            "stage_observations": 48, "reference_donor_batches": 2,
            "config_holdout_batches": 1,
        }
    ):
        raise _fail("reference_bank_plan_invalid")
    if (
        profile.get("artifact_type") != "inn67_frozen_synthetic_known_good_profile"
        or profile.get("status") != "frozen"
        or profile.get("injected_defect") is not False
        or profile.get("prescribed_leading_residual_volume_ml") != 0.0
    ):
        raise _fail("known_good_profile_invalid")
    if (
        freeze.get("fit_batches") != list(DONOR_BATCHES)
        or freeze.get("held_out_development_batch") != CONFIGURATION_BATCH
        or freeze.get("threshold_value") is not None
    ):
        raise _fail("delta_e_protocol_invalid")
    amendment, _, _ = _load_json(amendment_path, "amendment", AMENDMENT_FILE_SHA256)
    bound = amendment.get("bound_freeze_inputs")
    scope = amendment.get("effective_delta_e_freeze_scope")
    if (
        amendment.get("artifact_type") != "inn67_pre_access_protocol_amendment"
        or amendment.get("status") != "frozen_before_first_render"
        or amendment.get("receipt_sha256") != AMENDMENT_RECEIPT_SHA256
        or amendment.get("render_or_test_access_performed") is not False
        or bound != {
            "reference_bank_plan_sha256": PLAN_SHA256,
            "reference_bank_inventory_sha256": PLAN_INVENTORY_SHA256,
            "known_good_profile_sha256": PROFILE_SHA256,
            "delta_e_protocol_sha256": FREEZE_PROTOCOL_SHA256,
            "leakage_audit_sha256": LEAKAGE_SHA256,
            "no_access_attestation_sha256": NO_ACCESS_SHA256,
        }
        or not isinstance(scope, dict)
        or scope.get("reference_donor_batches") != list(DONOR_BATCHES)
        or scope.get("configuration_freeze_target_batches") != [CONFIGURATION_BATCH]
        or scope.get("wp1_test_used_for_configuration") is not False
        or scope.get("video5_used_for_configuration") is not False
    ):
        raise _fail("amendment_contract_invalid")
    render_plan_hashes: dict[str, str] = {}
    render_schedules: dict[str, dict[str, object]] = {}
    for raw in _sequence(plan.get("subruns"), "plan_subruns", 6):
        if not isinstance(raw, dict):
            raise _fail("plan_subruns_invalid")
        relative = raw.get("render_plan_artifact")
        if not isinstance(relative, str) or files.get(relative) is None:
            raise _fail("subrun_plan_inventory_missing")
        subrun_plan = _load_json(
            root / relative, "subrun_plan", _sha(files[relative], "subrun_plan")
        )[0]
        subrun_id = raw.get("subrun_id")
        if not isinstance(subrun_id, str) or subrun_id in render_plan_hashes:
            raise _fail("plan_subruns_invalid")
        render_plan_hashes[subrun_id] = _canonical_sha(subrun_plan)
        schedules = subrun_plan.get("execution_schedules")
        if not isinstance(schedules, list) or len(schedules) != 3:
            raise _fail("subrun_plan_schedules_invalid")
        by_renderer = {}
        for schedule in schedules:
            binding = schedule.get("case_binding") if isinstance(schedule, dict) else None
            renderer_id = binding.get("case_id") if isinstance(binding, dict) else None
            if not isinstance(renderer_id, str) or renderer_id in by_renderer:
                raise _fail("subrun_plan_schedules_invalid")
            by_renderer[renderer_id] = schedule
        render_schedules[subrun_id] = by_renderer
    return {
        "root": root,
        "plan": plan,
        "inventory": inventory,
        "render_plan_hashes": render_plan_hashes,
        "render_schedules": render_schedules,
    }


def _validate_render_receipt_binding(
    receipt: object,
    *,
    planned_case: Mapping[str, object],
    planned_subrun: Mapping[str, object],
    planned_schedule: Mapping[str, object],
    renderer_id: str,
    render_plan_sha256: str,
) -> None:
    """Bind the real schema-20 receipt shape to one frozen INN-67 case."""

    if not isinstance(receipt, dict):
        raise _fail("case_receipt_not_accepted")
    arguments = receipt.get("arguments")
    schedule = receipt.get("execution_schedule")
    model = schedule.get("model_input_record") if isinstance(schedule, dict) else None
    renderer = (
        schedule.get("renderer_case_binding") if isinstance(schedule, dict) else None
    )
    expected_renderer = planned_schedule.get("case_binding")
    expected_model = planned_schedule.get("model_input_record")
    if (
        not isinstance(arguments, dict)
        or not isinstance(schedule, dict)
        or not isinstance(model, dict)
        or not isinstance(renderer, dict)
        or not isinstance(expected_renderer, dict)
        or not isinstance(expected_model, dict)
        or renderer != expected_renderer
        or receipt.get("schema_version") != 20
        or "artifact_type" in receipt
        or planned_subrun.get("development_batch_id") != planned_case.get("batch_id")
        or planned_subrun.get("run_id") != planned_case.get("run_id")
        or receipt.get("status") != "pass"
        or arguments.get("train_case_id") != renderer_id
        or arguments.get("train_render_plan_sha256") != render_plan_sha256
        or schedule.get("case_id") != renderer_id
        or schedule.get("cohort_plan_sha256") != render_plan_sha256
        or model.get("case_id") != renderer_id
        or renderer.get("case_id") != renderer_id
        or model.get("batch_id") != planned_subrun.get("renderer_internal_batch_id")
        or renderer.get("batch_id") != planned_subrun.get("renderer_internal_batch_id")
        or model.get("run_id") != planned_case.get("run_id")
        or renderer.get("run_id") != planned_case.get("run_id")
        or model.get("current_recipe_id") != planned_case.get("recipe_id")
        or renderer.get("current_recipe_id") != planned_case.get("recipe_id")
        or model.get("split_role") != planned_case.get("split_role")
        or renderer.get("split_role") != planned_case.get("split_role")
        or model.get("split_version") != "inn67-development-reference-bank-v1"
        or model.get("data_origin") != "twin"
        or renderer.get("batch_role") != "scored_target"
        or renderer.get("residual_volume_ml")
        != expected_renderer.get("residual_volume_ml")
        or renderer.get("vial_ordinal") != expected_renderer.get("vial_ordinal")
        or renderer.get("predecessor_context")
        != expected_renderer.get("predecessor_context")
        or renderer.get("previous_recipe_id")
        != expected_renderer.get("previous_recipe_id")
        or model.get("vial_position") != expected_renderer.get("vial_ordinal")
        or model.get("previous_recipe_id") != expected_model.get("previous_recipe_id")
        or model.get("previous_programmed_ml")
        != expected_model.get("previous_programmed_ml")
        or (
            "previous_final_tank_id" in model
            and model.get("previous_final_tank_id")
            != expected_model.get("previous_final_tank_id")
        )
        or model.get("current_programmed_ml") != planned_case.get("programmed_ml")
        or model.get("programmed_dispense_order") != planned_case.get("dispense_order")
    ):
        raise _fail("case_receipt_not_accepted")


def _receipt_rgb_declarations(
    receipt: Mapping[str, object], planned_case: Mapping[str, object]
) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Return only the prefill/stage RGB declarations; semantic masks stay excluded."""

    parity = receipt.get("pre_pour_parity")
    semantics = receipt.get("semantics")
    pairs = semantics.get("pairs") if isinstance(semantics, dict) else None
    stages = planned_case.get("stages")
    if (
        not isinstance(parity, dict)
        or not isinstance(pairs, list)
        or not isinstance(stages, list)
        or len(pairs) != len(stages)
    ):
        raise _fail("case_rgb_declarations_invalid")
    prefill_path = parity.get("f2_prefill_path")
    prefill_sha = parity.get("f2_prefill_sha256")
    if not isinstance(prefill_path, str) or _SHA.fullmatch(str(prefill_sha)) is None:
        raise _fail("case_rgb_declarations_invalid")
    rows = []
    for pair, stage in zip(pairs, stages, strict=True):
        if not isinstance(pair, dict) or not isinstance(stage, dict):
            raise _fail("case_rgb_declarations_invalid")
        rgb_path = pair.get("rgb_path")
        rgb_sha = pair.get("rgb_sha256")
        if (
            not isinstance(rgb_path, str)
            or _SHA.fullmatch(str(rgb_sha)) is None
            or pair.get("phase") != f"{stage.get('tank_id')}-stage"
            or isinstance(stage.get("ordinal"), bool)
            or not isinstance(stage.get("ordinal"), int)
            or not isinstance(stage.get("tank_id"), str)
        ):
            raise _fail("case_rgb_declarations_invalid")
        rows.append(
            {
                "stage_ordinal": stage["ordinal"],
                "tank_id": stage["tank_id"],
                "rgb_path": rgb_path,
                "rgb_sha256": rgb_sha,
            }
        )
    return {"rgb_path": prefill_path, "rgb_sha256": prefill_sha}, rows


def _receipt_relative_artifact(app_output: Path, relative: object, role: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise _fail(f"{role}_path_invalid")
    untrusted = Path(relative)
    if untrusted.is_absolute() or ".." in untrusted.parts:
        raise _fail(f"{role}_path_invalid")
    path = (app_output / untrusted).absolute()
    try:
        if (
            path.resolve(strict=True) != path
            or path.is_symlink()
            or not path.is_file()
            or not path.is_relative_to(app_output)
        ):
            raise OSError("invalid receipt-relative artifact")
    except OSError as error:
        raise _fail(f"{role}_path_invalid", error)
    return path


def _validate_render_inventory(
    path: Path, frozen: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    value, _, inventory_sha = _load_json(path, "render_inventory")
    _check_no_sealed_locator(value)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != "inn67_accepted_render_inventory"
        or value.get("status") != "accepted"
        or value.get("reference_bank_plan_sha256") != PLAN_SHA256
        or value.get("amendment_receipt_sha256") != AMENDMENT_RECEIPT_SHA256
    ):
        raise _fail("render_inventory_invalid")
    plan = frozen["plan"]
    render_plan_hashes = frozen.get("render_plan_hashes")
    if not isinstance(render_plan_hashes, dict):
        raise _fail("frozen_render_plan_hashes_missing")
    render_schedules = frozen.get("render_schedules")
    if not isinstance(render_schedules, dict):
        raise _fail("frozen_render_schedules_missing")
    expected_subruns = {row["subrun_id"]: row for row in plan["subruns"]}
    by_case: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for raw in _sequence(value.get("subruns"), "render_subruns", 6):
        _exact(raw, {"subrun_id", "render_root", "render_plan_sha256", "supervisor_receipt_path", "supervisor_receipt_sha256", "cases"}, "render_subrun")
        subrun_id = raw.get("subrun_id")
        if not isinstance(subrun_id, str) or subrun_id in seen or subrun_id not in expected_subruns:
            raise _fail("mixed_or_duplicate_subrun")
        seen.add(subrun_id)
        planned_subrun = expected_subruns[subrun_id]
        plan_hash = render_plan_hashes.get(subrun_id)
        if not isinstance(plan_hash, str):
            raise _fail("render_plan_hash_mismatch")
        if raw.get("render_plan_sha256") != plan_hash:
            raise _fail("render_plan_hash_mismatch")
        root_value = raw.get("render_root")
        if not isinstance(root_value, str):
            raise _fail("render_root_invalid")
        root = Path(root_value).expanduser().absolute()
        try:
            if root.resolve(strict=True) != root or root.is_symlink() or not root.is_dir():
                raise OSError("invalid render root")
        except OSError as error:
            raise _fail("render_root_invalid", error)
        supervisor_path, _ = _bound_file(raw.get("supervisor_receipt_path"), raw.get("supervisor_receipt_sha256"), "supervisor_receipt")
        if not supervisor_path.is_relative_to(root):
            raise _fail("supervisor_receipt_outside_render_root")
        supervisor = _load_json(supervisor_path, "supervisor_receipt")[0]
        planned_cases = [row for row in plan["cases"] if row["subrun_id"] == subrun_id]
        expected_renderer_ids = {row["renderer_case_id"] for row in planned_cases}
        supervisor_results = supervisor.get("case_results")
        supervisor_ids = (
            [row.get("case_id") for row in supervisor_results if isinstance(row, dict)]
            if isinstance(supervisor_results, list)
            else []
        )
        if (
            supervisor.get("schema_version") != 1
            or supervisor.get("artifact_type")
            != "physx_train_residual_serial_supervisor_attempt"
            or supervisor.get("status") != "pass"
            or supervisor.get("plan_sha256") != plan_hash
            or supervisor.get("planned_case_count") != 3
            or supervisor.get("validated_completed_case_count") != 3
            or not isinstance(supervisor_results, list)
            or len(supervisor_results) != 3
            or not all(isinstance(value, str) for value in supervisor_ids)
            or len(set(supervisor_ids)) != 3
            or set(supervisor_ids) != expected_renderer_ids
            or any(
                not isinstance(row, dict)
                or row.get("outcome") not in {"passed", "recognized_completed"}
                or not isinstance(row.get("evidence"), dict)
                or row["evidence"].get("artifact_validation", {}).get("status") != "pass"
                or row["evidence"].get("artifact_validation", {}).get("failures") != []
                for row in supervisor_results
            )
        ):
            raise _fail("supervisor_receipt_not_accepted")
        by_renderer = {row["renderer_case_id"]: row for row in planned_cases}
        for case_raw in _sequence(raw.get("cases"), "render_cases", 3):
            _exact(
                case_raw,
                {
                    "renderer_case_id", "receipt_path", "receipt_sha256",
                    "validation_receipt_path", "validation_receipt_sha256",
                    "prefill_rgb_path", "prefill_rgb_sha256", "stage_rgbs",
                },
                "render_case",
            )
            renderer_id = case_raw.get("renderer_case_id")
            if not isinstance(renderer_id, str) or renderer_id not in by_renderer:
                raise _fail("mixed_or_duplicate_case")
            planned_case = by_renderer.pop(renderer_id)
            receipt_path, receipt_hash = _bound_file(case_raw.get("receipt_path"), case_raw.get("receipt_sha256"), "render_receipt")
            validation_path, validation_hash = _bound_file(case_raw.get("validation_receipt_path"), case_raw.get("validation_receipt_sha256"), "validation_receipt")
            if not receipt_path.is_relative_to(root) or not validation_path.is_relative_to(root):
                raise _fail("case_receipt_outside_render_root")
            receipt = _load_json(receipt_path, "render_receipt")[0]
            validation = _load_json(validation_path, "validation_receipt")[0]
            _validate_render_receipt_binding(
                receipt,
                planned_case=planned_case,
                planned_subrun=planned_subrun,
                planned_schedule=render_schedules[subrun_id][renderer_id],
                renderer_id=renderer_id,
                render_plan_sha256=plan_hash,
            )
            declared_prefill, declared_stages = _receipt_rgb_declarations(
                receipt, planned_case
            )
            app_output = receipt_path.parent
            expected_prefill_path = _receipt_relative_artifact(
                app_output, declared_prefill["rgb_path"], "prefill_rgb"
            )
            prefill_path, prefill_hash = _bound_file(
                case_raw.get("prefill_rgb_path"),
                case_raw.get("prefill_rgb_sha256"),
                "prefill_rgb",
            )
            if (
                prefill_path != expected_prefill_path
                or prefill_hash != declared_prefill["rgb_sha256"]
            ):
                raise _fail("prefill_rgb_binding_mismatch")
            inventory_stage_rows = _sequence(
                case_raw.get("stage_rgbs"),
                "stage_rgbs",
                len(declared_stages),
            )
            accepted_stage_rows = []
            for inventory_stage, declared_stage in zip(
                inventory_stage_rows, declared_stages, strict=True
            ):
                _exact(
                    inventory_stage,
                    {"stage_ordinal", "tank_id", "rgb_path", "rgb_sha256"},
                    "stage_rgb",
                )
                expected_rgb_path = _receipt_relative_artifact(
                    app_output, declared_stage["rgb_path"], "stage_rgb"
                )
                rgb_path, rgb_hash = _bound_file(
                    inventory_stage.get("rgb_path"),
                    inventory_stage.get("rgb_sha256"),
                    "stage_rgb",
                )
                if (
                    inventory_stage.get("stage_ordinal")
                    != declared_stage["stage_ordinal"]
                    or inventory_stage.get("tank_id") != declared_stage["tank_id"]
                    or rgb_path != expected_rgb_path
                    or rgb_hash != declared_stage["rgb_sha256"]
                ):
                    raise _fail("stage_rgb_binding_mismatch")
                accepted_stage_rows.append(
                    {
                        "stage_ordinal": declared_stage["stage_ordinal"],
                        "tank_id": declared_stage["tank_id"],
                        "rgb_path": os.fspath(rgb_path),
                        "rgb_sha256": rgb_hash,
                    }
                )
            try:
                _validate_occlusion_receipt(
                    validation,
                    renderer_id=renderer_id,
                    render_receipt_sha256=receipt_hash,
                    render_plan_sha256=plan_hash,
                )
            except MaterializationError:
                raise _fail("validation_receipt_not_accepted") from None
            external_case_id = planned_case["case_id"]
            if external_case_id in by_case:
                raise _fail("mixed_or_duplicate_case")
            by_case[external_case_id] = {
                "render_receipt_sha256": receipt_hash,
                "validation_receipt_sha256": validation_hash,
                "render_plan_sha256": plan_hash,
                "prefill_rgb_path": os.fspath(prefill_path),
                "prefill_rgb_sha256": prefill_hash,
                "stage_rgbs": accepted_stage_rows,
            }
        if by_renderer:
            raise _fail("render_case_denominator_invalid")
    if seen != set(expected_subruns) or len(by_case) != 18:
        raise _fail("render_denominator_invalid")
    by_case["__inventory__"] = {"sha256": inventory_sha}
    return by_case


def _validate_materialization_amendment(path: Path) -> None:
    value = _load_json(
        path,
        "materialization_amendment",
        MATERIALIZATION_AMENDMENT_SHA256,
    )[0]
    rule = value.get("frozen_rule")
    anti_vacuity = value.get("anti_vacuity")
    bound = value.get("bound_inputs")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type")
        != "inn67_pre_render_materialization_protocol_amendment"
        or value.get("status") != "frozen_before_first_inn67_render"
        or value.get("recorded_before_any_inn67_render") is not True
        or value.get("render_or_test_access_performed") is not False
        or value.get("wp1_one_pass_status") != "sealed"
        or value.get("video5_status") != "sealed"
        or not isinstance(rule, dict)
        or rule.get("config_id") != TRIAD_CONFIG_ID
        or rule.get("distance_metric") != "CIE76"
        or rule.get("reference_donor_batches") != list(DONOR_BATCHES)
        or rule.get("configuration_target_batch") != CONFIGURATION_BATCH
        or rule.get("peer_pair_delta_e76_max")
        != TIGHT_PEER_PAIR_MAX_DELTA_E76
        or rule.get("threshold_formula")
        != "threshold_delta_e76 = peer_pair_delta_e76 / 2 + 1.0"
        or rule.get("target_outlier_margin_delta_e76")
        != TARGET_OUTLIER_MARGIN_DELTA_E76
        or not isinstance(anti_vacuity, dict)
        or anti_vacuity
        != {
            "threshold_uses_b03_target_distance": False,
            "threshold_uses_wp1_test": False,
            "threshold_uses_video5": False,
            "post_access_tuning_permitted": False,
        }
        or not isinstance(bound, dict)
        or bound.get("reference_bank_plan_sha256") != PLAN_SHA256
        or bound.get("reference_bank_inventory_sha256")
        != PLAN_INVENTORY_SHA256
        or bound.get("known_good_profile_sha256") != PROFILE_SHA256
        or bound.get("delta_e_freeze_protocol_sha256")
        != FREEZE_PROTOCOL_SHA256
        or bound.get("recipe_role_amendment_file_sha256")
        != AMENDMENT_FILE_SHA256
        or bound.get("accepted_occlusion_validator_sha256")
        != ACCEPTED_OCCLUSION_VALIDATOR_SHA256
    ):
        raise _fail("materialization_amendment_invalid")


def _validate_occlusion_receipt(
    validation: object,
    *,
    renderer_id: str,
    render_receipt_sha256: str,
    render_plan_sha256: str,
) -> None:
    wrapper = _exact(
        validation, _VALIDATION_RECEIPT_FIELDS, "validation_receipt"
    )
    if (
        wrapper.get("schema_version") != 1
        or wrapper.get("artifact_type")
        != "inn67_occlusion_aware_artifact_validation"
        or wrapper.get("status") != "pass"
        or wrapper.get("failures") != []
        or wrapper.get("case_id") != renderer_id
        or wrapper.get("source_render_receipt_sha256")
        != render_receipt_sha256
        or wrapper.get("source_render_plan_sha256") != render_plan_sha256
        or wrapper.get("known_good_profile_sha256") != PROFILE_SHA256
        or wrapper.get("validator_implementation_sha256")
        != ACCEPTED_OCCLUSION_VALIDATOR_SHA256
    ):
        raise _fail("validation_receipt_identity_invalid")
    result = _exact(
        wrapper.get("validator_result"),
        _VALIDATOR_RESULT_FIELDS,
        "validator_result",
    )
    verified_count = result.get("verified_artifact_count")
    warnings = result.get("warnings")
    revalidations = result.get("offline_revalidations")
    if (
        result.get("status") != "pass"
        or result.get("failures") != []
        or isinstance(verified_count, bool)
        or not isinstance(verified_count, int)
        or verified_count < 1
        or not isinstance(warnings, list)
        or any(
            not isinstance(warning, dict)
            or not isinstance(warning.get("warning_type"), str)
            for warning in warnings
        )
        or not isinstance(revalidations, list)
        or any(
            not isinstance(record, dict)
            or record.get("passed") is not True
            or not isinstance(record.get("revalidation_type"), str)
            for record in revalidations
        )
    ):
        raise _fail("validator_result_invalid")


def _finite_lab(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise _fail("median_lab_invalid")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise _fail("median_lab_invalid")
        result.append(float(item))
    return result


def _validate_store_export(
    path: Path,
    plan: Mapping[str, object],
    receipts: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object], str]:
    value, _, export_sha = _load_json(path, "predicted_store_export")
    _check_no_sealed_locator(value)
    _exact(value, _EXPORT_FIELDS, "predicted_store_export")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != "inn67_predicted_store_export"
        or value.get("status") != "accepted"
        or value.get("predictor") != ACCEPTED_CNN_PREDICTOR
    ):
        raise _fail("accepted_cnn_predictor_mismatch")
    store = _exact(value.get("store"), _STORE_FIELDS, "store_tuple")
    database_path, database_sha = _bound_file(store.get("database_path"), store.get("database_sha256"), "evidence_store")
    population_path, population_sha = _bound_file(store.get("population_receipt_path"), store.get("population_receipt_sha256"), "population_receipt")
    population = _load_json(population_path, "population_receipt")[0]
    raw_records = _sequence(value.get("records"), "predicted_records", 48)
    if (
        population.get("schema_version") != 1
        or population.get("artifact_type") != "inn67_predicted_evidence_store_population"
        or population.get("status") != "pass"
        or population.get("store_sha256") != database_sha
        or population.get("predictor") != ACCEPTED_CNN_PREDICTOR
        or population.get("record_count") != 48
        or population.get("records_sha256") != _canonical_sha(raw_records)
        or population.get("evidence_ids") != [row.get("evidence_id") if isinstance(row, dict) else None for row in raw_records]
    ):
        raise _fail("accepted_store_tuple_mismatch")
    expected_keys = {
        (case["case_id"], stage["ordinal"]): (case, stage)
        for case in plan["cases"] for stage in case["stages"]
    }
    records: list[dict[str, object]] = []
    observed_keys: set[tuple[str, int]] = set()
    evidence_ids: set[str] = set()
    cursors: set[int] = set()
    for raw in raw_records:
        record = _exact(raw, _RECORD_FIELDS, "predicted_record")
        if record.get("mask_source") != "predicted":
            raise _fail("mask_source_not_predicted")
        for field in ("rgb_sha256", "mask_sha256", "source_render_receipt_sha256", "source_validation_receipt_sha256", "source_render_plan_sha256", "known_good_profile_sha256"):
            _sha(record.get(field), field)
        case_id = record.get("case_id")
        ordinal = record.get("stage_ordinal")
        key = (case_id, ordinal) if isinstance(case_id, str) and isinstance(ordinal, int) and not isinstance(ordinal, bool) else None
        if key is None or key not in expected_keys or key in observed_keys:
            raise _fail("record_plan_binding_mismatch")
        case, stage = expected_keys[key]
        expected = {
            "batch_id": case["batch_id"], "run_id": case["run_id"], "case_id": case["case_id"],
            "recipe_id": case["recipe_id"], "split_role": "development",
            "stage_id": f"dispense-{stage['ordinal']}-{stage['tank_id']}",
            "stage_ordinal": stage["ordinal"], "tank_id": stage["tank_id"],
        }
        if any(record.get(field) != expected_value for field, expected_value in expected.items()):
            raise _fail("record_plan_binding_mismatch")
        receipt = receipts[case_id]
        if (
            record.get("source_render_receipt_sha256") != receipt["render_receipt_sha256"]
            or record.get("source_validation_receipt_sha256") != receipt["validation_receipt_sha256"]
            or record.get("source_render_plan_sha256") != receipt["render_plan_sha256"]
            or record.get("known_good_profile_sha256") != PROFILE_SHA256
        ):
            raise _fail("record_source_profile_hash_mismatch")
        evidence_id = record.get("evidence_id")
        cursor = record.get("event_cursor")
        if (
            not isinstance(evidence_id, str) or not evidence_id or evidence_id in evidence_ids
            or isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0 or cursor in cursors
        ):
            raise _fail("record_identity_invalid")
        normalized = deepcopy(record)
        normalized["median_lab"] = _finite_lab(record.get("median_lab"))
        records.append(normalized)
        observed_keys.add(key)
        evidence_ids.add(evidence_id)
        cursors.add(cursor)
    if observed_keys != set(expected_keys):
        raise _fail("record_denominator_invalid")
    donor_cursors = [row["event_cursor"] for row in records if row["batch_id"] in DONOR_BATCHES]
    config_cursors = [row["event_cursor"] for row in records if row["batch_id"] == CONFIGURATION_BATCH]
    if not donor_cursors or not config_cursors or max(donor_cursors) >= min(config_cursors):
        raise _fail("donor_not_earlier_than_configuration_target")
    return records, {
        "database_path": str(database_path), "database_sha256": database_sha,
        "population_receipt_path": str(population_path),
        "population_receipt_sha256": population_sha,
    }, export_sha


def _delta(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def _build_outputs(
    records: Sequence[Mapping[str, object]], provenance: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    by_key: dict[tuple[str, str, int], dict[str, object]] = {}
    for raw in records:
        row = dict(raw)
        key = (str(row["batch_id"]), str(row["recipe_id"]), int(row["stage_ordinal"]))
        if key in by_key:
            raise _fail("wrong_recipe_or_stage_peer")
        by_key[key] = row
    bank_rows = sorted(
        (dict(row) for row in records if row["batch_id"] in DONOR_BATCHES),
        key=lambda row: int(row["event_cursor"]),
    )
    bindings = []
    observations = []
    for target in sorted(
        (dict(row) for row in records if row["batch_id"] == CONFIGURATION_BATCH),
        key=lambda row: int(row["event_cursor"]),
    ):
        peer_rows = []
        for batch in DONOR_BATCHES:
            peer = by_key.get((batch, str(target["recipe_id"]), int(target["stage_ordinal"])))
            if peer is None or peer["stage_id"] != target["stage_id"] or peer["tank_id"] != target["tank_id"]:
                raise _fail("wrong_recipe_or_stage_peer")
            if int(peer["event_cursor"]) >= int(target["event_cursor"]):
                raise _fail("peer_not_earlier")
            peer_rows.append(peer)
        if len({row["run_id"] for row in peer_rows}) != 2:
            raise _fail("mixed_or_nonindependent_donor_runs")
        center = [
            (float(peer_rows[0]["median_lab"][index]) + float(peer_rows[1]["median_lab"][index])) / 2.0
            for index in range(3)
        ]
        peer_delta = _delta(peer_rows[0]["median_lab"], peer_rows[1]["median_lab"])
        if peer_delta > TIGHT_PEER_PAIR_MAX_DELTA_E76:
            raise _fail("donor_pair_not_tight")
        threshold = peer_delta / 2.0 + TARGET_OUTLIER_MARGIN_DELTA_E76
        distance = _delta(target["median_lab"], center)
        consistent = distance <= threshold
        if not consistent:
            raise _fail("configuration_target_not_consistent")
        bindings.append({
            "target_evidence_id": target["evidence_id"], "target_batch_id": CONFIGURATION_BATCH,
            "recipe_id": target["recipe_id"], "stage_id": target["stage_id"],
            "stage_ordinal": target["stage_ordinal"],
            "earlier_peer_evidence_ids": [row["evidence_id"] for row in peer_rows],
            "earlier_peer_batch_ids": list(DONOR_BATCHES),
        })
        observations.append({
            "recipe_id": target["recipe_id"], "stage_id": target["stage_id"],
            "stage_ordinal": target["stage_ordinal"], "tank_id": target["tank_id"],
            "reference_center_lab": center,
            "peer_evidence_ids": [row["evidence_id"] for row in peer_rows],
            "peer_pair_delta_e76": peer_delta,
            "threshold_delta_e76": threshold,
            "configuration_target_evidence_id": target["evidence_id"],
            "target_to_reference_center_delta_e76": distance,
            "relative_status": "consistent_with_two_peer_mode",
            "consistent_with_two_peer_mode": consistent,
        })
    if len(bank_rows) != 32 or len(bindings) != 16:
        raise _fail("materialized_denominator_invalid")
    bank = {
        "schema_version": 1, "artifact_type": "inn67_development_reference_bank",
        "status": "accepted_development_only", "split_role": "development",
        "reference_donor_batch_ids": list(DONOR_BATCHES),
        "configuration_target_batch_id": CONFIGURATION_BATCH,
        "predictor": deepcopy(ACCEPTED_CNN_PREDICTOR),
        "denominators": {"donor_batches": 2, "donor_records": 32, "configuration_targets": 16, "peers_per_configuration_target": 2},
        "records": bank_rows, "history_bindings": bindings,
        "claim_boundary": "synthetic contract-known-good development evidence; not physical cleanliness or plant truth",
        "provenance": deepcopy(dict(provenance)),
    }
    frozen = {
        "schema_version": 1, "artifact_type": "inn67_delta_e_frozen_development_config",
        "status": "frozen_before_test_or_video5_access", "distance_metric": "CIE76",
        "unit": "delta_e76",
        "config_id": TRIAD_CONFIG_ID,
        "tight_peer_pair_max_delta_e76": TIGHT_PEER_PAIR_MAX_DELTA_E76,
        "target_outlier_margin_delta_e76": TARGET_OUTLIER_MARGIN_DELTA_E76,
        "decision_rule": (
            "peer_delta = dE76(B01,B02); threshold = peer_delta/2 + 1.0; "
            "require peer_delta <= 2.0 and B03 target_to_center_delta_e76 <= threshold"
        ),
        "threshold_formula": (
            "threshold_delta_e76 = peer_pair_delta_e76 / 2 + 1.0"
        ),
        "configuration_target_gate": (
            "all 16 B03 recipe-stage targets must be consistent_with_two_peer_mode"
        ),
        "selection_input_batch_ids": [CONFIGURATION_BATCH],
        "reference_geometry_batch_ids": list(DONOR_BATCHES),
        "recipe_stage_configurations": observations,
        "recipe_stage_configuration_count": 16, "wp1_test_accessed": False,
        "video5_accessed": False,
        "claim_boundary": "development-only synthetic configuration; no held-out test or physical-validity result",
        "provenance": deepcopy(dict(provenance)),
    }
    return bank, frozen


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, body: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _remove_owned_output(path: Path, identity: tuple[int, int]) -> None:
    """Remove only the same directory inode created by this invocation."""
    try:
        current = path.stat(follow_symlinks=False)
    except OSError:
        return
    if (current.st_dev, current.st_ino) != identity or path.is_symlink():
        return
    shutil.rmtree(path)


def materialize_reference_bank(
    *, plan_root: str | os.PathLike[str], amendment_path: str | os.PathLike[str],
    materialization_amendment_path: str | os.PathLike[str],
    render_inventory_path: str | os.PathLike[str],
    predicted_store_export_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, object]:
    """Validate all accepted inputs and publish a no-clobber development bank."""

    _validate_materialization_amendment(Path(materialization_amendment_path))
    frozen = _validate_frozen_inputs(Path(plan_root), Path(amendment_path))
    receipts = _validate_render_inventory(Path(render_inventory_path), frozen)
    records, store_tuple, export_sha = _validate_store_export(Path(predicted_store_export_path), frozen["plan"], receipts)
    provenance = {
        "reference_bank_plan_sha256": PLAN_SHA256,
        "reference_bank_inventory_sha256": PLAN_INVENTORY_SHA256,
        "pre_access_amendment_file_sha256": AMENDMENT_FILE_SHA256,
        "pre_access_amendment_receipt_sha256": AMENDMENT_RECEIPT_SHA256,
        "materialization_protocol_amendment_sha256": MATERIALIZATION_AMENDMENT_SHA256,
        "render_inventory_sha256": receipts["__inventory__"]["sha256"],
        "predicted_store_export_sha256": export_sha,
        "accepted_store_tuple": store_tuple,
    }
    bank, frozen_config = _build_outputs(records, provenance)
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output_directory_exists")
    output.mkdir(parents=True, mode=0o750)
    created = output.stat(follow_symlinks=False)
    identity = (created.st_dev, created.st_ino)
    try:
        bodies = {
            "development-reference-bank.json": _json_bytes(bank),
            "delta-e-frozen-config.json": _json_bytes(frozen_config),
        }
        hashes = {}
        for name, body in bodies.items():
            _write_new(output / name, body)
            hashes[name] = _hash_bytes(body)
        receipt = {
            "schema_version": 1, "artifact_type": "inn67_reference_bank_materialization_receipt",
            "status": "published", "cpu_only": True, "gpu_or_renderer_launched": False,
            "image_or_video_bytes_opened": False, "wp1_test_accessed": False,
            "video5_accessed": False, "outputs": hashes, "provenance": provenance,
        }
        receipt_body = _json_bytes(receipt)
        _write_new(output / "materialization-receipt.json", receipt_body)
        return {
            "status": "published", "output_dir": str(output),
            "outputs": {**hashes, "materialization-receipt.json": _hash_bytes(receipt_body)},
        }
    except BaseException:
        _remove_owned_output(output, identity)
        raise


__all__ = [
    "ACCEPTED_CNN_PREDICTOR", "ACCEPTED_OCCLUSION_VALIDATOR_SHA256", "AMENDMENT_FILE_SHA256",
    "AMENDMENT_RECEIPT_SHA256", "MATERIALIZATION_AMENDMENT_SHA256",
    "MaterializationError",
    "materialize_reference_bank",
]
