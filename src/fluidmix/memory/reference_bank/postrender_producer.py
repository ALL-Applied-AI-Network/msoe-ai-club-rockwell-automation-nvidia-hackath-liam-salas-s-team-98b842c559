"""CPU-only INN-67 post-render validation and inventory production.

This boundary only reads the frozen development plan and completed render artifacts. It
reruns the canonical validator and publishes new validation wrappers plus one complete
inventory. It does not render, export a store, or rewrite existing render artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

from .materializer import (
    ACCEPTED_OCCLUSION_VALIDATOR_SHA256,
    AMENDMENT_RECEIPT_SHA256,
    MaterializationError,
    PLAN_INVENTORY_SHA256,
    PLAN_SHA256,
    PROFILE_SHA256,
    _receipt_rgb_declarations,
    _validate_render_receipt_binding,
)


_RESULT_FIELDS = {
    "status", "verified_artifact_count", "failures", "warnings",
    "offline_revalidations",
}


class PostrenderProductionError(ValueError):
    """The post-render input contract failed closed."""


def _fail(code: str, error: BaseException | None = None) -> PostrenderProductionError:
    result = PostrenderProductionError(code)
    if error is not None:
        result.__cause__ = error
    return result


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key: {key}")
        value[key] = item
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    try:
        body = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _fail("subrun_plan_noncanonical", error)
    return hashlib.sha256(body).hexdigest()


def _canonical_directory(path_value: str | os.PathLike[str], role: str) -> Path:
    path = Path(path_value).expanduser().absolute()
    try:
        if path.is_symlink() or path.resolve(strict=True) != path or not path.is_dir():
            raise OSError("not a canonical directory")
    except OSError as error:
        raise _fail(f"{role}_invalid", error)
    return path


def _regular_file(path: Path, role: str, *, within: Path) -> Path:
    try:
        if (
            path.is_symlink()
            or path.resolve(strict=True) != path
            or not path.is_file()
            or not path.is_relative_to(within)
        ):
            raise OSError("not an in-root canonical regular file")
    except OSError as error:
        raise _fail(f"{role}_invalid", error)
    return path


def _load_json(
    path: Path, role: str, *, within: Path, expected_sha256: str | None = None
) -> tuple[dict[str, object], str]:
    _regular_file(path, role, within=within)
    try:
        body = path.read_bytes()
        value = json.loads(
            body,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise _fail(f"{role}_json_invalid", error)
    observed = hashlib.sha256(body).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise _fail(f"{role}_hash_mismatch")
    if not isinstance(value, dict):
        raise _fail(f"{role}_json_invalid")
    return value, observed


def _reject_symlinks(root: Path) -> None:
    for directory, names, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*names, *filenames]:
            if (base / name).is_symlink():
                raise _fail("render_symlink_forbidden")


def _bound_receipt_rgb(
    app_output: Path, relative: object, expected_sha256: object, role: str
) -> tuple[Path, str]:
    if not isinstance(relative, str) or not relative:
        raise _fail(f"{role}_path_invalid")
    untrusted = Path(relative)
    if untrusted.is_absolute() or ".." in untrusted.parts:
        raise _fail(f"{role}_path_invalid")
    path = _regular_file(app_output / untrusted, role, within=app_output)
    observed = _sha256(path)
    if not isinstance(expected_sha256, str) or observed != expected_sha256:
        raise _fail(f"{role}_hash_mismatch")
    return path, observed


def _load_canonical_validator(path: Path) -> Callable[[Path], dict[str, object]]:
    """Load the accepted validator source rather than restating its validation logic."""
    scripts = path.parent
    root = scripts.parent
    for entry in (os.fspath(root / "src"), os.fspath(scripts)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(
        "_fluidmix_inn67_validate_physx_f2_artifacts", path
    )
    if spec is None or spec.loader is None:
        raise _fail("validator_import_invalid")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        validator = module.validate_output
    except (AttributeError, ImportError, OSError) as error:
        raise _fail("validator_import_invalid", error)
    return validator


def _validate_result(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _RESULT_FIELDS:
        raise _fail("validator_result_invalid")
    count = value.get("verified_artifact_count")
    warnings = value.get("warnings")
    revalidations = value.get("offline_revalidations")
    if (
        value.get("status") != "pass"
        or value.get("failures") != []
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(warnings, list)
        or any(
            not isinstance(row, dict) or not isinstance(row.get("warning_type"), str)
            for row in warnings
        )
        or not isinstance(revalidations, list)
        or any(
            not isinstance(row, dict)
            or row.get("passed") is not True
            or not isinstance(row.get("revalidation_type"), str)
            for row in revalidations
        )
    ):
        raise _fail("validator_result_not_pass")
    return value


def _load_frozen_plan(
    plan_root: Path,
) -> tuple[
    dict[str, object],
    Mapping[str, str],
    dict[str, str],
    dict[str, dict[str, object]],
]:
    inventory, _ = _load_json(
        plan_root / "artifact-hashes.json", "plan_inventory", within=plan_root,
        expected_sha256=PLAN_INVENTORY_SHA256,
    )
    files = inventory.get("files")
    if (
        inventory.get("schema_version") != 1
        or inventory.get("artifact_type") != "inn67_reference_bank_plan_hash_inventory"
        or not isinstance(files, dict)
        or files.get("reference-bank-plan.json") != PLAN_SHA256
        or files.get("known-good-profile.json") != PROFILE_SHA256
    ):
        raise _fail("plan_inventory_invalid")
    plan, _ = _load_json(
        plan_root / "reference-bank-plan.json", "reference_bank_plan", within=plan_root,
        expected_sha256=PLAN_SHA256,
    )
    _load_json(
        plan_root / "known-good-profile.json", "known_good_profile", within=plan_root,
        expected_sha256=PROFILE_SHA256,
    )
    subruns = plan.get("subruns")
    cases = plan.get("cases")
    if (
        plan.get("schema_version") != 1
        or plan.get("artifact_type") != "inn67_development_reference_bank_plan"
        or plan.get("status") != "frozen_unrendered"
        or plan.get("denominators") != {
            "development_batches": 3, "palette_subruns": 6, "fill_cases": 18,
            "stage_observations": 48, "reference_donor_batches": 2,
            "config_holdout_batches": 1,
        }
        or not isinstance(subruns, list) or len(subruns) != 6
        or not isinstance(cases, list) or len(cases) != 18
    ):
        raise _fail("reference_bank_plan_invalid")
    subrun_ids = [row.get("subrun_id") for row in subruns if isinstance(row, dict)]
    case_ids = [row.get("case_id") for row in cases if isinstance(row, dict)]
    renderer_ids = [row.get("renderer_case_id") for row in cases if isinstance(row, dict)]
    if (
        len(subrun_ids) != 6 or len(set(subrun_ids)) != 6
        or not all(isinstance(value, str) and value for value in subrun_ids)
        or len(case_ids) != 18 or len(set(case_ids)) != 18
        or len(renderer_ids) != 18 or len(set(renderer_ids)) != 18
        or not all(isinstance(value, str) and value for value in [*case_ids, *renderer_ids])
    ):
        raise _fail("plan_identity_duplicate_or_invalid")
    render_plan_hashes = {}
    render_schedules = {}
    for row in subruns:
        relative = row.get("render_plan_artifact")
        if not isinstance(relative, str) or not isinstance(files.get(relative), str):
            raise _fail("subrun_plan_binding_invalid")
        subrun_plan, _ = _load_json(
            plan_root / relative, "subrun_plan", within=plan_root,
            expected_sha256=files[relative],
        )
        render_plan_hashes[row["subrun_id"]] = _canonical_sha256(subrun_plan)
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
        render_schedules[row["subrun_id"]] = by_renderer
    return plan, files, render_plan_hashes, render_schedules


def _publish(
    output: Path,
    wrappers: Mapping[str, dict[str, object]],
    inventory: dict[str, object],
) -> None:
    output_created = False
    try:
        output.mkdir(mode=0o750)
        output_created = True
        validation = output / "validation"
        validation.mkdir(mode=0o750)
        for renderer_id, value in wrappers.items():
            with (validation / f"{renderer_id}.json").open("xb") as stream:
                stream.write(
                    (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
                )
        with (output / "render-inventory.json").open("xb") as stream:
            stream.write(
                (json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            )
    except (OSError, TypeError, ValueError) as error:
        if output_created:
            try:
                shutil.rmtree(output)
            except OSError as cleanup_error:
                raise _fail("publication_cleanup_failed", cleanup_error) from error
        raise _fail("publication_failed", error)


def produce_postrender_inputs(
    *, plan_root: str | Path, render_root: str | Path, output_dir: str | Path
) -> dict[str, object]:
    """Validate all six subruns/18 cases and publish the materializer inputs."""
    frozen_root = _canonical_directory(plan_root, "plan_root")
    renders = _canonical_directory(render_root, "render_root")
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output_exists")
    if output.parent != renders:
        raise _fail("output_must_be_new_render_root_child")

    plan, _files, render_plan_hashes, render_schedules = _load_frozen_plan(
        frozen_root
    )
    validator_path = (
        Path(__file__).resolve().parents[4]
        / "cli/twin/physx" / "validate_physx_f2_artifacts.py"
    )
    _regular_file(
        validator_path,
        "validator_implementation",
        within=validator_path.parent.parent,
    )
    if _sha256(validator_path) != ACCEPTED_OCCLUSION_VALIDATOR_SHA256:
        raise _fail("validator_implementation_hash_mismatch")
    validator = _load_canonical_validator(validator_path)

    expected_subruns = {row["subrun_id"]: row for row in plan["subruns"]}
    planned_cases = plan["cases"]
    wrappers: dict[str, dict[str, object]] = {}
    inventory_rows = []
    for subrun_id, subrun in expected_subruns.items():
        subrun_root = _canonical_directory(renders / subrun_id, "subrun_root")
        if not subrun_root.is_relative_to(renders):
            raise _fail("subrun_outside_render_root")
        _reject_symlinks(subrun_root)
        plan_hash = render_plan_hashes[subrun_id]
        cases = [row for row in planned_cases if row.get("subrun_id") == subrun_id]
        if len(cases) != 3:
            raise _fail("subrun_case_denominator_invalid")
        supervisor_paths = sorted((subrun_root / "supervisor-attempts").glob("attempt-*.json"))
        if len(supervisor_paths) != 1:
            raise _fail("supervisor_receipt_missing_or_duplicate")
        supervisor, supervisor_sha = _load_json(
            supervisor_paths[0], "supervisor_receipt", within=renders
        )
        results = supervisor.get("case_results")
        expected_renderer_ids = {row["renderer_case_id"] for row in cases}
        result_ids = (
            [row.get("case_id") for row in results if isinstance(row, dict)]
            if isinstance(results, list) else []
        )
        if (
            supervisor.get("schema_version") != 1
            or supervisor.get("artifact_type")
            != "physx_train_residual_serial_supervisor_attempt"
            or supervisor.get("status") != "pass"
            or supervisor.get("plan_sha256") != plan_hash
            or supervisor.get("planned_case_count") != 3
            or supervisor.get("validated_completed_case_count") != 3
            or not isinstance(results, list)
            or len(results) != 3
            or not all(isinstance(value, str) for value in result_ids)
            or len(set(result_ids)) != 3
            or set(result_ids) != expected_renderer_ids
            or any(
                not isinstance(row, dict)
                or row.get("outcome") not in {"passed", "recognized_completed"}
                or not isinstance(row.get("evidence"), dict)
                or row["evidence"].get("artifact_validation", {}).get("status") != "pass"
                or row["evidence"].get("artifact_validation", {}).get("failures") != []
                for row in results
            )
        ):
            raise _fail("supervisor_receipt_not_accepted")

        case_rows = []
        for case in cases:
            renderer_id = case["renderer_case_id"]
            app_output = _canonical_directory(
                subrun_root / "cases" / renderer_id / "app-output", "case_app_output"
            )
            if not app_output.is_relative_to(subrun_root):
                raise _fail("case_outside_render_root")
            receipt_path = app_output / "receipt.json"
            receipt, receipt_sha = _load_json(
                receipt_path, "render_receipt", within=subrun_root
            )
            try:
                _validate_render_receipt_binding(
                    receipt,
                    planned_case=case,
                    planned_subrun=subrun,
                    planned_schedule=render_schedules[subrun_id][renderer_id],
                    renderer_id=renderer_id,
                    render_plan_sha256=plan_hash,
                )
                declared_prefill, declared_stages = _receipt_rgb_declarations(
                    receipt, case
                )
            except MaterializationError as error:
                raise _fail("render_receipt_binding_invalid", error)
            prefill_path, prefill_sha = _bound_receipt_rgb(
                app_output,
                declared_prefill["rgb_path"],
                declared_prefill["rgb_sha256"],
                "prefill_rgb",
            )
            stage_rgbs = []
            for declared_stage in declared_stages:
                rgb_path, rgb_sha = _bound_receipt_rgb(
                    app_output,
                    declared_stage["rgb_path"],
                    declared_stage["rgb_sha256"],
                    "stage_rgb",
                )
                stage_rgbs.append(
                    {
                        "stage_ordinal": declared_stage["stage_ordinal"],
                        "tank_id": declared_stage["tank_id"],
                        "rgb_path": os.fspath(rgb_path),
                        "rgb_sha256": rgb_sha,
                    }
                )
            try:
                result = _validate_result(validator(app_output))
            except PostrenderProductionError:
                raise
            except Exception as error:  # noqa: BLE001 - validator errors are evidence failures
                raise _fail("validator_execution_failed", error)
            wrapper = {
                "schema_version": 1,
                "artifact_type": "inn67_occlusion_aware_artifact_validation",
                "status": "pass",
                "failures": [],
                "case_id": renderer_id,
                "source_render_receipt_sha256": receipt_sha,
                "source_render_plan_sha256": plan_hash,
                "known_good_profile_sha256": PROFILE_SHA256,
                "validator_implementation_sha256": ACCEPTED_OCCLUSION_VALIDATOR_SHA256,
                "validator_result": result,
            }
            wrapper_path = output / "validation" / f"{renderer_id}.json"
            wrapper_body = (
                json.dumps(wrapper, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode()
            wrappers[renderer_id] = wrapper
            case_rows.append(
                {
                    "renderer_case_id": renderer_id,
                    "receipt_path": os.fspath(receipt_path),
                    "receipt_sha256": receipt_sha,
                    "validation_receipt_path": os.fspath(wrapper_path),
                    "validation_receipt_sha256": hashlib.sha256(wrapper_body).hexdigest(),
                    "prefill_rgb_path": os.fspath(prefill_path),
                    "prefill_rgb_sha256": prefill_sha,
                    "stage_rgbs": stage_rgbs,
                }
            )
        inventory_rows.append(
            {
                "subrun_id": subrun_id,
                "render_root": os.fspath(renders),
                "render_plan_sha256": plan_hash,
                "supervisor_receipt_path": os.fspath(supervisor_paths[0]),
                "supervisor_receipt_sha256": supervisor_sha,
                "cases": case_rows,
            }
        )

    if len(wrappers) != 18 or len(inventory_rows) != 6:
        raise _fail("render_denominator_invalid")
    inventory = {
        "schema_version": 1,
        "artifact_type": "inn67_accepted_render_inventory",
        "status": "accepted",
        "reference_bank_plan_sha256": PLAN_SHA256,
        "amendment_receipt_sha256": AMENDMENT_RECEIPT_SHA256,
        "subruns": inventory_rows,
    }
    if output.exists():
        raise _fail("output_exists")
    _publish(output, wrappers, inventory)
    return {
        "status": "published",
        "subrun_count": 6,
        "case_count": 18,
        "render_inventory_path": os.fspath(output / "render-inventory.json"),
    }


__all__ = ["PostrenderProductionError", "produce_postrender_inputs"]
