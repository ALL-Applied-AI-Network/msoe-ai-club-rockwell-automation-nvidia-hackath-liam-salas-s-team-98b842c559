"""CPU-only INN-67 adapter from accepted renders to a predicted evidence store.

Only hash-bound development RGB and the same-run prefill RGB enter inference.
Semantic masks, labels, recipe volumes, WP1, and video-5 are forbidden inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3

from .inn67_bank_materializer import (
    ACCEPTED_CNN_PREDICTOR, CONFIGURATION_BATCH, DONOR_BATCHES, PLAN_SHA256,
    PROFILE_SHA256, MaterializationError, _check_no_sealed_locator,
    _validate_frozen_inputs, _validate_materialization_amendment,
    _validate_render_inventory,
)

CHECKPOINT_SHA256 = "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba"
CANDIDATE_SHA256 = "f677eab5f9de95958306d4832fb2e23f2fe04e442776ac2c6a5cc23e2ffa94ff"
REPLAY_SHA256 = "a4b9d412ce063fa6449fbfcacf49b1b3ba3e7d64724fd93d96381c467ee4836a"
REVIEW_SHA256 = "17e1bb4809898c95b3af59cb6d187955a7bf36b4f6fffb39e342e07a59f2c8df"

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_FIELDS = {"schema_version", "artifact_type", "status", "render_inventory_sha256", "reference_bank_plan_sha256", "known_good_profile_sha256", "lineage", "cases"}
_CASE_FIELDS = {"case_id", "run_id", "batch_id", "recipe_id", "render_receipt_sha256", "validation_receipt_sha256", "prefill_rgb_path", "prefill_rgb_sha256", "stages"}
_STAGE_FIELDS = {"stage_ordinal", "rgb_path", "rgb_sha256"}
_LINEAGE_HASHES = {"checkpoint": CHECKPOINT_SHA256, "candidate": CANDIDATE_SHA256, "replay": REPLAY_SHA256, "review": REVIEW_SHA256}


class PredictedStoreProducerError(ValueError):
    """The INN-67 source adapter refused an unsafe or incomplete input."""


def _fail(code: str, error: BaseException | None = None) -> PredictedStoreProducerError:
    failure = PredictedStoreProducerError(code)
    if error is not None:
        failure.__cause__ = error
    return failure


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _canonical_sha(value: object) -> str:
    try:
        body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise _fail("noncanonical_value", error)
    return hashlib.sha256(body).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path_value: str | os.PathLike[str], role: str) -> tuple[dict[str, object], str]:
    path = Path(path_value).expanduser().absolute()
    try:
        if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
            raise OSError("not an exact regular file")
        body = path.read_bytes()
        value = json.loads(body, object_pairs_hook=_strict_object, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise _fail(f"{role}_invalid", error)
    if not isinstance(value, dict):
        raise _fail(f"{role}_invalid")
    return value, hashlib.sha256(body).hexdigest()


def _exact(value: object, fields: set[str], role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _fail(f"{role}_fields_invalid")
    return value


def _bound_file(path_value: object, sha_value: object, role: str) -> Path:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise _fail(f"{role}_path_invalid")
    if not isinstance(sha_value, str) or _SHA.fullmatch(sha_value) is None:
        raise _fail(f"{role}_sha256_invalid")
    path = Path(path_value).absolute()
    try:
        if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
            raise OSError("not an exact regular file")
        observed = _hash_file(path)
    except OSError as error:
        raise _fail(f"{role}_file_invalid", error)
    if observed != sha_value:
        raise _fail(f"{role}_hash_mismatch")
    return path


def _verify_lineage_file(path: Path, expected: str) -> bool:
    """Verify accepted lineage bytes; tests may replace this private seam."""
    return _hash_file(path) == expected


def _validate_source(source_path: Path, *, inventory_path: Path, plan: Mapping[str, object], receipts: Mapping[str, Mapping[str, str]]) -> tuple[list[dict[str, object]], dict[str, Path], str]:
    source, source_sha = _load_json(source_path, "source_manifest")
    _check_no_sealed_locator(source)
    _exact(source, _SOURCE_FIELDS, "source_manifest")
    if (source.get("schema_version") != 1 or source.get("artifact_type") != "inn67_prediction_source_manifest" or source.get("status") != "accepted" or source.get("render_inventory_sha256") != _hash_file(inventory_path) or source.get("reference_bank_plan_sha256") != PLAN_SHA256 or source.get("known_good_profile_sha256") != PROFILE_SHA256):
        raise _fail("source_identity_mismatch")
    lineage = source.get("lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != set(_LINEAGE_HASHES):
        raise _fail("lineage_identity_mismatch")
    lineage_paths: dict[str, Path] = {}
    for role, expected in _LINEAGE_HASHES.items():
        binding = _exact(lineage.get(role), {"path", "sha256"}, f"{role}_lineage")
        if binding.get("sha256") != expected or not isinstance(binding.get("path"), str) or not Path(str(binding["path"])).is_absolute():
            raise _fail("lineage_identity_mismatch")
        path = Path(str(binding["path"])).absolute()
        try:
            if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file() or not _verify_lineage_file(path, expected):
                raise OSError("invalid lineage binding")
        except (OSError, ValueError) as error:
            raise _fail("lineage_identity_mismatch", error)
        lineage_paths[role] = path
    raw_cases = source.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 18:
        raise _fail("source_case_denominator_invalid")
    expected_cases = {str(case["case_id"]): case for case in plan["cases"]}
    seen: set[str] = set()
    prefill_by_run: dict[str, tuple[Path, str]] = {}
    prefill_paths: dict[Path, str] = {}
    stage_paths: set[Path] = set()
    stage_hashes: set[str] = set()
    normalized: list[dict[str, object]] = []
    for raw in raw_cases:
        case = _exact(raw, _CASE_FIELDS, "source_case")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in seen or case_id not in expected_cases:
            raise _fail("source_case_denominator_invalid")
        seen.add(case_id)
        planned = expected_cases[case_id]
        receipt = receipts[case_id]
        if (case.get("run_id") != planned["run_id"] or case.get("batch_id") != planned["batch_id"] or case.get("recipe_id") != planned["recipe_id"] or case.get("render_receipt_sha256") != receipt["render_receipt_sha256"] or case.get("validation_receipt_sha256") != receipt["validation_receipt_sha256"]):
            raise _fail("source_plan_binding_mismatch")
        prefill_path = _bound_file(case.get("prefill_rgb_path"), case.get("prefill_rgb_sha256"), "prefill_rgb")
        prefill_binding = (prefill_path, str(case["prefill_rgb_sha256"]))
        run_id = str(case["run_id"])
        if run_id in prefill_by_run and prefill_by_run[run_id] != prefill_binding:
            raise _fail("same_run_prefill_drift")
        if prefill_path in prefill_paths and prefill_paths[prefill_path] != run_id:
            raise _fail("cross_run_prefill_reuse")
        prefill_by_run[run_id] = prefill_binding
        prefill_paths[prefill_path] = run_id
        raw_stages = case.get("stages")
        if not isinstance(raw_stages, list) or len(raw_stages) != len(planned["stages"]):
            raise _fail("source_stage_denominator_invalid")
        planned_stages = {stage["ordinal"]: stage for stage in planned["stages"]}
        stages, ordinals = [], set()
        for raw_stage in raw_stages:
            stage = _exact(raw_stage, _STAGE_FIELDS, "source_stage")
            ordinal = stage.get("stage_ordinal")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal in ordinals or ordinal not in planned_stages:
                raise _fail("source_stage_plan_binding_mismatch")
            rgb_path = _bound_file(stage.get("rgb_path"), stage.get("rgb_sha256"), "stage_rgb")
            rgb_sha = str(stage["rgb_sha256"])
            if rgb_path in stage_paths or rgb_sha in stage_hashes:
                raise _fail("stage_rgb_duplicate")
            stage_paths.add(rgb_path)
            stage_hashes.add(rgb_sha)
            stages.append({**dict(stage), "rgb_path": rgb_path})
            ordinals.add(ordinal)
        normalized.append({**dict(case), "prefill_rgb_path": prefill_path, "stages": stages})
    if seen != set(expected_cases):
        raise _fail("source_case_denominator_invalid")
    if stage_paths.intersection(prefill_paths):
        raise _fail("stage_rgb_aliases_prefill")
    return normalized, lineage_paths, source_sha


def _load_default_predictor(checkpoint_path: Path) -> Callable[[Path, Path], object]:
    """Load the frozen six-channel CNN lazily and keep execution on CPU."""
    try:
        import numpy as np
        import torch
        from PIL import Image
        from .cnn_predictor import ResNet18UNet, predict_pair
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model = ResNet18UNet(weights=None)
        model.load_state_dict(state, strict=True)
        model.to("cpu")
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as error:
        raise _fail("frozen_cnn_runtime_unavailable", error)
    def predict(current_path: Path, reference_path: Path) -> object:
        with Image.open(current_path) as image:
            current = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(reference_path) as image:
            reference = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return predict_pair(model, current, reference, threshold=0.5, device="cpu").mask
    return predict


def _write_mask(path: Path, prediction: object, rgb_path: Path) -> str:
    import numpy as np
    from PIL import Image
    values = np.asarray(prediction)
    try:
        with Image.open(rgb_path) as image:
            expected_shape = (image.height, image.width)
    except OSError as error:
        raise _fail("stage_rgb_decode_failed", error)
    if values.dtype != np.bool_ or values.shape != expected_shape or not bool(np.any(values)):
        raise _fail("predicted_mask_invalid")
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path, format="PNG")
    return _hash_file(path)


def _create_database(path: Path, records: list[dict[str, object]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript("""PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT; CREATE TABLE predicted_records (event_cursor INTEGER PRIMARY KEY, evidence_id TEXT NOT NULL UNIQUE, record_json TEXT NOT NULL, record_sha256 TEXT NOT NULL UNIQUE) STRICT;""")
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("predictor", _canonical_sha(ACCEPTED_CNN_PREDICTOR)))
        for record in records:
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            connection.execute("INSERT INTO predicted_records VALUES (?, ?, ?, ?)", (record["event_cursor"], record["evidence_id"], encoded, hashlib.sha256(encoded.encode()).hexdigest()))
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise _fail("evidence_store_integrity_failed")
    finally:
        connection.close()


def produce_predicted_store(*, plan_root: str | os.PathLike[str], amendment_path: str | os.PathLike[str], materialization_amendment_path: str | os.PathLike[str], render_inventory_path: str | os.PathLike[str], source_manifest_path: str | os.PathLike[str], output_dir: str | os.PathLike[str], predictor: Callable[[Path, Path], object] | None = None) -> dict[str, object]:
    """Publish a fresh 48-row SQLite store, receipt, and consumer export."""
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output_exists")
    try:
        frozen = _validate_frozen_inputs(Path(plan_root), Path(amendment_path))
        _validate_materialization_amendment(Path(materialization_amendment_path))
        inventory_path = Path(render_inventory_path).expanduser().absolute()
        receipts = _validate_render_inventory(inventory_path, frozen)
        source_cases, lineage_paths, source_sha = _validate_source(Path(source_manifest_path), inventory_path=inventory_path, plan=frozen["plan"], receipts=receipts)
    except MaterializationError as error:
        raise _fail(str(error), error) from error
    by_case = {str(case["case_id"]): case for case in source_cases}
    planned_cases = list(frozen["plan"]["cases"])
    ordered_cases = [case for batch in (*DONOR_BATCHES, CONFIGURATION_BATCH) for case in planned_cases if case["batch_id"] == batch]
    if len(ordered_cases) != 18:
        raise _fail("plan_case_denominator_invalid")
    try:
        output.mkdir(mode=0o750, parents=True, exist_ok=False)
        masks = output / "predicted-masks"
        masks.mkdir(mode=0o750)
        active_predictor = predictor or _load_default_predictor(lineage_paths["checkpoint"])
        locator_images: dict[str, dict[str, str]] = {}
        pending: list[tuple[Mapping[str, object], Mapping[str, object], str]] = []
        for planned in ordered_cases:
            source_case = by_case[str(planned["case_id"])]
            by_ordinal = {stage["stage_ordinal"]: stage for stage in source_case["stages"]}
            for stage in planned["stages"]:
                source_stage = by_ordinal[stage["ordinal"]]
                image_id = f"INN67-{planned['case_id']}-STAGE-{stage['ordinal']:02d}"
                mask_path = masks / f"{image_id}.png"
                mask_sha = _write_mask(mask_path, active_predictor(source_stage["rgb_path"], source_case["prefill_rgb_path"]), source_stage["rgb_path"])
                locator_images[image_id] = {"rgb_path": str(source_stage["rgb_path"]), "rgb_sha256": str(source_stage["rgb_sha256"]), "mask_path": str(mask_path), "mask_sha256": mask_sha}
                pending.append((planned, stage, image_id))
        if len(pending) != 48:
            raise _fail("record_denominator_invalid")
        locator_artifact = {"schema_version": 2, "artifact_type": "rgb_liquid_mask_prediction_locators", "data_origin": "twin", "mask_source": "rgb_only_prediction", "producer": deepcopy(ACCEPTED_CNN_PREDICTOR), "images": locator_images}
        from .rgb_mask_prediction import build_trusted_predicted_mask_registry_v1
        from .vision_measurement import measure_trusted_predicted_image
        registry = build_trusted_predicted_mask_registry_v1(locator_artifact)
        records: list[dict[str, object]] = []
        for event_cursor, (planned, stage, image_id) in enumerate(pending):
            evidence_id = f"EVIDENCE-{image_id}"
            measurement, binding = measure_trusted_predicted_image(registry.entry(image_id), measurement_id=f"MEASUREMENT-{image_id}", evidence_id=evidence_id)
            lab = measurement.get("lab_raw")
            if (measurement.get("validity") != "valid" or not isinstance(lab, list) or len(lab) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in lab) or binding.evidence_id != evidence_id):
                raise _fail("predicted_measurement_invalid")
            source_case, locator, receipt = by_case[str(planned["case_id"])], locator_images[image_id], receipts[str(planned["case_id"])]
            records.append({"evidence_id": evidence_id, "event_cursor": event_cursor, "batch_id": planned["batch_id"], "run_id": planned["run_id"], "case_id": planned["case_id"], "recipe_id": planned["recipe_id"], "split_role": "development", "stage_id": f"dispense-{stage['ordinal']}-{stage['tank_id']}", "stage_ordinal": stage["ordinal"], "tank_id": stage["tank_id"], "rgb_sha256": locator["rgb_sha256"], "mask_sha256": locator["mask_sha256"], "mask_source": "predicted", "median_lab": [float(value) for value in lab], "source_render_receipt_sha256": source_case["render_receipt_sha256"], "source_validation_receipt_sha256": source_case["validation_receipt_sha256"], "source_render_plan_sha256": receipt["render_plan_sha256"], "known_good_profile_sha256": PROFILE_SHA256})
        database_path = output / "evidence.sqlite3"
        _create_database(database_path, records)
        database_sha = _hash_file(database_path)
        receipt = {"schema_version": 1, "artifact_type": "inn67_predicted_evidence_store_population", "status": "pass", "store_sha256": database_sha, "predictor": deepcopy(ACCEPTED_CNN_PREDICTOR), "lineage": {f"{role}_sha256": digest for role, digest in _LINEAGE_HASHES.items()}, "source_bindings": {"source_manifest_sha256": source_sha, "render_inventory_sha256": _hash_file(inventory_path), "reference_bank_plan_sha256": PLAN_SHA256, "known_good_profile_sha256": PROFILE_SHA256}, "record_count": 48, "records_sha256": _canonical_sha(records), "evidence_ids": [row["evidence_id"] for row in records]}
        receipt_path = output / "population-receipt.json"
        receipt_path.write_bytes(_json_bytes(receipt))
        receipt_sha = _hash_file(receipt_path)
        export = {"schema_version": 1, "artifact_type": "inn67_predicted_store_export", "status": "accepted", "predictor": deepcopy(ACCEPTED_CNN_PREDICTOR), "store": {"database_path": str(database_path), "database_sha256": database_sha, "population_receipt_path": str(receipt_path), "population_receipt_sha256": receipt_sha}, "records": records}
        export_path = output / "predicted-store-export.json"
        export_path.write_bytes(_json_bytes(export))
        return {"status": "published", "record_count": 48, "database_sha256": database_sha, "population_receipt_sha256": receipt_sha, "predicted_store_export_sha256": _hash_file(export_path), "source_manifest_sha256": source_sha, "trusted_registry_sha256": registry.registry_sha256}
    except BaseException:
        if output.exists() and output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise


__all__ = ["CANDIDATE_SHA256", "CHECKPOINT_SHA256", "REPLAY_SHA256", "REVIEW_SHA256", "PredictedStoreProducerError", "produce_predicted_store"]
