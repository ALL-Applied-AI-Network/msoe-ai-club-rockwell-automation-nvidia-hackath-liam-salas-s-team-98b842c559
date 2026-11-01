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

from .materializer import (
    ACCEPTED_CNN_PREDICTOR,
    CONFIGURATION_BATCH,
    DONOR_BATCHES,
    PLAN_SHA256,
    PROFILE_SHA256,
    MaterializationError,
    _validate_frozen_inputs,
    _validate_materialization_amendment,
    _validate_render_inventory,
)

CHECKPOINT_SHA256 = "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba"
CANDIDATE_RECEIPT_SHA256 = "27a4e49553f9fc0d34832e8f8ebc73f11fbf7dfa7e9c39f50353b1fae4b28727"
REPLAY_SHA256 = "a4b9d412ce063fa6449fbfcacf49b1b3ba3e7d64724fd93d96381c467ee4836a"
REVIEW_SHA256 = "17e1bb4809898c95b3af59cb6d187955a7bf36b4f6fffb39e342e07a59f2c8df"
CNN_PREDICTOR_SOURCE_SHA256 = "ec9f4b589f088fe979f084ebff1ad51cb0fe0dd29cf12a64a7c993bab534be19"
EMPTY_REFERENCE_SOURCE_SHA256 = "1e933c6eb8e6a729c820211c190aaa72f8d5ea203706e503484e3531800d80a1"

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LINEAGE_HASHES = {
    "checkpoint": CHECKPOINT_SHA256,
    "candidate_receipt": CANDIDATE_RECEIPT_SHA256,
    "deterministic_replay": REPLAY_SHA256,
    "independent_review": REVIEW_SHA256,
}
_ACCEPTED_CNN_CONFIG = {
    "architecture": "six-channel-resnet18-unet",
    "candidate_receipt_sha256": CANDIDATE_RECEIPT_SHA256,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "deterministic_replay_sha256": REPLAY_SHA256,
    "independent_review_sha256": REVIEW_SHA256,
    "probability_threshold": 0.5,
    "postprocessing": "unchanged_approach_a_vial_roi_and_clamp_raster",
}


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


def _verified_lineage_path(path_value: object, expected: str, role: str) -> Path:
    if not isinstance(path_value, (str, os.PathLike)):
        raise _fail("lineage_identity_mismatch")
    path = Path(path_value).expanduser().absolute()
    try:
        if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
            raise OSError("invalid lineage file")
        if not _verify_lineage_file(path, expected):
            raise OSError("lineage hash mismatch")
    except OSError as error:
        raise _fail("lineage_identity_mismatch", error)
    return path


def _validate_lineage(
    checkpoint_path: str | os.PathLike[str],
    replay_path: str | os.PathLike[str],
    review_path: str | os.PathLike[str],
) -> dict[str, Path]:
    predictor_source = (
        Path(__file__).resolve().parents[4]
        / "src" / "fluidmix" / "perception" / "cnn" / "cnn_predictor.py"
    )
    empty_source = (
        Path(__file__).resolve().parents[4]
        / "src" / "fluidmix" / "perception" / "baselines"
        / "empty_reference_rgb_predictor.py"
    )
    if (
        _hash_file(predictor_source) != CNN_PREDICTOR_SOURCE_SHA256
        or _hash_file(empty_source) != EMPTY_REFERENCE_SOURCE_SHA256
        or _canonical_sha(_ACCEPTED_CNN_CONFIG)
        != ACCEPTED_CNN_PREDICTOR["config_sha256"]
    ):
        raise _fail("accepted_cnn_runtime_source_mismatch")
    checkpoint = _verified_lineage_path(checkpoint_path, CHECKPOINT_SHA256, "checkpoint")
    replay_file = _verified_lineage_path(replay_path, REPLAY_SHA256, "deterministic_replay")
    review_file = _verified_lineage_path(review_path, REVIEW_SHA256, "independent_review")
    replay, _ = _load_json(replay_file, "deterministic_replay")
    scoring = replay.get("scoring")
    sources = replay.get("source_bindings")
    if (
        replay.get("artifact_type") != "wp5_cnn_deterministic_bounds_replay"
        or replay.get("status") != "swap_review_ready"
        or replay.get("checkpoint_frozen") is not True
        or replay.get("threshold_frozen") != 0.5
        or not isinstance(scoring, Mapping)
        or any(
            not isinstance(scoring.get(scope), Mapping)
            or scoring[scope].get("all_gates_pass") is not True
            or scoring[scope].get("decision") != "replace_approach_a"
            for scope in ("all", "cleaned")
        )
        or not isinstance(sources, Mapping)
    ):
        raise _fail("deterministic_replay_invalid")
    checkpoint_binding = sources.get("checkpoint")
    candidate_binding = sources.get("candidate_receipt")
    if (
        not isinstance(checkpoint_binding, Mapping)
        or checkpoint_binding.get("sha256") != CHECKPOINT_SHA256
        or not isinstance(candidate_binding, Mapping)
        or candidate_binding.get("sha256") != CANDIDATE_RECEIPT_SHA256
    ):
        raise _fail("lineage_identity_mismatch")
    try:
        replay_checkpoint = Path(str(checkpoint_binding.get("path"))).resolve(strict=True)
    except OSError as error:
        raise _fail("lineage_identity_mismatch", error)
    if replay_checkpoint != checkpoint:
        raise _fail("lineage_identity_mismatch")
    candidate = _verified_lineage_path(
        candidate_binding.get("path"), CANDIDATE_RECEIPT_SHA256, "candidate_receipt"
    )
    candidate_value, _ = _load_json(candidate, "candidate_receipt")
    if (
        candidate_value.get("artifact_type") != "wp5_cnn_frozen_candidate"
        or candidate_value.get("source_commit") != ACCEPTED_CNN_PREDICTOR["code_commit"]
        or candidate_value.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or candidate_value.get("threshold") != 0.5
    ):
        raise _fail("candidate_receipt_invalid")
    try:
        if not review_file.read_text(encoding="utf-8").lstrip().startswith("# Independent"):
            raise ValueError("review heading drifted")
    except (OSError, UnicodeError, ValueError) as error:
        raise _fail("independent_review_invalid", error)
    return {
        "checkpoint": checkpoint,
        "candidate_receipt": candidate,
        "deterministic_replay": replay_file,
        "independent_review": review_file,
    }


def _bound_receipt_rgb(root: Path, path_value: object, sha_value: object, role: str) -> Path:
    if not isinstance(path_value, str) or Path(path_value).is_absolute() or ".." in Path(path_value).parts:
        raise _fail(f"{role}_path_invalid")
    parts = {part.lower() for part in Path(path_value).parts}
    if parts.intersection({"oracle", "oracles", "semantic", "semantics", "mask", "masks", "label", "labels"}):
        raise _fail("forbidden_semantic_or_oracle_locator")
    lower = path_value.lower()
    if any(token in lower for token in ("wp1", "video5", "video-5", "video_5")):
        raise _fail("forbidden_test_or_video5_locator")
    path = (root / path_value).absolute()
    _bound_file(str(path), sha_value, role)
    if not path.is_relative_to(root):
        raise _fail(f"{role}_path_invalid")
    try:
        from PIL import Image
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (640, 480):
                raise ValueError("not exact renderer RGB")
    except (OSError, ValueError) as error:
        raise _fail(f"{role}_rgb_invalid", error)
    return path


def _derive_sources(
    inventory_path: Path,
    *,
    plan: Mapping[str, object],
    receipts: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, object]], str]:
    inventory, inventory_sha = _load_json(inventory_path, "render_inventory")
    inventory_rows = {
        str(case["renderer_case_id"]): case
        for subrun in inventory["subruns"]
        for case in subrun["cases"]
    }
    stage_paths: set[Path] = set()
    stage_hashes: set[str] = set()
    sources: list[dict[str, object]] = []
    for planned in plan["cases"]:
        renderer_id = str(planned["renderer_case_id"])
        inventory_row = inventory_rows.get(renderer_id)
        if not isinstance(inventory_row, Mapping):
            raise _fail("receipt_case_denominator_invalid")
        receipt_path_value = inventory_row.get("receipt_path")
        if not isinstance(receipt_path_value, str):
            raise _fail("render_receipt_path_invalid")
        receipt_path = Path(receipt_path_value).absolute()
        receipt, receipt_sha = _load_json(receipt_path, "render_receipt")
        if receipt_sha != receipts[str(planned["case_id"])]["render_receipt_sha256"]:
            raise _fail("render_receipt_hash_mismatch")
        arguments = receipt.get("arguments")
        schedule = receipt.get("execution_schedule")
        if (
            not isinstance(arguments, Mapping)
            or arguments.get("train_case_id") != renderer_id
            or not isinstance(schedule, Mapping)
            or schedule.get("case_id") != renderer_id
            or not isinstance(schedule.get("model_input_record"), Mapping)
            or schedule["model_input_record"].get("case_id") != renderer_id
            or not isinstance(schedule.get("renderer_case_binding"), Mapping)
            or schedule["renderer_case_binding"].get("case_id") != renderer_id
            or arguments.get("train_render_plan_sha256")
            != receipts[str(planned["case_id"])]["render_plan_sha256"]
            or schedule.get("cohort_plan_sha256")
            != receipts[str(planned["case_id"])]["render_plan_sha256"]
        ):
            raise _fail("receipt_plan_binding_mismatch")
        parity = receipt.get("pre_pour_parity")
        if not isinstance(parity, Mapping):
            raise _fail("prefill_binding_missing")
        root = receipt_path.parent
        prefill = _bound_receipt_rgb(
            root, parity.get("f2_prefill_path"), parity.get("f2_prefill_sha256"), "prefill_rgb"
        )
        semantics = receipt.get("semantics")
        pairs = semantics.get("pairs") if isinstance(semantics, Mapping) else None
        if not isinstance(pairs, list):
            raise _fail("stage_rgb_binding_missing")
        expected_phases = [f"{stage['tank_id']}-stage" for stage in planned["stages"]]
        observed_stage_pairs = [
            pair for pair in pairs
            if isinstance(pair, Mapping)
            and isinstance(pair.get("phase"), str)
            and str(pair["phase"]).endswith("-stage")
        ]
        if [pair.get("phase") for pair in observed_stage_pairs] != expected_phases:
            raise _fail("stage_rgb_binding_missing")
        stages: list[dict[str, object]] = []
        for stage, pair in zip(planned["stages"], observed_stage_pairs, strict=True):
            rgb = _bound_receipt_rgb(root, pair.get("rgb_path"), pair.get("rgb_sha256"), "stage_rgb")
            rgb_sha = str(pair["rgb_sha256"])
            if rgb == prefill or rgb in stage_paths or rgb_sha in stage_hashes:
                raise _fail("stage_rgb_duplicate")
            stage_paths.add(rgb)
            stage_hashes.add(rgb_sha)
            stages.append({"stage_ordinal": stage["ordinal"], "rgb_path": rgb, "rgb_sha256": rgb_sha})
        if len(stages) != len(planned["stages"]):
            raise _fail("source_stage_denominator_invalid")
        sources.append({
            "case_id": planned["case_id"],
            "run_id": planned["run_id"],
            "batch_id": planned["batch_id"],
            "recipe_id": planned["recipe_id"],
            "render_receipt_sha256": receipt_sha,
            "validation_receipt_sha256": receipts[str(planned["case_id"])]["validation_receipt_sha256"],
            "prefill_rgb_path": prefill,
            "prefill_rgb_sha256": parity["f2_prefill_sha256"],
            "stages": stages,
        })
    if len(sources) != 18 or sum(len(case["stages"]) for case in sources) != 48:
        raise _fail("source_case_denominator_invalid")
    return sources, inventory_sha


def _load_default_predictor(checkpoint_path: Path) -> Callable[[Path, Path], object]:
    """Load the frozen six-channel CNN lazily and keep execution on CPU."""
    try:
        import numpy as np
        import torch
        from PIL import Image
        from ...perception.cnn.cnn_predictor import ResNet18UNet, predict_pair
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


def produce_predicted_store(*, plan_root: str | os.PathLike[str], amendment_path: str | os.PathLike[str], materialization_amendment_path: str | os.PathLike[str], render_inventory_path: str | os.PathLike[str], cnn_checkpoint_path: str | os.PathLike[str], cnn_replay_path: str | os.PathLike[str], cnn_review_path: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict[str, object]:
    """Publish a fresh 48-row SQLite store, receipt, and consumer export."""
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise _fail("output_exists")
    try:
        frozen = _validate_frozen_inputs(Path(plan_root), Path(amendment_path))
        _validate_materialization_amendment(Path(materialization_amendment_path))
        inventory_path = Path(render_inventory_path).expanduser().absolute()
        receipts = _validate_render_inventory(inventory_path, frozen)
        lineage_paths = _validate_lineage(cnn_checkpoint_path, cnn_replay_path, cnn_review_path)
        source_cases, source_sha = _derive_sources(
            inventory_path, plan=frozen["plan"], receipts=receipts
        )
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
        derived_source = {
            "schema_version": 1,
            "artifact_type": "inn67_receipt_derived_prediction_sources",
            "status": "derived_from_accepted_receipts",
            "render_inventory_sha256": source_sha,
            "reference_bank_plan_sha256": PLAN_SHA256,
            "known_good_profile_sha256": PROFILE_SHA256,
            "case_count": 18,
            "stage_count": 48,
            "cases": [
                {
                    **{key: value for key, value in case.items() if key not in {"prefill_rgb_path", "stages"}},
                    "prefill_rgb_path": str(case["prefill_rgb_path"]),
                    "stages": [
                        {**stage, "rgb_path": str(stage["rgb_path"])}
                        for stage in case["stages"]
                    ],
                }
                for case in source_cases
            ],
        }
        source_manifest_path = output / "derived-source-manifest.json"
        source_manifest_path.write_bytes(_json_bytes(derived_source))
        source_manifest_sha = _hash_file(source_manifest_path)
        active_predictor = _load_default_predictor(lineage_paths["checkpoint"])
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
        from ...perception.masks import build_trusted_predicted_mask_registry_v1
        from ...measurement.vision_measurement import measure_trusted_predicted_image
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
        receipt = {
            "schema_version": 1,
            "artifact_type": "inn67_predicted_evidence_store_population",
            "status": "pass",
            "store_sha256": database_sha,
            "predictor": deepcopy(ACCEPTED_CNN_PREDICTOR),
            "lineage": {
                **{
                    f"{role}_sha256": digest
                    for role, digest in _LINEAGE_HASHES.items()
                },
                "predictor_source_sha256": CNN_PREDICTOR_SOURCE_SHA256,
                "empty_reference_source_sha256": EMPTY_REFERENCE_SOURCE_SHA256,
                "config_sha256": ACCEPTED_CNN_PREDICTOR["config_sha256"],
            },
            "source_bindings": {
                "derived_source_manifest_path": str(source_manifest_path),
                "derived_source_manifest_sha256": source_manifest_sha,
                "render_inventory_sha256": source_sha,
                "reference_bank_plan_sha256": PLAN_SHA256,
                "known_good_profile_sha256": PROFILE_SHA256,
            },
            "record_count": 48,
            "records_sha256": _canonical_sha(records),
            "evidence_ids": [row["evidence_id"] for row in records],
        }
        receipt_path = output / "population-receipt.json"
        receipt_path.write_bytes(_json_bytes(receipt))
        receipt_sha = _hash_file(receipt_path)
        export = {"schema_version": 1, "artifact_type": "inn67_predicted_store_export", "status": "accepted", "predictor": deepcopy(ACCEPTED_CNN_PREDICTOR), "store": {"database_path": str(database_path), "database_sha256": database_sha, "population_receipt_path": str(receipt_path), "population_receipt_sha256": receipt_sha}, "records": records}
        export_path = output / "predicted-store-export.json"
        export_path.write_bytes(_json_bytes(export))
        return {"status": "published", "record_count": 48, "database_sha256": database_sha, "population_receipt_sha256": receipt_sha, "predicted_store_export_sha256": _hash_file(export_path), "derived_source_manifest_sha256": source_manifest_sha, "render_inventory_sha256": source_sha, "trusted_registry_sha256": registry.registry_sha256}
    except BaseException:
        if output.exists() and output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise


__all__ = [
    "CANDIDATE_RECEIPT_SHA256",
    "CHECKPOINT_SHA256",
    "CNN_PREDICTOR_SOURCE_SHA256",
    "EMPTY_REFERENCE_SOURCE_SHA256",
    "REPLAY_SHA256",
    "REVIEW_SHA256",
    "PredictedStoreProducerError",
    "produce_predicted_store",
]
