"""Preparation-only WP7 population from accepted synthetic development evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3

from fluidmix.core.io import sha256_file as _file_hash
from ...core.contracts._common import bounded_control_free_text, exact_lower_sha256


ACCEPTED_SOURCE_ROOT = Path("/home/hackathon5/liamreporoot")
ACCEPTED_RECEIPT = ACCEPTED_SOURCE_ROOT / (
    "outputs/pipeline-20261028/"
    "wp5-cnn-swap-sixth-dev-blocker-fix-ff0fab6-01a18128-attempt01/"
    "population-receipt.json"
)
ACCEPTED_DATABASE = ACCEPTED_RECEIPT.with_name("evidence.sqlite3")
ACCEPTED_REVIEW = ACCEPTED_SOURCE_ROOT / (
    "outputs/pipeline-20261028/wp5-cnn-swap-review-01a18128/"
    "independent-rereview-ff0fab6.md"
)
ACCEPTED_RECEIPT_SHA256 = (
    "836001f775d20e4835ed0cdee4f9197992b7a79def75ae1e06c9863c8e207810"
)
ACCEPTED_DATABASE_SHA256 = (
    "b83deef57e2c17d4b8cc068440c763729f9ebe3a1c078711516f57bc732dc793"
)
ACCEPTED_REVIEW_SHA256 = (
    "0c7f843764179e0533d151320cca461ba9d5cff3b6b96965493754704dbdb63a"
)
ACCEPTED_ORDERED_VIDEO = ACCEPTED_SOURCE_ROOT / (
    "outputs/pipeline-20261028/wp7-ordered-test-video-01a18128-attempt01/"
    "wp7-ordered-test-video.mp4"
)
ACCEPTED_ORDERED_VIDEO_SHA256 = (
    "1021878c646f1019ac941ab645520446441d6f401ff664d05d5a4e64a682760b"
)
ACCEPTED_COMPOSITION_RECEIPT = ACCEPTED_ORDERED_VIDEO.with_suffix(".json")
ACCEPTED_COMPOSITION_RECEIPT_SHA256 = (
    "0a88c47515af9fb0e389900f32504994914fe48bcc1e06bc392185382116b037"
)
ACCEPTED_ORDERED_VIDEO_REVIEW = ACCEPTED_SOURCE_ROOT / (
    "outputs/pipeline-20261028/"
    "wp7-ordered-test-video-final-review-01a18128-attempt01/FINAL-REVIEW.json"
)
ACCEPTED_ORDERED_VIDEO_REVIEW_SHA256 = (
    "ae4f9b2fdc3c6341a77dfa25044b673d7357155a98795ffa0f2df8b93ddec683"
)
_BASE_LIMITS = (
    "frozen_decision_config_unavailable",
    "matched_peer_batches_unavailable",
)
_MODEL_REVISION = "deterministic-template:wp7-development-history-preparation-v1"
_INN67_DECISION_RULE = (
    "peer_delta = dE76(B01,B02); threshold = peer_delta/2 + 1.0; "
    "require peer_delta <= 2.0 and B03 target_to_center_delta_e76 <= threshold"
)
_INN67_THRESHOLD_FORMULA = (
    "threshold_delta_e76 = peer_pair_delta_e76 / 2 + 1.0"
)
_INN67_CONFIGURATION_GATE = (
    "all 16 B03 recipe-stage targets must be consistent_with_two_peer_mode"
)
_INN67_ARITHMETIC_TOLERANCE = 1e-12
_INN67_ACCEPTED_PREDICTOR = {
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
_INN67_RECORD_FIELDS = {
    "evidence_id",
    "event_cursor",
    "batch_id",
    "run_id",
    "case_id",
    "recipe_id",
    "split_role",
    "stage_id",
    "stage_ordinal",
    "tank_id",
    "rgb_sha256",
    "mask_sha256",
    "mask_source",
    "median_lab",
    "source_render_receipt_sha256",
    "source_validation_receipt_sha256",
    "source_render_plan_sha256",
    "known_good_profile_sha256",
}
_INN67_PROVENANCE_FIXED = {
    "reference_bank_plan_sha256": "e015a0d02aef6fce283f85a51ea4ab3ff9bde2fb01293a5a3d2b42ec08ad765b",
    "reference_bank_inventory_sha256": "f0ab668df8bb2f4abe8708a486ee00a913e129772a83a4d1abd2b95d28d467d7",
    "pre_access_amendment_file_sha256": "5a39a1c8de636d0d0de83440a8483c2ec6a692b1722c1a77ac6ec613bee93128",
    "pre_access_amendment_receipt_sha256": "3cc4212f84a6a011b0d3dd579cb6cf817285d3a946598908756f6c95c9f349db",
    "materialization_protocol_amendment_sha256": "2f39a4507d429902e241fff0b5709d7c646a702e8d8331a54c9c941f62ccb4bd",
}
_INN67_PROVENANCE_FIELDS = set(_INN67_PROVENANCE_FIXED) | {
    "render_inventory_sha256",
    "predicted_store_export_sha256",
    "accepted_store_tuple",
}
_INN67_STORE_FIELDS = {
    "database_path",
    "database_sha256",
    "population_receipt_path",
    "population_receipt_sha256",
}
_INN67_KNOWN_GOOD_PROFILE_SHA256 = (
    "47bc0b6f483a121a28871b3eb74ec27efb79996b83d03d8c45d6f5b0b9cdeb7c"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _publish_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(value) + b"\n")


def _strict_json(path: Path, role: str) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{role} is not readable strict JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{role} must be a JSON object")
    return value


def _validate_inn67_provenance(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _INN67_PROVENANCE_FIELDS:
        raise ValueError("INN-67 provenance contract mismatch")
    if any(
        value.get(field) != digest
        for field, digest in _INN67_PROVENANCE_FIXED.items()
    ):
        raise ValueError("INN-67 provenance frozen-source identity mismatch")
    for field in ("render_inventory_sha256", "predicted_store_export_sha256"):
        try:
            exact_lower_sha256(value.get(field), f"INN-67 provenance {field}")
        except (TypeError, ValueError) as error:
            raise ValueError("INN-67 provenance hash mismatch") from error
    store = value.get("accepted_store_tuple")
    if not isinstance(store, Mapping) or set(store) != _INN67_STORE_FIELDS:
        raise ValueError("INN-67 provenance accepted-store tuple mismatch")
    for field in ("database_sha256", "population_receipt_sha256"):
        try:
            exact_lower_sha256(store.get(field), f"INN-67 provenance {field}")
        except (TypeError, ValueError) as error:
            raise ValueError("INN-67 provenance accepted-store hash mismatch") from error
    database_path = store.get("database_path")
    population_path = store.get("population_receipt_path")
    if (
        not isinstance(database_path, str)
        or not Path(database_path).is_absolute()
        or Path(database_path).suffix != ".sqlite3"
        or not isinstance(population_path, str)
        or not Path(population_path).is_absolute()
        or Path(population_path).name != "population-receipt.json"
    ):
        raise ValueError("INN-67 provenance accepted-store path identity mismatch")
    return value


def _validate_inn67_binding(
    *,
    materialization_receipt: Path,
    expected_materialization_receipt_sha256: str,
    development_reference_bank: Path,
    expected_development_reference_bank_sha256: str,
    delta_e_frozen_config: Path,
    expected_delta_e_frozen_config_sha256: str,
) -> dict[str, object]:
    artifacts = (
        (
            Path(materialization_receipt),
            expected_materialization_receipt_sha256,
            "INN-67 materialization receipt",
        ),
        (
            Path(development_reference_bank),
            expected_development_reference_bank_sha256,
            "INN-67 development reference bank",
        ),
        (
            Path(delta_e_frozen_config),
            expected_delta_e_frozen_config_sha256,
            "INN-67 frozen config",
        ),
    )
    resolved: dict[str, Path] = {}
    for path, expected_sha256, role in artifacts:
        exact_lower_sha256(expected_sha256, f"expected {role} SHA-256")
        try:
            exact_path = path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{role} is not an existing file") from error
        if not exact_path.is_file() or path.is_symlink():
            raise ValueError(f"{role} must be a regular non-symlink file")
        if _file_hash(exact_path) != expected_sha256:
            raise ValueError(f"{role} SHA-256 mismatch")
        resolved[role] = exact_path

    receipt = _strict_json(
        resolved["INN-67 materialization receipt"],
        "INN-67 materialization receipt",
    )
    bank = _strict_json(
        resolved["INN-67 development reference bank"],
        "INN-67 development reference bank",
    )
    config = _strict_json(
        resolved["INN-67 frozen config"],
        "INN-67 frozen config",
    )
    receipt_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "cpu_only",
        "gpu_or_renderer_launched",
        "image_or_video_bytes_opened",
        "wp1_test_accessed",
        "video5_accessed",
        "outputs",
        "provenance",
    }
    no_access = {
        "cpu_only": True,
        "gpu_or_renderer_launched": False,
        "image_or_video_bytes_opened": False,
        "wp1_test_accessed": False,
        "video5_accessed": False,
    }
    outputs = receipt.get("outputs")
    provenance = receipt.get("provenance")
    if (
        set(receipt) != receipt_fields
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_type")
        != "inn67_reference_bank_materialization_receipt"
        or receipt.get("status") != "published"
        or any(receipt.get(key) is not value for key, value in no_access.items())
        or outputs
        != {
            "development-reference-bank.json": expected_development_reference_bank_sha256,
            "delta-e-frozen-config.json": expected_delta_e_frozen_config_sha256,
        }
        or not isinstance(provenance, Mapping)
    ):
        raise ValueError("INN-67 materialization receipt contract mismatch")
    provenance = _validate_inn67_provenance(provenance)

    bank_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "split_role",
        "reference_donor_batch_ids",
        "configuration_target_batch_id",
        "predictor",
        "denominators",
        "records",
        "history_bindings",
        "claim_boundary",
        "provenance",
    }
    donor_batches = bank.get("reference_donor_batch_ids")
    predictor = bank.get("predictor")
    denominators = bank.get("denominators")
    records = bank.get("records")
    bindings = bank.get("history_bindings")
    expected_denominators = {
        "donor_batches": 2,
        "donor_records": 32,
        "configuration_targets": 16,
        "peers_per_configuration_target": 2,
    }
    if (
        set(bank) != bank_fields
        or bank.get("schema_version") != 1
        or bank.get("artifact_type") != "inn67_development_reference_bank"
        or bank.get("status") != "accepted_development_only"
        or bank.get("split_role") != "development"
        or donor_batches != ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"]
        or len(set(donor_batches)) != 2
        or bank.get("configuration_target_batch_id") != "INN67-DEV-RB-B03"
        or not isinstance(predictor, Mapping)
        or predictor != _INN67_ACCEPTED_PREDICTOR
        or denominators != expected_denominators
        or not isinstance(records, list)
        or len(records) != 32
        or not isinstance(bindings, list)
        or len(bindings) != 16
        or bank.get("provenance") != provenance
    ):
        raise ValueError("INN-67 development reference bank contract mismatch")
    record_by_id: dict[str, Mapping[str, object]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("INN-67 bank record is not an object")
        evidence_id = row.get("evidence_id")
        batch_id = row.get("batch_id")
        string_fields = (
            "evidence_id",
            "batch_id",
            "run_id",
            "case_id",
            "recipe_id",
            "stage_id",
            "tank_id",
        )
        sha_fields = (
            "rgb_sha256",
            "mask_sha256",
            "source_render_receipt_sha256",
            "source_validation_receipt_sha256",
            "source_render_plan_sha256",
            "known_good_profile_sha256",
        )
        median_lab = row.get("median_lab")
        if (
            set(row) != _INN67_RECORD_FIELDS
            or any(
                not isinstance(row.get(field), str) or not row[field]
                for field in string_fields
            )
            or not isinstance(evidence_id, str)
            or not evidence_id
            or evidence_id in record_by_id
            or batch_id not in donor_batches
            or row.get("split_role") != "development"
            or row.get("mask_source") != "predicted"
            or type(row.get("event_cursor")) is not int
            or row.get("event_cursor", -1) < 0
            or type(row.get("stage_ordinal")) is not int
            or row.get("stage_ordinal") not in {1, 2, 3}
            or not isinstance(median_lab, list)
            or len(median_lab) != 3
            or any(
                type(channel) not in {int, float} or not math.isfinite(float(channel))
                for channel in median_lab
            )
            or row.get("known_good_profile_sha256")
            != _INN67_KNOWN_GOOD_PROFILE_SHA256
        ):
            raise ValueError("INN-67 bank record schema or predicted-source contract mismatch")
        try:
            for field in sha_fields:
                exact_lower_sha256(row[field], f"INN-67 bank record {field}")
        except (TypeError, ValueError) as error:
            raise ValueError("INN-67 bank record source hash mismatch") from error
        record_by_id[evidence_id] = row
    if Counter(row.get("batch_id") for row in records) != Counter(
        {"INN67-DEV-RB-B01": 16, "INN67-DEV-RB-B02": 16}
    ):
        raise ValueError("INN-67 donor record denominators mismatch")

    binding_by_target: dict[str, Mapping[str, object]] = {}
    binding_configuration_keys: set[tuple[object, object, object]] = set()
    used_peer_ids: list[str] = []
    for row in bindings:
        if not isinstance(row, Mapping):
            raise ValueError("INN-67 history binding is not an object")
        target = row.get("target_evidence_id")
        peer_ids = row.get("earlier_peer_evidence_ids")
        peer_batches = row.get("earlier_peer_batch_ids")
        if (
            not isinstance(target, str)
            or not target
            or target in binding_by_target
            or row.get("target_batch_id") != "INN67-DEV-RB-B03"
            or not isinstance(row.get("recipe_id"), str)
            or not row.get("recipe_id")
            or not isinstance(row.get("stage_id"), str)
            or not row.get("stage_id")
            or type(row.get("stage_ordinal")) is not int
            or not isinstance(peer_ids, list)
            or len(peer_ids) != 2
            or any(not isinstance(peer, str) or not peer for peer in peer_ids)
            or len(set(peer_ids)) != 2
            or any(peer not in record_by_id for peer in peer_ids)
            or peer_batches != donor_batches
            or [record_by_id[peer].get("batch_id") for peer in peer_ids] != donor_batches
            or any(
                record_by_id[peer].get("recipe_id") != row.get("recipe_id")
                or record_by_id[peer].get("stage_id") != row.get("stage_id")
                or record_by_id[peer].get("stage_ordinal") != row.get("stage_ordinal")
                for peer in peer_ids
            )
        ):
            raise ValueError("INN-67 history bindings must bind 16 B03 targets to two earlier peers")
        binding_by_target[target] = row
        binding_configuration_keys.add(
            (row.get("recipe_id"), row.get("stage_id"), row.get("stage_ordinal"))
        )
        used_peer_ids.extend(peer_ids)
    if Counter(used_peer_ids) != Counter({evidence_id: 1 for evidence_id in record_by_id}):
        raise ValueError("INN-67 history bindings must use each donor record exactly once")
    if len(binding_configuration_keys) != 16:
        raise ValueError("INN-67 history bindings must contain 16 unique recipe-stage configurations")

    config_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "distance_metric",
        "unit",
        "config_id",
        "tight_peer_pair_max_delta_e76",
        "target_outlier_margin_delta_e76",
        "decision_rule",
        "threshold_formula",
        "configuration_target_gate",
        "selection_input_batch_ids",
        "reference_geometry_batch_ids",
        "recipe_stage_configurations",
        "recipe_stage_configuration_count",
        "wp1_test_accessed",
        "video5_accessed",
        "claim_boundary",
        "provenance",
    }
    configurations = config.get("recipe_stage_configurations")
    if (
        set(config) != config_fields
        or config.get("schema_version") != 1
        or config.get("artifact_type") != "inn67_delta_e_frozen_development_config"
        or config.get("status") != "frozen_before_test_or_video5_access"
        or config.get("distance_metric") != "CIE76"
        or config.get("unit") != "delta_e76"
        or not isinstance(config.get("config_id"), str)
        or config.get("config_id") != "v3-relative-diagnostic-inherited-stage-v1"
        or config.get("tight_peer_pair_max_delta_e76") != 2.0
        or config.get("target_outlier_margin_delta_e76") != 1.0
        or config.get("decision_rule") != _INN67_DECISION_RULE
        or config.get("threshold_formula") != _INN67_THRESHOLD_FORMULA
        or config.get("configuration_target_gate") != _INN67_CONFIGURATION_GATE
        or config.get("selection_input_batch_ids") != ["INN67-DEV-RB-B03"]
        or config.get("reference_geometry_batch_ids") != donor_batches
        or config.get("recipe_stage_configuration_count") != 16
        or not isinstance(configurations, list)
        or len(configurations) != 16
        or config.get("wp1_test_accessed") is not False
        or config.get("video5_accessed") is not False
        or config.get("provenance") != provenance
    ):
        raise ValueError("INN-67 frozen development config contract mismatch")
    configuration_targets: set[str] = set()
    configuration_keys: set[tuple[object, object, object]] = set()
    for row in configurations:
        if not isinstance(row, Mapping):
            raise ValueError("INN-67 recipe-stage configuration is not an object")
        target = row.get("configuration_target_evidence_id")
        binding = binding_by_target.get(str(target))
        numeric_fields = (
            "peer_pair_delta_e76",
            "threshold_delta_e76",
            "target_to_reference_center_delta_e76",
        )
        if (
            not isinstance(target, str)
            or target in configuration_targets
            or binding is None
            or row.get("consistent_with_two_peer_mode") is not True
            or row.get("relative_status") != "consistent_with_two_peer_mode"
            or row.get("peer_evidence_ids") != binding.get("earlier_peer_evidence_ids")
            or row.get("recipe_id") != binding.get("recipe_id")
            or row.get("stage_id") != binding.get("stage_id")
            or row.get("stage_ordinal") != binding.get("stage_ordinal")
            or any(
                type(row.get(field)) not in {int, float}
                or not math.isfinite(float(row[field]))
                or float(row[field]) < 0.0
                for field in numeric_fields
            )
            or float(row.get("peer_pair_delta_e76", math.inf)) > 2.0
            or float(row.get("target_to_reference_center_delta_e76", math.inf))
            > float(row.get("threshold_delta_e76", -math.inf))
        ):
            raise ValueError("INN-67 frozen config gate or history binding mismatch")
        expected_threshold = (
            float(row["peer_pair_delta_e76"]) / 2.0
            + float(config["target_outlier_margin_delta_e76"])
        )
        if not math.isclose(
            float(row["threshold_delta_e76"]),
            expected_threshold,
            rel_tol=_INN67_ARITHMETIC_TOLERANCE,
            abs_tol=_INN67_ARITHMETIC_TOLERANCE,
        ):
            raise ValueError("INN-67 frozen threshold relation mismatch")
        configuration_targets.add(target)
        configuration_keys.add(
            (row.get("recipe_id"), row.get("stage_id"), row.get("stage_ordinal"))
        )
    if len(configuration_keys) != 16:
        raise ValueError("INN-67 frozen config must contain 16 unique recipe-stage configurations")

    for _path, expected_sha256, role in artifacts:
        exact_path = resolved[role]
        if _file_hash(exact_path) != expected_sha256:
            raise ValueError(f"{role} changed during validation")

    return {
        "materialization_receipt_path": str(resolved["INN-67 materialization receipt"]),
        "materialization_receipt_sha256": expected_materialization_receipt_sha256,
        "development_reference_bank_path": str(resolved["INN-67 development reference bank"]),
        "development_reference_bank_sha256": expected_development_reference_bank_sha256,
        "delta_e_frozen_config_path": str(resolved["INN-67 frozen config"]),
        "delta_e_frozen_config_sha256": expected_delta_e_frozen_config_sha256,
        "bank_identity": {
            "artifact_type": bank["artifact_type"],
            "status": bank["status"],
            "split_role": bank["split_role"],
            "predictor": dict(predictor),
            "predictor_sha256": _canonical_hash(predictor),
            "reference_donor_batch_ids": list(donor_batches),
            "configuration_target_batch_id": bank["configuration_target_batch_id"],
            "record_count": len(records),
            "history_binding_count": len(bindings),
        },
        "config_identity": {
            "artifact_type": config["artifact_type"],
            "status": config["status"],
            "config_id": config["config_id"],
            "distance_metric": config["distance_metric"],
            "unit": config["unit"],
            "selection_input_batch_ids": list(config["selection_input_batch_ids"]),
            "reference_geometry_batch_ids": list(config["reference_geometry_batch_ids"]),
            "configuration_count": len(configurations),
        },
        "provenance": dict(provenance),
        "no_access_claims": no_access,
    }


def _validate_exact_source(
    *,
    source_receipt: Path,
    expected_receipt_sha256: str,
    source_database: Path,
    expected_database_sha256: str,
    independent_review: Path,
    expected_review_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    expected = (
        (Path(source_receipt), ACCEPTED_RECEIPT, expected_receipt_sha256, ACCEPTED_RECEIPT_SHA256, "receipt"),
        (Path(source_database), ACCEPTED_DATABASE, expected_database_sha256, ACCEPTED_DATABASE_SHA256, "database"),
        (Path(independent_review), ACCEPTED_REVIEW, expected_review_sha256, ACCEPTED_REVIEW_SHA256, "review"),
    )
    for supplied_path, accepted_path, supplied_hash, accepted_hash, role in expected:
        exact_lower_sha256(supplied_hash, f"expected {role} SHA-256")
        if supplied_path.resolve() != accepted_path.resolve():
            if role == "database":
                raise ValueError("source_database is not the exact current source database")
            raise ValueError(f"{role} path is not the exact accepted {role}")
        if supplied_hash != accepted_hash:
            raise ValueError(f"expected {role} SHA-256 is not the accepted {role} SHA-256")
        if _file_hash(supplied_path) != accepted_hash:
            raise ValueError(f"accepted {role} SHA-256 mismatch")

    receipt = _strict_json(Path(source_receipt), "accepted source receipt")
    database = receipt.get("database")
    if not isinstance(database, Mapping) or (
        database.get("path") != "evidence.sqlite3"
        or database.get("sha256") != ACCEPTED_DATABASE_SHA256
        or database.get("sqlite_user_version") != 3
    ):
        raise ValueError("accepted receipt does not bind the exact current source database")
    denominators = receipt.get("denominators")
    expected_denominators = {
        "production_runs": 6,
        "production_vials": 18,
        "production_stage_records": 45,
        "train_stage_records": 39,
        "validation_stage_records": 6,
        "predicted_mask_images": 45,
        "rgb_measurements": 45,
        "fill_clips": 18,
    }
    if not isinstance(denominators, Mapping) or any(
        denominators.get(key) != count for key, count in expected_denominators.items()
    ):
        raise ValueError("accepted receipt denominators drifted")
    locator_binding = receipt.get("combined_prediction_locators")
    if not isinstance(locator_binding, Mapping):
        raise ValueError("accepted receipt lacks prediction-locator binding")
    locator_path = Path(str(locator_binding.get("path")))
    locator_sha = locator_binding.get("sha256")
    if not isinstance(locator_sha, str) or _file_hash(locator_path) != locator_sha:
        raise ValueError("accepted prediction-locator SHA-256 mismatch")
    locator_receipt = _strict_json(locator_path, "accepted prediction locators")
    if (
        locator_receipt.get("data_origin") != "twin"
        or locator_receipt.get("mask_source") != "rgb_only_prediction"
        or not isinstance(locator_receipt.get("images"), dict)
        or len(locator_receipt["images"]) != 45
    ):
        raise ValueError("accepted prediction locators are not exact twin RGB-only predictions")
    return receipt, dict(locator_receipt["images"])


def _source_counts(database: Path) -> dict[str, int]:
    with sqlite3.connect(database) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("source SQLite integrity check failed")
        values = {
            "run_count": connection.execute("SELECT count(*) FROM runs").fetchone()[0],
            "vial_count": connection.execute("SELECT count(*) FROM vials").fetchone()[0],
            "stage_record_count": connection.execute("SELECT count(*) FROM stages").fetchone()[0],
            "train_stage_record_count": connection.execute(
                "SELECT count(*) FROM stages WHERE split_role = 'train'"
            ).fetchone()[0],
            "validation_stage_record_count": connection.execute(
                "SELECT count(*) FROM stages WHERE split_role = 'validation'"
            ).fetchone()[0],
            "decision_row_count": connection.execute("SELECT count(*) FROM decisions").fetchone()[0],
            "explanation_row_count": connection.execute("SELECT count(*) FROM explanations").fetchone()[0],
            "sim_metadata_row_count": connection.execute("SELECT count(*) FROM sim_metadata").fetchone()[0],
        }
        forbidden_roles = connection.execute(
            "SELECT count(*) FROM stages WHERE split_role IN ('oracle','evaluation','test')"
        ).fetchone()[0]
        non_predicted = connection.execute(
            "SELECT count(*) FROM images WHERE mask_source <> 'predicted'"
        ).fetchone()[0]
    if values != {
        "run_count": 6,
        "vial_count": 18,
        "stage_record_count": 45,
        "train_stage_record_count": 39,
        "validation_stage_record_count": 6,
        "decision_row_count": 0,
        "explanation_row_count": 0,
        "sim_metadata_row_count": 0,
    }:
        raise ValueError("source SQLite denominators drifted")
    if forbidden_roles or non_predicted:
        raise ValueError("source SQLite contains forbidden roles or non-predicted masks")
    return values


def _validate_ordered_video(
    *,
    ordered_video: Path,
    expected_ordered_video_sha256: str,
    composition_receipt: Path,
    expected_composition_receipt_sha256: str,
    ordered_video_review: Path,
    expected_ordered_video_review_sha256: str,
) -> dict[str, object]:
    exact = (
        (ordered_video, ACCEPTED_ORDERED_VIDEO, expected_ordered_video_sha256, ACCEPTED_ORDERED_VIDEO_SHA256, "ordered video"),
        (composition_receipt, ACCEPTED_COMPOSITION_RECEIPT, expected_composition_receipt_sha256, ACCEPTED_COMPOSITION_RECEIPT_SHA256, "composition receipt"),
        (ordered_video_review, ACCEPTED_ORDERED_VIDEO_REVIEW, expected_ordered_video_review_sha256, ACCEPTED_ORDERED_VIDEO_REVIEW_SHA256, "ordered-video review"),
    )
    for supplied, accepted, expected_hash, accepted_hash, role in exact:
        exact_lower_sha256(expected_hash, f"expected {role} SHA-256")
        if Path(supplied).resolve() != accepted.resolve():
            raise ValueError(f"{role} path is not the exact accepted artifact")
        if expected_hash != accepted_hash:
            raise ValueError(f"expected {role} SHA-256 is stale")
        if _file_hash(Path(supplied)) != accepted_hash:
            raise ValueError(f"accepted {role} SHA-256 mismatch")

    composition = _strict_json(Path(composition_receipt), "composition receipt")
    output = composition.get("output")
    runs = composition.get("ordered_runs")
    verification = composition.get("verification")
    if (
        composition.get("artifact_type") != "wp7_ordered_test_video_v1"
        or composition.get("status") != "pass"
        or not isinstance(output, Mapping)
        or output.get("path") != ACCEPTED_ORDERED_VIDEO.name
        or output.get("sha256") != ACCEPTED_ORDERED_VIDEO_SHA256
        or output.get("frame_count") != 7322
        or output.get("duration_seconds") != "732.2"
        or not isinstance(verification, Mapping)
        or verification.get("case_count") != 18
        or not isinstance(runs, list)
        or len(runs) != 2
    ):
        raise ValueError("composition receipt ordered-video contract drifted")
    case_ids = [case for run in runs for case in run.get("case_ids", [])]
    if len(case_ids) != 18 or len(set(case_ids)) != 18:
        raise ValueError("composition receipt does not bind 18 unique cases")

    review = _strict_json(Path(ordered_video_review), "ordered-video final review")
    identity = review.get("identity")
    reviewed_media = review.get("output_media")
    case_contract = review.get("case_contract")
    if (
        review.get("status") != "pass"
        or review.get("verdict") != "pass"
        or not isinstance(identity, Mapping)
        or identity.get("output_mp4_sha256") != ACCEPTED_ORDERED_VIDEO_SHA256
        or identity.get("output_receipt_sha256") != ACCEPTED_COMPOSITION_RECEIPT_SHA256
        or not isinstance(reviewed_media, Mapping)
        or reviewed_media.get("sha256") != ACCEPTED_ORDERED_VIDEO_SHA256
        or reviewed_media.get("frame_count") != 7322
        or reviewed_media.get("duration_seconds") != "732.2"
        or not isinstance(case_contract, Mapping)
        or case_contract.get("count") != 18
        or case_contract.get("unique") is not True
    ):
        raise ValueError("ordered-video final review contract drifted")
    return {
        "video_path": str(ACCEPTED_ORDERED_VIDEO),
        "video_sha256": ACCEPTED_ORDERED_VIDEO_SHA256,
        "composition_receipt_path": str(ACCEPTED_COMPOSITION_RECEIPT),
        "composition_receipt_sha256": ACCEPTED_COMPOSITION_RECEIPT_SHA256,
        "final_review_path": str(ACCEPTED_ORDERED_VIDEO_REVIEW),
        "final_review_sha256": ACCEPTED_ORDERED_VIDEO_REVIEW_SHA256,
        "case_count": 18,
        "frame_count": 7322,
        "duration_seconds": "732.2",
    }


def _group_vials(
    records: list[dict[str, object]], cursors: Mapping[str, int]
) -> list[list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(
            (str(record["run_id"]), int(record["vial_ordinal"])), []
        ).append(record)
    return [
        sorted(rows, key=lambda record: cursors[str(record["evidence_id"])])
        for _, rows in sorted(grouped.items())
    ]


def _derive_vial(
    measured: list[dict[str, object]],
    query: object,
    assert_history_citation_fn: Callable[[object, str, str], object],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    first = measured[0]
    limits = list(_BASE_LIMITS)
    if not query.members:
        limits.append("prior_reference_unavailable")
    limits.sort()
    history_refs = [
        {"evidence_id": member.evidence_id, "query_sha256": query.query_sha256}
        for member in sorted(query.members, key=lambda item: item.evidence_id)
    ]
    for member in query.members:
        assert_history_citation_fn(query, member.evidence_id, member.image_id)
    identity = {
        "run_id": first["run_id"],
        "vial_ordinal": first["vial_ordinal"],
        "query_sha256": query.query_sha256,
        "limit_codes": limits,
    }
    token = _canonical_hash(identity)
    decision_id = f"DEC-WP7-{token[:24]}"
    explanation_id = f"EXP-WP7-{token[24:48]}"
    prompt = {
        "schema_version": "wp7-development-history-template-v1",
        "status": "UNKNOWN",
        "action": "abstain_and_collect_evidence",
        "limit_codes": limits,
    }
    guard_result = (
        "UNKNOWN; abstain; collect evidence; limit_codes=" + ",".join(limits)
    )
    explained: list[dict[str, object]] = []
    for predecessor in measured:
        record = deepcopy(predecessor)
        record.update(
            {
                "decision_id": decision_id,
                "disposition": "insufficient_evidence",
                "limit_codes": limits,
                "history_refs": history_refs,
                "explanation_id": explanation_id,
                "model_revision": _MODEL_REVISION,
                "prompt_sha256": _canonical_hash(prompt),
                "image_ids_sent": [],
                "guard_result": "fallback",
            }
        )
        explained.append(record)
    query_receipt = {
        "run_id": first["run_id"],
        "vial_ordinal": first["vial_ordinal"],
        "target_evidence_id": first["evidence_id"],
        "query_sha256": query.query_sha256,
        "query_kind": "vial_history",
        "filters": query.filters,
        "members": [
            {
                "evidence_id": member.evidence_id,
                "image_id": member.image_id,
                "event_cursor": member.event_cursor,
                "split_role": member.split_role,
            }
            for member in query.members
        ],
    }
    explanation = {
        "decision_id": decision_id,
        "explanation_id": explanation_id,
        "run_id": first["run_id"],
        "vial_ordinal": first["vial_ordinal"],
        "disposition": "insufficient_evidence",
        "status": "UNKNOWN",
        "action": "abstain_and_collect_evidence",
        "limit_codes": limits,
        "history_refs": history_refs,
        "model_revision": _MODEL_REVISION,
        "prompt_sha256": _canonical_hash(prompt),
        "image_ids_sent": [],
        "guard_result": guard_result,
    }
    return explained, query_receipt, explanation


def _build_development_history_population(
    *,
    source_receipt: Path,
    expected_receipt_sha256: str,
    source_database: Path,
    expected_database_sha256: str,
    independent_review: Path,
    expected_review_sha256: str,
    owner_id: str,
    destination: Path,
    evidence_store_factory: Callable[..., object],
    assert_history_citation_fn: Callable[[object, str, str], object],
) -> dict[str, object]:
    """Build one fresh derived synthetic population without opening one-shot inputs."""

    if not callable(evidence_store_factory) or not callable(
        assert_history_citation_fn
    ):
        raise TypeError("history storage capabilities must be callable")
    bounded_control_free_text(owner_id, "owner_id")
    _, accepted_locators = _validate_exact_source(
        source_receipt=source_receipt,
        expected_receipt_sha256=expected_receipt_sha256,
        source_database=source_database,
        expected_database_sha256=expected_database_sha256,
        independent_review=independent_review,
        expected_review_sha256=expected_review_sha256,
    )
    _source_counts(Path(source_database))
    source_hashes_before = {
        Path(source_receipt): _file_hash(Path(source_receipt)),
        Path(source_database): _file_hash(Path(source_database)),
        Path(independent_review): _file_hash(Path(independent_review)),
    }
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    copied_database = root / "evidence.sqlite3"
    shutil.copyfile(source_database, copied_database)
    if _file_hash(copied_database) != ACCEPTED_DATABASE_SHA256:
        raise ValueError("copied source database SHA-256 mismatch")
    if any(_file_hash(path) != digest for path, digest in source_hashes_before.items()):
        raise ValueError("accepted source tuple changed during copy")

    store = evidence_store_factory(
        copied_database, artifact_root=ACCEPTED_SOURCE_ROOT
    )
    measured, locators, cursors = store.measured_population()
    if locators != accepted_locators:
        raise ValueError("stored normalized locators do not match accepted locator receipt")
    vials = _group_vials(measured, cursors)
    if len(vials) != 18 or len(measured) != 45:
        raise ValueError("reconstructed source population denominators drifted")

    explained_records: list[dict[str, object]] = []
    query_receipts: list[dict[str, object]] = []
    explanations: list[dict[str, object]] = []
    queries: list[object] = []
    for vial in vials:
        first = vial[0]
        query = store.vial_history(
            str(first["run_id"]),
            before=cursors[str(first["evidence_id"])],
            allowed_roles={"train"},
            target_evidence_id=str(first["evidence_id"]),
        )
        queries.append(query)
        explained, query_receipt, explanation = _derive_vial(
            vial, query, assert_history_citation_fn
        )
        explained_records.extend(explained)
        query_receipts.append(query_receipt)
        explanations.append(explanation)
    result = store.append_explained(
        explained_records, locators, history_queries=queries
    )
    if result.inserted != 45 or result.existing:
        raise ValueError("derived decision/explanation append count drifted")

    with sqlite3.connect(copied_database) as connection:
        post_counts = {
            "decision_row_count": connection.execute("SELECT count(*) FROM decisions").fetchone()[0],
            "explanation_row_count": connection.execute("SELECT count(*) FROM explanations").fetchone()[0],
            "store_history_query_row_count": connection.execute("SELECT count(*) FROM history_queries").fetchone()[0],
            "sim_metadata_row_count": connection.execute("SELECT count(*) FROM sim_metadata").fetchone()[0],
            "forbidden_role_count": connection.execute(
                "SELECT count(*) FROM stages WHERE split_role IN ('oracle','evaluation','test')"
            ).fetchone()[0],
            "non_predicted_mask_count": connection.execute(
                "SELECT count(*) FROM images WHERE mask_source <> 'predicted'"
            ).fetchone()[0],
        }
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if (
        integrity != "ok"
        or post_counts["decision_row_count"] != 45
        or post_counts["explanation_row_count"] != 45
        or post_counts["sim_metadata_row_count"]
        or post_counts["forbidden_role_count"]
        or post_counts["non_predicted_mask_count"]
    ):
        raise ValueError("derived store verification failed")

    population_path = root / "population-records.json"
    query_path = root / "query-receipts.json"
    explanation_path = root / "explanation-records.json"
    _publish_json(population_path, explained_records)
    _publish_json(query_path, query_receipts)
    _publish_json(explanation_path, explanations)
    primary_paths = (copied_database, population_path, query_path, explanation_path)
    primary_inventory = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _file_hash(path)}
        for path in primary_paths
    ]
    limit_counts = Counter(
        code for explanation in explanations for code in explanation["limit_codes"]
    )
    receipt: dict[str, object] = {
        "schema_version": "wp7-development-history-population-v1",
        "artifact_type": "synthetic_development_history_preparation",
        "owner_id": owner_id,
        "source_provenance": "synthetic_twin_development_only",
        "route_approved": False,
        "one_shot_inputs_opened": False,
        "source_tuple": {
            "receipt_path": str(ACCEPTED_RECEIPT),
            "receipt_sha256": ACCEPTED_RECEIPT_SHA256,
            "database_path": str(ACCEPTED_DATABASE),
            "database_sha256": ACCEPTED_DATABASE_SHA256,
            "independent_review_path": str(ACCEPTED_REVIEW),
            "independent_review_sha256": ACCEPTED_REVIEW_SHA256,
        },
        "denominators": {
            "run_count": 6,
            "vial_count": 18,
            "stage_record_count": 45,
            "train_stage_record_count": 39,
            "validation_stage_record_count": 6,
            "decision_row_count": 45,
            "explanation_row_count": 45,
            "unique_vial_decision_count": 18,
            "insufficient_evidence_vial_count": 18,
        },
        "limit_code_vial_counts": dict(sorted(limit_counts.items())),
        "population_query_sha256s": [query.query_sha256 for query in queries],
        "population_query_count": 18,
        "store_history_query_row_count": post_counts["store_history_query_row_count"],
        "output_inventory": primary_inventory,
    }
    receipt["receipt_sha256"] = _canonical_hash(receipt)
    receipt_path = root / "population-receipt.json"
    _publish_json(receipt_path, receipt)
    inventory = {
        "schema_version": "wp7-development-history-sha-inventory-v1",
        "artifacts": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _file_hash(path)}
            for path in (*primary_paths, receipt_path)
        ],
    }
    inventory["inventory_sha256"] = _canonical_hash(inventory)
    _publish_json(root / "sha-inventory.json", inventory)
    if any(_file_hash(path) != digest for path, digest in source_hashes_before.items()):
        raise ValueError("accepted source tuple changed during population build")
    return receipt


def publish_preparation_preflight(
    *,
    source_receipt: Path,
    expected_receipt_sha256: str,
    source_database: Path,
    expected_database_sha256: str,
    independent_review: Path,
    expected_review_sha256: str,
    ordered_video: Path,
    expected_ordered_video_sha256: str,
    composition_receipt: Path,
    expected_composition_receipt_sha256: str,
    ordered_video_review: Path,
    expected_ordered_video_review_sha256: str,
    population_receipt: Path,
    expected_population_receipt_sha256: str,
    materialization_receipt: Path,
    expected_materialization_receipt_sha256: str,
    development_reference_bank: Path,
    expected_development_reference_bank_sha256: str,
    delta_e_frozen_config: Path,
    expected_delta_e_frozen_config_sha256: str,
    owner_id: str,
    destination: Path,
) -> dict[str, object]:
    """Publish a sealed preparation receipt that execution APIs must reject."""

    bounded_control_free_text(owner_id, "owner_id")
    _validate_exact_source(
        source_receipt=source_receipt,
        expected_receipt_sha256=expected_receipt_sha256,
        source_database=source_database,
        expected_database_sha256=expected_database_sha256,
        independent_review=independent_review,
        expected_review_sha256=expected_review_sha256,
    )
    ordered_video_binding = _validate_ordered_video(
        ordered_video=ordered_video,
        expected_ordered_video_sha256=expected_ordered_video_sha256,
        composition_receipt=composition_receipt,
        expected_composition_receipt_sha256=expected_composition_receipt_sha256,
        ordered_video_review=ordered_video_review,
        expected_ordered_video_review_sha256=expected_ordered_video_review_sha256,
    )
    inn67_binding = _validate_inn67_binding(
        materialization_receipt=materialization_receipt,
        expected_materialization_receipt_sha256=expected_materialization_receipt_sha256,
        development_reference_bank=development_reference_bank,
        expected_development_reference_bank_sha256=expected_development_reference_bank_sha256,
        delta_e_frozen_config=delta_e_frozen_config,
        expected_delta_e_frozen_config_sha256=expected_delta_e_frozen_config_sha256,
    )
    exact_lower_sha256(
        expected_population_receipt_sha256,
        "expected population receipt SHA-256",
    )
    population_path = Path(population_receipt)
    if _file_hash(population_path) != expected_population_receipt_sha256:
        raise ValueError("derived population receipt SHA-256 mismatch")
    population = _strict_json(population_path, "derived population receipt")
    embedded = population.get("receipt_sha256")
    canonical = _canonical_hash(
        {key: value for key, value in population.items() if key != "receipt_sha256"}
    )
    if embedded != canonical:
        raise ValueError("derived population canonical receipt hash mismatch")
    exact_population_fields = {
        "schema_version",
        "artifact_type",
        "owner_id",
        "source_provenance",
        "route_approved",
        "one_shot_inputs_opened",
        "source_tuple",
        "denominators",
        "limit_code_vial_counts",
        "population_query_sha256s",
        "population_query_count",
        "store_history_query_row_count",
        "output_inventory",
        "receipt_sha256",
    }
    expected_denominators = {
        "run_count": 6,
        "vial_count": 18,
        "stage_record_count": 45,
        "train_stage_record_count": 39,
        "validation_stage_record_count": 6,
        "decision_row_count": 45,
        "explanation_row_count": 45,
        "unique_vial_decision_count": 18,
        "insufficient_evidence_vial_count": 18,
    }
    query_hashes = population.get("population_query_sha256s")
    inventory = population.get("output_inventory")
    if (
        set(population) != exact_population_fields
        or population.get("schema_version") != "wp7-development-history-population-v1"
        or population.get("artifact_type") != "synthetic_development_history_preparation"
        or population.get("owner_id") != owner_id
        or population.get("source_provenance") != "synthetic_twin_development_only"
        or population.get("route_approved") is not False
        or population.get("one_shot_inputs_opened") is not False
        or population.get("source_tuple")
        != {
            "receipt_path": str(ACCEPTED_RECEIPT),
            "receipt_sha256": ACCEPTED_RECEIPT_SHA256,
            "database_path": str(ACCEPTED_DATABASE),
            "database_sha256": ACCEPTED_DATABASE_SHA256,
            "independent_review_path": str(ACCEPTED_REVIEW),
            "independent_review_sha256": ACCEPTED_REVIEW_SHA256,
        }
        or population.get("denominators") != expected_denominators
        or population.get("limit_code_vial_counts")
        != {
            "frozen_decision_config_unavailable": 18,
            "matched_peer_batches_unavailable": 18,
            "prior_reference_unavailable": 8,
        }
        or not isinstance(query_hashes, list)
        or len(query_hashes) != 18
        or len(set(query_hashes)) != 18
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or set(value) - set("0123456789abcdef")
            for value in query_hashes
        )
        or population.get("population_query_count") != 18
        or type(population.get("store_history_query_row_count")) is not int
        or population["store_history_query_row_count"] < 18
        or not isinstance(inventory, list)
        or len(inventory) != 4
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "sha256"}
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or set(row["sha256"]) - set("0123456789abcdef")
            for row in inventory
        )
        or {row.get("path") for row in inventory if isinstance(row, Mapping)}
        != {
            "evidence.sqlite3",
            "population-records.json",
            "query-receipts.json",
            "explanation-records.json",
        }
    ):
        raise ValueError("derived population receipt is not the exact sealed population")
    preflight: dict[str, object] = {
        "schema_version": "wp7-development-history-preparation-preflight-v2",
        "artifact_type": "sealed_synthetic_development_history_preflight",
        "owner_id": owner_id,
        "source_provenance": "synthetic_twin_development_only",
        "source_tuple": population["source_tuple"],
        "ordered_video": ordered_video_binding,
        "population_receipt_sha256": expected_population_receipt_sha256,
        "population_canonical_sha256": embedded,
        "inn67_binding": inn67_binding,
        "denominators": {
            "vials": 18,
            "insufficient_evidence": {"count": 18, "total": 18},
            "limit_code_vial_counts": population["limit_code_vial_counts"],
        },
        "one_shot_inputs_opened": False,
        "access_status": "sealed_pending_liam_decision",
        "route_decision": None,
        "history_backed_explanations_enabled": False,
        "route_status": "option3_inn67_inputs_bound_pending_liam_access_decision",
        "fallback": {
            "trigger": "accepted INN-67 reference bank and config unavailable after approximately 18:00 Saturday America/Chicago",
            "route": "option1_all_insufficient_evidence",
            "option2_approved": False,
        },
    }
    preflight["preflight_sha256"] = _canonical_hash(preflight)
    _publish_json(Path(destination), preflight)
    return preflight


__all__ = [
    "ACCEPTED_DATABASE",
    "ACCEPTED_DATABASE_SHA256",
    "ACCEPTED_RECEIPT",
    "ACCEPTED_RECEIPT_SHA256",
    "ACCEPTED_REVIEW",
    "ACCEPTED_REVIEW_SHA256",
    "ACCEPTED_ORDERED_VIDEO",
    "ACCEPTED_ORDERED_VIDEO_SHA256",
    "ACCEPTED_COMPOSITION_RECEIPT",
    "ACCEPTED_COMPOSITION_RECEIPT_SHA256",
    "ACCEPTED_ORDERED_VIDEO_REVIEW",
    "ACCEPTED_ORDERED_VIDEO_REVIEW_SHA256",
    "publish_preparation_preflight",
]
