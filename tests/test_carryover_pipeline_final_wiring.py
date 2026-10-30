"""Public-boundary tests for the WP7 final-decision production wiring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def _runtime_output(tmp_path: Path, recipe: dict[str, object]):
    import fluidmix.vision_measurement as measurement
    from fluidmix.carryover_pipeline import RuntimeProjectionInput
    from fluidmix.rgb_mask_prediction import build_trusted_predicted_mask_registry_v1

    images = {}
    templates = []
    locators = {}
    cursors = {}
    bindings = []
    video = tmp_path / "video.bin"
    cursor = 0
    for run in recipe["runs"]:
        for vial in run["vials"]:
            for ordinal, tank in enumerate(vial["dispense_order"], start=1):
                image_id = f"IMAGE-{vial['case_id']}-{ordinal}"
                rgb = tmp_path / "rgb" / f"{image_id}.png"
                mask = tmp_path / "mask" / f"{image_id}.png"
                rgb.parent.mkdir(exist_ok=True)
                mask.parent.mkdir(exist_ok=True)
                colour = (80, 100, 120) if vial["vial_ordinal"] != 2 else (220, 20, 20)
                Image.new("RGB", (4, 3), colour).save(rgb)
                pixels = np.zeros((3, 4), dtype=np.uint8)
                if vial["vial_ordinal"] != 4:
                    pixels[1:, 1:3] = 255
                Image.fromarray(pixels, mode="L").save(mask)
                images[image_id] = {
                    "rgb_path": str(rgb), "rgb_sha256": _sha(rgb),
                    "mask_path": str(mask), "mask_sha256": _sha(mask),
                }
    artifact = {
        "schema_version": 2, "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin", "mask_source": "rgb_only_prediction",
        "producer": {
            "producer_id": "segmenter-1", "producer_kind": "rgb_only_liquid_segmenter",
            "model_id": "model-1", "model_sha256": "a" * 64,
            "config_id": "config-1", "config_sha256": "b" * 64,
            "code_commit": "c" * 40, "threshold_source": "twin_validation_frozen",
            "training_data_origin": "twin", "input_fields": ["rgb"],
        },
        "images": images,
    }
    registry = build_trusted_predicted_mask_registry_v1(artifact)
    for run in recipe["runs"]:
        for vial in run["vials"]:
            for ordinal, tank in enumerate(vial["dispense_order"], start=1):
                image_id = f"IMAGE-{vial['case_id']}-{ordinal}"
                evidence_id = f"EVID-{vial['case_id']}-{ordinal}"
                measured, binding = measurement.measure_trusted_predicted_image(
                    registry.entry(image_id),
                    measurement_id=f"MEAS-{vial['case_id']}-{ordinal}",
                    evidence_id=evidence_id,
                )
                bindings.append(binding)
                entry = registry.entry(image_id)
                templates.append({
                    "run_id": run["run_id"], "case_id": vial["case_id"],
                    "vial_ordinal": vial["vial_ordinal"],
                    "block_recipe_id": vial["block_recipe_id"],
                    "block_position": vial["block_position"],
                    "recipe_id": vial["recipe_id"],
                    "previous_recipe_id": vial["previous_recipe_id"],
                    "recipe_ml": dict(vial["programmed_tank_ml"]),
                    "dispense_order": list(vial["dispense_order"]),
                    "split_role": recipe["split_role"],
                    "split_version": recipe["split_version"],
                    "stage_id": f"STAGE-{vial['case_id']}-{ordinal}",
                    "tank_id": tank, "stage_ordinal": ordinal,
                    "fill_clip_path": video.name, "fill_frame_range": [0, 1],
                    "transition_frame_ranges": {"entry": None, "exit": None},
                    "clip_sha256": _sha(video), "image_id": image_id,
                    "rgb_sha256": entry.rgb_sha256, "mask_sha256": entry.mask_sha256,
                    "mask_source": "predicted", "measurement_id": binding.measurement_id,
                    "median_lab": [0.0, 0.0, 0.0], "mad_lab": [0.0, 0.0, 0.0],
                    "iqr_lab": [0.0, 0.0, 0.0], "pairwise_delta_e76": [],
                    "colour_regime_id": None, "evidence_id": evidence_id,
                    "decision_id": None, "disposition": None, "limit_codes": [],
                    "history_refs": [], "explanation_id": None, "model_revision": None,
                    "prompt_sha256": None, "image_ids_sent": [], "guard_result": None,
                })
                locators[image_id] = entry.locator()
                cursors[evidence_id] = cursor
                cursor += 1
    trusted = measurement.TrustedMeasuredEvidenceResult(
        {"measurements": []}, tuple(bindings), registry,
        _factory_token=measurement._TRUSTED_MEASUREMENT_TOKEN,
    )
    return RuntimeProjectionInput(tuple(templates), trusted, locators, cursors, {"test": True}), bindings


def _development_store(tmp_path: Path, bindings):
    from fluidmix.wp7_development_reference import DevelopmentReferenceStore, _create_minimal_store

    records = []
    configurations = []
    cursor = 0
    usable = [binding for binding in bindings if binding.validity == "valid"]
    first_by_ordinal = {int(binding.evidence_id.rsplit("-", 1)[1]): binding for binding in usable[:2]}
    for batch, run, sign in (("INN67-DEV-RB-B01", "DONOR-A", -1),
                             ("INN67-DEV-RB-B02", "DONOR-B", 1)):
        for ordinal, tank in enumerate(("t100", "t200"), start=1):
            binding = first_by_ordinal[ordinal]
            evidence_id = f"DONOR-{batch}-{ordinal}"
            image_id = f"DONOR-IMAGE-{batch}-{ordinal}"
            rgb = tmp_path / "donors" / f"{image_id}.rgb"
            mask = tmp_path / "donors" / f"{image_id}.mask"
            rgb.parent.mkdir(exist_ok=True)
            rgb.write_bytes(evidence_id.encode())
            mask.write_bytes(image_id.encode())
            lab = list(binding.median_lab)
            lab[1] += sign * 0.5
            records.append({
                "evidence_id": evidence_id, "event_cursor": cursor, "batch_id": batch,
                "run_id": run, "case_id": f"CASE-{batch}", "recipe_id": "R-CURRENT",
                "stage_id": f"DEV-STAGE-{ordinal}", "stage_ordinal": ordinal,
                "tank_id": tank, "split_role": "development_reference",
                "image_id": image_id, "rgb_path": str(rgb), "rgb_sha256": _sha(rgb),
                "mask_path": str(mask), "mask_sha256": _sha(mask), "median_lab": lab,
                "source_render_receipt_sha256": "1" * 64,
                "source_validation_receipt_sha256": "2" * 64,
                "source_render_plan_sha256": "3" * 64,
                "known_good_profile_sha256": "4" * 64,
            })
            cursor += 1
    for ordinal, tank in enumerate(("t100", "t200"), start=1):
        configurations.append({
            "recipe_id": "R-CURRENT", "stage_id": f"DEV-STAGE-{ordinal}",
            "stage_ordinal": ordinal, "tank_id": tank,
            "peer_evidence_ids": [f"DONOR-INN67-DEV-RB-B01-{ordinal}",
                                  f"DONOR-INN67-DEV-RB-B02-{ordinal}"],
            "threshold_delta_e76": 1.5,
            "configuration_target_evidence_id": f"CONFIG-{ordinal}",
        })
    database = tmp_path / "development.sqlite3"
    _create_minimal_store(database, records=records, configurations=configurations,
        provenance={
            "development_reference_bank_sha256": "a" * 64,
            "delta_e_frozen_config_sha256": "b" * 64,
            "materialization_receipt_sha256": "c" * 64,
            "predicted_store_export_sha256": "d" * 64,
            "derived_source_manifest_sha256": "e" * 64,
            "predictor": {"producer_id": "p", "model_id": "m", "model_sha256": "8" * 64,
                          "config_sha256": "9" * 64, "code_commit": "a" * 40},
            "predictor_sha256": "f" * 64,
            "accepted_store_tuple": {"database_path": "/never/read.sqlite3",
                "database_sha256": "6" * 64, "population_receipt_path": "/never/read.json",
                "population_receipt_sha256": "7" * 64},
            "inn67_provenance": {"fixed": "test"},
        })
    digest = _sha(database)
    return database, digest, DevelopmentReferenceStore(database, expected_snapshot_sha256=digest)


def _fixture(tmp_path: Path, monkeypatch):
    import fluidmix.carryover_pipeline as pipeline
    from test_carryover_pipeline import _freeze, _recipe, _write_json

    recipe_value = _recipe(vial_count=4)
    recipe_value["runs"][0]["vials"][2]["recipe_id"] = "R-UNSUPPORTED"
    video = tmp_path / "video.bin"
    video.write_bytes(b"synthetic camera fixture")
    recipe_path = tmp_path / "recipe.json"
    _write_json(recipe_path, recipe_value)
    freeze, freeze_sha = _freeze(tmp_path, accepts_real_rgb_only=True)
    runtime_output, bindings = _runtime_output(tmp_path, recipe_value)
    database, snapshot, store = _development_store(tmp_path, bindings)
    monkeypatch.setattr(pipeline, "_validate_execution_preflight", lambda *_args, **_kwargs: {
        "preflight_sha256": "e" * 64,
        "test": {"vial_count": 4}, "real_check": {"vial_count": 30},
        "ordered_video": {"ordered_video_path": str(video.resolve()),
                          "ordered_video_sha256": _sha(video)},
        "history": {"store_snapshot_sha256": snapshot},
    })

    class Runtime:
        runtime_id = "reviewed-clamp-runtime-v1"
        runtime_sha256 = "2" * 64
        config_sha256 = "3" * 64
        accepts_real_rgb_only = True

        def analyze(self, **_kwargs):
            return runtime_output

    return {
        "video": video, "recipe_sequence": recipe_path, "freeze": freeze,
        "expected_freeze_sha256": freeze_sha,
        "evidence_out_store": tmp_path / "explained.sqlite3",
        "out": tmp_path / "receipt.json", "claim": tmp_path / "claim.json",
        "owner_id": "owner", "preflight_receipt": {},
        "expected_preflight_sha256": "e" * 64,
        "selected_mode": "development_reference_backed",
        "development_reference_store": database,
        "development_reference_store_sha256": snapshot,
    }, Runtime(), runtime_output, store


def _runtime_with_missing_and_integrity_invalid_binding(runtime_output):
    import fluidmix.vision_measurement as measurement
    from fluidmix.carryover_pipeline import RuntimeProjectionInput

    registry = measurement.trusted_prediction_registry_from_result_v1(
        runtime_output.trusted_adapter_result
    )
    bindings = list(measurement.trusted_measurement_bindings_from_result_v1(
        runtime_output.trusted_adapter_result
    ))
    missing_image = "IMAGE-CASE-1-1"
    invalid_image = "IMAGE-CASE-1-2"
    invalid_entry = registry.entry(invalid_image)
    Path(invalid_entry.rgb_path).write_bytes(b"integrity-invalid RGB")
    _measured, invalid_binding = measurement.measure_trusted_predicted_image(
        invalid_entry,
        measurement_id="MEAS-CASE-1-2",
        evidence_id="EVID-CASE-1-2",
    )
    assert invalid_binding.limit_codes == ("rgb_hash_mismatch",)
    retained = [
        binding for binding in bindings
        if binding.image_id not in {missing_image, invalid_image}
    ]
    retained.append(invalid_binding)
    trusted = measurement.TrustedMeasuredEvidenceResult(
        {"measurements": []}, tuple(retained), registry,
        _factory_token=measurement._TRUSTED_MEASUREMENT_TOKEN,
    )
    return RuntimeProjectionInput(
        runtime_output.record_templates,
        trusted,
        runtime_output.image_locators,
        runtime_output.event_cursors,
        runtime_output.runtime_provenance,
    )


def test_public_run_wires_keep_discard_unsupported_and_measurement_insufficient(tmp_path, monkeypatch):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    database = kwargs["development_reference_store"]
    snapshot = kwargs["development_reference_store_sha256"]
    evidence_out = kwargs["evidence_out_store"]
    claim = kwargs["claim"]
    upstream = {
        path: _sha(path)
        for path in (kwargs["video"], kwargs["recipe_sequence"], database)
    }
    projection_calls = []
    original_projection = pipeline.project_runtime_measurements

    def tracked_projection(value):
        projection_calls.append(value)
        return original_projection(value)

    monkeypatch.setattr(pipeline, "project_runtime_measurements", tracked_projection)

    def load_reference():
        assert claim.exists()
        assert projection_calls
        assert not evidence_out.exists()
        return store

    receipt = pipeline.run_test(
        **kwargs, mask_runtime=runtime,
        development_reference_store_loader=load_reference,
    )

    assert [row["disposition"] for row in receipt["operator_records"]] == [
        "keep", "discard", "insufficient_evidence", "insufficient_evidence"
    ]
    assert receipt["denominators"]["decision_count"] == 2
    assert receipt["denominators"]["abstention_count"] == 2
    assert receipt["denominators"]["expected_stage_count"] == 8
    assert receipt["provenance_inventory"]["measured_stage_count"] == 6
    assert receipt["provenance_inventory"]["explained_stage_count"] == 6
    assert receipt["source_bindings"]["evidence_out_store_sha256"] == _sha(evidence_out)
    assert receipt["source_bindings"]["development_reference_store_snapshot_sha256"] == snapshot
    assert not {
        "sim_metadata", "truth", "label", "model_answer", "residue_ml",
        "tank_colour_preset",
    }.intersection(_keys(receipt))
    with sqlite3.connect(evidence_out) as connection:
        assert connection.execute("SELECT COUNT(*) FROM stages WHERE phase='explained'").fetchone() == (6,)
        assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone() == (6,)
        assert connection.execute("SELECT COUNT(*) FROM explanations").fetchone() == (6,)
        assert connection.execute("SELECT COUNT(*) FROM decisions WHERE history_refs_json != '[]'").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM sim_metadata").fetchone() == (0,)
    donor_record = receipt["operator_records"][0]["source_bindings"]["development_references"][0]
    assert donor_record["donor_store_snapshot_sha256"] == snapshot
    assert len(donor_record["donors"]) == 2
    assert {path: _sha(path) for path in upstream} == upstream


@pytest.mark.parametrize("runtime_case", ["missing", "freeze_mismatch"])
def test_development_reference_runtime_identity_failure_is_fatal(
    tmp_path, monkeypatch, runtime_case
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, _runtime_output_value, store = _fixture(tmp_path, monkeypatch)
    reference_calls = []
    if runtime_case == "missing":
        selected_runtime = None
    else:
        class FreezeMismatchedRuntime:
            runtime_id = runtime.runtime_id
            runtime_sha256 = "0" * 64
            config_sha256 = runtime.config_sha256
            accepts_real_rgb_only = True

            def analyze(self, **values):
                return runtime.analyze(**values)

        selected_runtime = FreezeMismatchedRuntime()

    with pytest.raises(ValueError, match="development reference runtime"):
        pipeline.run_test(
            **kwargs,
            mask_runtime=selected_runtime,
            development_reference_store_loader=(
                lambda: reference_calls.append("reference") or store
            ),
        )

    assert reference_calls == []
    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"
    assert not kwargs["evidence_out_store"].exists()


def test_missing_binding_cannot_suppress_present_integrity_failure(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    mixed_output = _runtime_with_missing_and_integrity_invalid_binding(runtime_output)

    class MixedRuntime:
        runtime_id = runtime.runtime_id
        runtime_sha256 = runtime.runtime_sha256
        config_sha256 = runtime.config_sha256
        accepts_real_rgb_only = True

        def analyze(self, **_values):
            return mixed_output

    with pytest.raises(ValueError, match="runtime measurement integrity failure"):
        pipeline.run_test(
            **kwargs,
            mask_runtime=MixedRuntime(),
            development_reference_store_loader=lambda: store,
        )

    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"
    assert not kwargs["evidence_out_store"].exists()


def test_cli_reference_arguments_are_conditional_and_store_name_is_unambiguous():
    from scripts.run_carryover_pipeline import build_parser

    help_text = build_parser().format_help()
    run_help = build_parser()._subparsers._group_actions[0].choices["run-test"].format_help()
    assert "--evidence-out-store" in run_help
    assert "--development-reference-store" in run_help
    assert "--store" not in run_help
    assert "run-test" in help_text


@pytest.mark.parametrize("occupied_kind", ["existing", "symlink", "alias"])
def test_output_store_no_clobber_and_alias_fail_before_either_loader(
    tmp_path, monkeypatch, occupied_kind
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, _runtime_output_value, store = _fixture(tmp_path, monkeypatch)
    calls = []
    if occupied_kind == "existing":
        kwargs["evidence_out_store"].write_bytes(b"existing")
        error = FileExistsError
    elif occupied_kind == "symlink":
        kwargs["evidence_out_store"].symlink_to(tmp_path / "missing-target.sqlite3")
        error = FileExistsError
    else:
        kwargs["evidence_out_store"] = kwargs["out"]
        error = ValueError

    with pytest.raises(error):
        pipeline.run_test(
            **kwargs,
            mask_runtime_loader=lambda: calls.append("runtime") or runtime,
            development_reference_store_loader=lambda: calls.append("reference") or store,
        )

    assert calls == []
    assert not kwargs["claim"].exists()


def test_reference_integrity_failure_is_fatal_and_upstreams_remain_immutable(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline
    from fluidmix.wp7_development_reference import DevelopmentReferenceError

    kwargs, runtime, _runtime_output_value, store = _fixture(tmp_path, monkeypatch)
    upstream = {
        path: _sha(path)
        for path in (
            kwargs["video"], kwargs["recipe_sequence"],
            kwargs["development_reference_store"],
        )
    }

    def corrupt_query(_target):
        raise DevelopmentReferenceError("configuration_donor_binding_invalid")

    monkeypatch.setattr(store, "reference_bank", corrupt_query)
    with pytest.raises(
        DevelopmentReferenceError, match="configuration_donor_binding_invalid"
    ):
        pipeline.run_test(
            **kwargs, mask_runtime=runtime,
            development_reference_store_loader=lambda: store,
        )

    assert {path: _sha(path) for path in upstream} == upstream
    assert not kwargs["evidence_out_store"].exists()
    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"


def test_runtime_artifact_hash_corruption_is_fatal_not_a_vial_abstention(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    image_id = next(iter(runtime_output.image_locators))
    rgb = Path(runtime_output.image_locators[image_id]["rgb_path"])
    rgb.write_bytes(b"corrupt after trusted binding")
    upstream = {
        path: _sha(path)
        for path in (
            kwargs["video"], kwargs["recipe_sequence"],
            kwargs["development_reference_store"],
        )
    }

    with pytest.raises(ValueError, match="RGB image SHA-256 mismatch"):
        pipeline.run_test(
            **kwargs, mask_runtime=runtime,
            development_reference_store_loader=lambda: store,
        )

    assert {path: _sha(path) for path in upstream} == upstream
    assert not kwargs["evidence_out_store"].exists()
    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"


def test_pipeline_source_hash_covers_each_production_dependency_once(monkeypatch):
    import fluidmix.carryover_pipeline as pipeline

    observed = []
    original = pipeline._read_regular_bytes

    def tracked(path, role):
        observed.append(Path(path).relative_to(ROOT).as_posix())
        return original(path, role)

    monkeypatch.setattr(pipeline, "_read_regular_bytes", tracked)
    digest = pipeline.pipeline_source_sha256()
    assert len(digest) == 64
    assert len(observed) == len(set(observed))
    assert set(observed) == {
        "scripts/run_carryover_pipeline.py",
        "src/fluidmix/carryover_pipeline.py",
        "src/fluidmix/evidence_store.py",
        "src/fluidmix/part_b_evidence.py",
        "src/fluidmix/vision_measurement.py",
        "src/fluidmix/wp7_development_reference.py",
        "src/fluidmix/wp7_final_decision.py",
        "src/fluidmix/wp7_rgb_runtime.py",
    }
