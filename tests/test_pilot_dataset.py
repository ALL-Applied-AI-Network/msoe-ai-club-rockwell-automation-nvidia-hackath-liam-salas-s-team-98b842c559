"""Contracts for the frozen eight-row appearance-pilot artifact boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CURRENT_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
sys.path.insert(0, os.fspath(ROOT / "src"))

from fluidmix.pilot_dataset import (  # noqa: E402
    CONFIG_PATH,
    build_pilot_dataset,
    build_experiment_graph,
    canonical_pilot_config,
    check_pilot_dataset,
    load_context_features,
    load_inference_rgb,
    launch_pilot_attempt,
    load_pilot_config,
    load_prepublication_diagnostic_inputs,
    load_training_samples,
    validate_experiment_graph,
    validate_pilot_config,
    write_attempt_manifests,
)

from pilot_fixture import build_valid_pilot


def test_tracked_pilot_config_is_the_exact_frozen_acceptance_matrix():
    loaded = load_pilot_config(CONFIG_PATH, repo_root=ROOT)

    assert loaded["payload"] == canonical_pilot_config()
    assert loaded["payload"]["purpose"] == "acceptance_only"
    assert loaded["payload"]["training_eligible"] is False
    assert loaded["payload"]["recipe_appearance_mapping_validated"] is False
    assert [row["sample_id"] for row in loaded["payload"]["rows"]] == [
        f"R{index:03d}" for index in range(8)
    ]
    assert loaded["payload"]["rows"][0]["appearance_blend_alpha"] is None
    assert loaded["payload"]["rows"][1]["appearance_blend_alpha"] is None
    assert loaded["payload"]["rows"][2]["appearance_blend_alpha"] == 0.30
    assert loaded["payload"]["rows"][5]["appearance_blend_alpha"] == 0.10


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["rows"][0].update({"expected_recipe_color": [0, 0, 0]}), "exact"),
        (lambda value: value["rows"][2].update({"appearance_blend_alpha": True}), "exact"),
        (lambda value: value["rows"][2].update({"run_position": True}), "exact"),
        (lambda value: value["rows"][2].update({"partition": "test"}), "exact"),
        (lambda value: value["controls"].update({"seed": 2_000_001}), "exact"),
    ],
)
def test_config_validator_rejects_forbidden_extra_type_and_split_mutations(
    mutation, match
):
    candidate = copy.deepcopy(canonical_pilot_config())
    mutation(candidate)

    with pytest.raises(ValueError, match=match):
        validate_pilot_config(candidate)


def test_config_loader_rejects_duplicate_json_keys(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')

    with pytest.raises(ValueError, match="duplicate"):
        load_pilot_config(duplicate, repo_root=ROOT)


@pytest.mark.parametrize("damage", ["seed_range", "reserved_real"])
def test_config_loader_rechecks_current_split_seed_and_reserved_real_contract(tmp_path, damage):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("sim_recipes.csv", "sim_transitions.csv"):
        (data / name).write_bytes((ROOT / "data" / name).read_bytes())
    spec = json.loads((ROOT / "data/sim_split_spec.json").read_text())
    if damage == "seed_range":
        spec["render_seed_ranges"]["train"] = [0, 100]
    else:
        spec["real_recipes_reserved"]["fixture"] = {
            "t100": 0,
            "t200": 10,
            "t400": 20,
        }
    (data / "sim_split_spec.json").write_text(json.dumps(spec))

    with pytest.raises(ValueError, match="seed|reserved real"):
        load_pilot_config(CONFIG_PATH, repo_root=tmp_path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_graph_inputs(pilot_root: Path, *, commit: str = CURRENT_COMMIT) -> None:
    config = canonical_pilot_config()
    source = pilot_root / "source-twin/stations/fill_station/fill_station.usda"
    source.parent.mkdir(parents=True)
    source.write_text("#usda 1.0\n# fixture canonical source\n")
    (pilot_root / "appearance-layers").mkdir(parents=True)
    (pilot_root / "attempt-manifests").mkdir()
    for row in config["rows"]:
        sample_id = row["sample_id"]
        layer = pilot_root / "appearance-layers" / f"{sample_id}.usda"
        layer.write_text(f"#usda 1.0\n# fixture {sample_id}\n")
        appearance = config["appearances"][row["appearance_id"]]
        appearance_layer = {
            "schema": "direct_appearance_v1",
            "variant_id": sample_id,
            "stage_sha256": _sha256(layer),
            "stage_bytes": layer.stat().st_size,
            "source_stage": "stations/fill_station/fill_station.usda",
            "source_stage_sha256": _sha256(source),
            "material_path": "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF",
            "diffuse_color_linear": appearance["linear_diffuse_rgb"],
            "opacity": appearance["opacity"],
        }
        manifest = {
            "schema_version": 1,
            "run_id": sample_id,
            "hypothesis": "The fixed renderer emits one mechanically valid pilot capture.",
            "falsifier": "Any lifecycle, provenance, or capture gate fails.",
            "command_plan": {
                "entry_point": "scripts/render_twin_dataset.py",
                "implementation_commit": commit,
                "command": [
                    "/usr/bin/python3",
                    "scripts/render_twin_dataset.py",
                    "--isaac-python",
                    "/opt/isaac/python.sh",
                    "--runtime-lock",
                    "/opt/isaac/runtime-lock.json",
                    "--experiment-manifest",
                    str((pilot_root / "attempt-manifests" / f"{sample_id}.json").resolve()),
                    "--twin",
                    "/opt/twin",
                    "--run-id",
                    sample_id,
                    "--out-parent",
                    str(pilot_root.resolve()),
                    "--appearance-stage",
                    str(layer.resolve()),
                    "--frames",
                    "686:687:1",
                    "--subframes",
                    "64",
                    "--warmup",
                    "60",
                    "--max-tries",
                    "20",
                    "--seed",
                    "310241",
                    "--require-liquid-pixels",
                    "--renderer",
                    "PathTracing",
                    "--timeout",
                    "1800",
                    "--environment-profile",
                    "gb10-aarch64",
                ],
                "appearance_layer": appearance_layer,
                "requested_controls": config["controls"],
                "process_attempt_limit": 1,
                "replacement_process_attempts": 0,
            },
            "seed": config["controls"]["seed"],
            "denominators": {
                "pilot_rows": 8,
                "requested_frames": 1,
                "process_attempts": 1,
            },
            "stop_conditions": [
                "no replacement process attempt",
                "bounded timeout",
                "provenance or safety failure",
            ],
        }
        (pilot_root / "attempt-manifests" / f"{sample_id}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


def test_graph_builder_hash_binds_exact_order_layers_manifests_and_controls(tmp_path):
    _write_graph_inputs(tmp_path)

    graph_path = build_experiment_graph(
        tmp_path,
        config_path=CONFIG_PATH,
        implementation_commit=CURRENT_COMMIT,
        repo_root=ROOT,
    )
    graph = validate_experiment_graph(
        graph_path,
        pilot_root=tmp_path,
        config_path=CONFIG_PATH,
        repo_root=ROOT,
    )

    assert graph["payload"]["ordered_sample_ids"] == [f"R{i:03d}" for i in range(8)]
    assert graph["payload"]["denominators"] == {
        "process_attempts": 8,
        "replacement_process_attempts": 0,
        "shared_capture_tries_per_process": 20,
    }
    assert len(graph["payload"]["entries"]) == 8
    assert graph_path == tmp_path / "experiment-graph.json"
    with pytest.raises(FileExistsError):
        build_experiment_graph(
            tmp_path,
            config_path=CONFIG_PATH,
            implementation_commit=CURRENT_COMMIT,
            repo_root=ROOT,
        )


def test_attempt_manifest_authoring_is_deterministic_exact_and_no_clobber(tmp_path):
    source = tmp_path / "source-twin/stations/fill_station/fill_station.usda"
    source.parent.mkdir(parents=True)
    source.write_text("#usda 1.0\n# source\n")
    (tmp_path / "appearance-layers").mkdir()
    (tmp_path / "layer-inspections").mkdir()
    config = canonical_pilot_config()
    for row in config["rows"]:
        sample_id = row["sample_id"]
        layer = tmp_path / "appearance-layers" / f"{sample_id}.usda"
        layer.write_text(f"#usda 1.0\n# {sample_id}\n")
        appearance = config["appearances"][row["appearance_id"]]
        inspection = {
            "path": str(layer.resolve()),
            "schema": "direct_appearance_v1",
            "variant_id": sample_id,
            "stage_sha256": _sha256(layer),
            "stage_bytes": layer.stat().st_size,
            "source_stage": "stations/fill_station/fill_station.usda",
            "source_stage_sha256": _sha256(source),
            "material_path": "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF",
            "diffuse_color_linear": appearance["linear_diffuse_rgb"],
            "opacity": appearance["opacity"],
            "uncalibrated": True,
            "timing": {"start_time_code": 0.0, "end_time_code": 686.0, "frames_per_second": 60.0, "time_codes_per_second": 60.0},
            "non_session_usd_layer_stack": [
                {"ordinal": 1, "resolved_path": str(layer.resolve()), "sha256": _sha256(layer), "bytes": layer.stat().st_size},
                {"ordinal": 2, "resolved_path": str(source.resolve()), "sha256": _sha256(source), "bytes": source.stat().st_size},
            ],
        }
        (tmp_path / "layer-inspections" / f"{sample_id}.json").write_text(json.dumps(inspection))

    paths = write_attempt_manifests(
        tmp_path,
        config_path=CONFIG_PATH,
        implementation_commit=CURRENT_COMMIT,
        controller_python="/usr/bin/python3",
        isaac_python="/opt/isaac/python.sh",
        runtime_lock="/opt/isaac/runtime-lock.json",
        twin="/opt/twin",
        repo_root=ROOT,
    )

    assert [path.name for path in paths] == [f"R{i:03d}.json" for i in range(8)]
    assert json.loads(paths[0].read_text())["command_plan"]["appearance_layer"]["variant_id"] == "R000"
    with pytest.raises(FileExistsError):
        write_attempt_manifests(
            tmp_path,
            config_path=CONFIG_PATH,
            implementation_commit=CURRENT_COMMIT,
            controller_python="/usr/bin/python3",
            isaac_python="/opt/isaac/python.sh",
            runtime_lock="/opt/isaac/runtime-lock.json",
            twin="/opt/twin",
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    "damage",
    ["layer_bytes", "manifest_control", "command_control", "manifest_extra", "symlink"],
)
def test_graph_validator_rejects_drift_schema_and_symlink_inputs(tmp_path, damage):
    _write_graph_inputs(tmp_path)
    graph_path = build_experiment_graph(
        tmp_path,
        config_path=CONFIG_PATH,
        implementation_commit=CURRENT_COMMIT,
        repo_root=ROOT,
    )
    if damage == "layer_bytes":
        (tmp_path / "appearance-layers/R003.usda").write_text("drift")
    elif damage == "symlink":
        layer = tmp_path / "appearance-layers/R003.usda"
        target = tmp_path / "target.usda"
        target.write_bytes(layer.read_bytes())
        layer.unlink()
        layer.symlink_to(target)
    else:
        manifest_path = tmp_path / "attempt-manifests/R003.json"
        manifest = json.loads(manifest_path.read_text())
        if damage == "manifest_control":
            manifest["seed"] += 1
        elif damage == "command_control":
            index = manifest["command_plan"]["command"].index("--frames")
            manifest["command_plan"]["command"][index + 1] = "685:686:1"
        else:
            manifest["command_plan"]["oracle_mask"] = True
        manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        validate_experiment_graph(
            graph_path,
            pilot_root=tmp_path,
            config_path=CONFIG_PATH,
            repo_root=ROOT,
        )


def test_graph_builder_rejects_layers_bound_to_mixed_source_stage_hashes(tmp_path):
    _write_graph_inputs(tmp_path)
    path = tmp_path / "attempt-manifests/R007.json"
    manifest = json.loads(path.read_text())
    manifest["command_plan"]["appearance_layer"]["source_stage_sha256"] = "c" * 64
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="one unchanged source"):
        build_experiment_graph(
            tmp_path,
            config_path=CONFIG_PATH,
            implementation_commit=CURRENT_COMMIT,
            repo_root=ROOT,
        )


@pytest.mark.parametrize("extra", ["appearance-layers/extra.usda", "attempt-manifests/extra.json"])
def test_graph_builder_rejects_unbound_extra_preregistration_files(tmp_path, extra):
    _write_graph_inputs(tmp_path)
    (tmp_path / extra).write_text("extra")

    with pytest.raises(ValueError, match="exactly"):
        build_experiment_graph(
            tmp_path,
            config_path=CONFIG_PATH,
            implementation_commit=CURRENT_COMMIT,
            repo_root=ROOT,
        )


def test_graph_must_be_frozen_before_any_process_attempt_starts(tmp_path):
    _write_graph_inputs(tmp_path)
    (tmp_path / "R000").mkdir()
    (tmp_path / "R000/started").write_text("attempted")

    with pytest.raises(ValueError, match="before attempt"):
        build_experiment_graph(
            tmp_path,
            config_path=CONFIG_PATH,
            implementation_commit=CURRENT_COMMIT,
            repo_root=ROOT,
        )


def test_post_review_builder_publishes_separate_hash_bound_artifacts_and_receipt(tmp_path):
    build_valid_pilot(tmp_path, ROOT)

    report = check_pilot_dataset(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)
    receipt = json.loads((tmp_path / "dataset-receipt.json").read_text())

    assert report == {
        "dataset_version": "twin-appearance-pilot-v1",
        "purpose": "acceptance_only",
        "training_eligible": False,
        "rows": 8,
        "mechanical_rows_passed": 8,
        "visual_review_status": "accepted",
        "mask_sha256": report["mask_sha256"],
        "semantic_sha256": report["semantic_sha256"],
        "instance_sha256": report["instance_sha256"],
    }
    assert set(receipt["artifacts"]) == {"model_input", "label_provenance"}
    assert receipt["purpose"] == "acceptance_only"
    assert receipt["training_eligible"] is False
    with pytest.raises(FileExistsError):
        build_pilot_dataset(
            tmp_path,
            config_path=CONFIG_PATH,
            graph_path=tmp_path / "experiment-graph.json",
            visual_review_path=tmp_path / "reviews/visual-review.json",
            repo_root=ROOT,
        )


def test_visual_review_relative_path_is_resolved_from_pilot_root(tmp_path):
    build_valid_pilot(tmp_path, ROOT, publish=False)
    review = json.loads((tmp_path / "reviews/visual-review.json").read_text())

    assert [item["sample_id"] for item in review["reviewed_artifacts"]] == [
        f"R{index:03d}" for index in range(8)
    ]

    paths = build_pilot_dataset(
        tmp_path,
        config_path=CONFIG_PATH,
        graph_path=tmp_path / "experiment-graph.json",
        visual_review_path=Path("reviews/visual-review.json"),
        repo_root=ROOT,
    )

    assert paths["receipt"] == tmp_path / "dataset-receipt.json"


def test_builder_rejects_visual_review_that_does_not_bind_current_artifacts(tmp_path):
    build_valid_pilot(tmp_path, ROOT, publish=False)
    review_path = tmp_path / "reviews/visual-review.json"
    review = json.loads(review_path.read_text())
    review["reviewed_artifacts"][0]["rgb_sha256"] = "0" * 64
    review_path.write_text(json.dumps(review))

    with pytest.raises(ValueError, match="visual review"):
        build_pilot_dataset(
            tmp_path,
            config_path=CONFIG_PATH,
            graph_path=tmp_path / "experiment-graph.json",
            visual_review_path=review_path,
            repo_root=ROOT,
        )


def test_loaders_expose_rgb_only_mask_only_as_target_and_exact_context_allowlist(tmp_path):
    build_valid_pilot(tmp_path, ROOT)

    rgb = load_inference_rgb(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)
    training = load_training_samples(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)
    contexts = load_context_features(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)

    assert len(rgb) == len(training) == len(contexts) == 8
    assert all(isinstance(value, np.ndarray) and value.shape == (480, 640, 3) for value in rgb)
    assert all(value.dtype == np.uint8 for value in rgb)
    for index, (pixels, target) in enumerate(training):
        assert np.array_equal(pixels, rgb[index])
        assert target.shape == (480, 640)
        assert target.dtype == np.bool_
    assert set(contexts[0]) == {
        "view_id",
        "current_programmed_ml",
        "previous_programmed_ml",
        "programmed_dispense_order",
        "run_position",
        "recipe_changed",
    }
    assert not ({"sample_id", "artifact", "label", "mask", "sha256", "path"} & set(contexts[0]))


def _rehash_dataset_file_in_receipt(root: Path, key: str) -> None:
    receipt_path = root / "dataset-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    record = receipt["artifacts"][key]
    artifact = root / record["path"]
    record["sha256"] = _sha256(artifact)
    record["bytes"] = artifact.stat().st_size
    record["rows"] = len(artifact.read_text().splitlines())
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row["features"].update({"sequence_id": "Q000"}), "features|schema"),
        (lambda row: row["features"]["current_programmed_ml"].update({"label": 1}), "features|schema"),
        (lambda row: row["artifact"].update({"path": "R000/capture/liquid/0686.png"}), "RGB|artifact|model"),
        (lambda row: row.update({"partition": "train"}), "fields|schema|model"),
    ],
)
def test_recursive_feature_and_artifact_role_mutations_never_reach_loaders(
    tmp_path, mutation, match
):
    build_valid_pilot(tmp_path, ROOT)
    model_path = tmp_path / "model_input.jsonl"
    rows = [json.loads(line) for line in model_path.read_text().splitlines()]
    mutation(rows[0])
    model_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    _rehash_dataset_file_in_receipt(tmp_path, "model_input")

    with pytest.raises(ValueError, match=match):
        load_inference_rgb(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)
    with pytest.raises(ValueError):
        load_training_samples(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)
    with pytest.raises(ValueError):
        load_context_features(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)


@pytest.mark.parametrize("mutation", ["join", "partition", "seed", "alpha", "class", "mask_role"])
def test_label_join_split_matrix_and_oracle_mutations_fail_closed(tmp_path, mutation):
    build_valid_pilot(tmp_path, ROOT)
    label_path = tmp_path / "label_provenance.jsonl"
    rows = [json.loads(line) for line in label_path.read_text().splitlines()]
    row = rows[0]
    if mutation == "join":
        row["sample_id"] = "R777"
    elif mutation == "partition":
        row["partition"] = "test"
    elif mutation == "seed":
        row["capture"]["seed"] = 2_000_000
    elif mutation == "alpha":
        row["label"]["appearance_blend_alpha"] = 0.5
    elif mutation == "class":
        row["taxonomy"]["liquid_class_ids"] = [77]
    else:
        row["oracle_artifacts"]["mask"]["path"] = "R000/capture/rgb/0686.png"
    label_path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in rows))
    _rehash_dataset_file_in_receipt(tmp_path, "label_provenance")

    with pytest.raises(ValueError):
        check_pilot_dataset(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        "rgb_hash",
        "condition",
        "semantic_path",
        "instance_path",
        "mask_hash",
        "appearance_variant",
        "layer_hash",
        "duplicate_id",
    ],
)
def test_hash_condition_oracle_variant_and_duplicate_mutations_fail_closed(tmp_path, mutation):
    build_valid_pilot(tmp_path, ROOT)
    model_path = tmp_path / "model_input.jsonl"
    label_path = tmp_path / "label_provenance.jsonl"
    models = [json.loads(line) for line in model_path.read_text().splitlines()]
    labels = [json.loads(line) for line in label_path.read_text().splitlines()]
    if mutation == "rgb_hash":
        models[0]["artifact"]["sha256"] = "0" * 64
    elif mutation == "condition":
        models[0]["features"]["recipe_changed"] = True
    elif mutation == "semantic_path":
        labels[0]["oracle_artifacts"]["semantic"]["path"] = labels[0]["oracle_artifacts"]["instance"]["path"]
    elif mutation == "instance_path":
        labels[0]["oracle_artifacts"]["instance"]["path"] = labels[0]["oracle_artifacts"]["semantic"]["path"]
    elif mutation == "mask_hash":
        labels[0]["oracle_artifacts"]["mask"]["sha256"] = "0" * 64
    elif mutation == "appearance_variant":
        labels[0]["appearance"]["authored_appearance_id"] = "A"
    elif mutation == "layer_hash":
        labels[0]["capture"]["layer"]["sha256"] = "0" * 64
    else:
        models[1]["sample_id"] = models[0]["sample_id"]
    model_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in models))
    label_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels))
    _rehash_dataset_file_in_receipt(tmp_path, "model_input")
    _rehash_dataset_file_in_receipt(tmp_path, "label_provenance")

    with pytest.raises(ValueError):
        check_pilot_dataset(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)


def test_checker_rejects_symlinked_rgb_even_when_target_bytes_match(tmp_path):
    build_valid_pilot(tmp_path, ROOT)
    rgb = tmp_path / "R000/capture/rgb/0686.png"
    target = tmp_path / "copied-rgb.png"
    target.write_bytes(rgb.read_bytes())
    rgb.unlink()
    rgb.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        check_pilot_dataset(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)


def test_checker_rejects_an_unregistered_replacement_attempt(tmp_path):
    build_valid_pilot(tmp_path, ROOT)
    (tmp_path / "R000-retry").mkdir()

    with pytest.raises(ValueError, match="exact eight|replacement"):
        check_pilot_dataset(tmp_path, config_path=CONFIG_PATH, repo_root=ROOT)


@pytest.mark.parametrize("damage", ["inner_argv", "runtime_lock", "profile", "twin"])
def test_prepublication_gate_rejects_actual_runtime_drift_from_command_plan(tmp_path, damage):
    build_valid_pilot(tmp_path, ROOT, publish=False)
    attempt_path = tmp_path / "R000/attempt.json"
    attempt = json.loads(attempt_path.read_text())
    if damage == "inner_argv":
        attempt["safe_command"][attempt["safe_command"].index("--seed") + 1] = "310242"
    elif damage == "runtime_lock":
        attempt["provenance"]["files"]["runtime_lock"]["path"] = "/wrong/lock"
    elif damage == "profile":
        attempt["environment_policy"]["profile"] = "generic"
    else:
        attempt["provenance"]["twin"]["root"] = "/wrong/twin"
    attempt_path.write_text(json.dumps(attempt))

    with pytest.raises(ValueError, match="runtime provenance|renderer argv"):
        load_prepublication_diagnostic_inputs(tmp_path, repo_root=ROOT)


def test_graph_reverification_rejects_bound_implementation_file_drift(tmp_path, monkeypatch):
    import fluidmix.pilot_dataset as pilot_dataset

    _write_graph_inputs(tmp_path)
    graph_path = build_experiment_graph(
        tmp_path,
        config_path=CONFIG_PATH,
        implementation_commit=CURRENT_COMMIT,
        repo_root=ROOT,
    )
    real_hash = pilot_dataset.hash_regular_file

    def drift(path):
        record = real_hash(path)
        if Path(path).name == "pilot_diagnostic.py":
            return {"sha256": "0" * 64, "bytes": record["bytes"]}
        return record

    monkeypatch.setattr(pilot_dataset, "hash_regular_file", drift)

    with pytest.raises(ValueError, match="graph"):
        validate_experiment_graph(
            graph_path,
            pilot_root=tmp_path,
            config_path=CONFIG_PATH,
            repo_root=ROOT,
        )


def test_launch_path_spawns_the_frozen_outer_command_verbatim(tmp_path, monkeypatch):
    import fluidmix.pilot_dataset as pilot_dataset

    _write_graph_inputs(tmp_path)
    build_experiment_graph(
        tmp_path,
        config_path=CONFIG_PATH,
        implementation_commit=CURRENT_COMMIT,
        repo_root=ROOT,
    )
    manifest = json.loads((tmp_path / "attempt-manifests/R000.json").read_text())
    calls = []

    def record(command, *, cwd, check):
        calls.append((command, cwd, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pilot_dataset.subprocess, "run", record)

    returncode = launch_pilot_attempt(
        tmp_path,
        "R000",
        config_path=CONFIG_PATH,
        repo_root=ROOT,
    )

    assert returncode == 0
    assert calls == [
        (
            manifest["command_plan"]["command"],
            ROOT.resolve(),
            False,
        )
    ]


def test_launch_path_rejects_out_of_order_or_replacement_attempt(tmp_path):
    _write_graph_inputs(tmp_path)
    build_experiment_graph(
        tmp_path,
        config_path=CONFIG_PATH,
        implementation_commit=CURRENT_COMMIT,
        repo_root=ROOT,
    )
    (tmp_path / "R001").mkdir()

    with pytest.raises(ValueError, match="order|replacement"):
        launch_pilot_attempt(
            tmp_path,
            "R000",
            config_path=CONFIG_PATH,
            repo_root=ROOT,
        )
