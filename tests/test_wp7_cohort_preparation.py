"""Metadata-only preparation for the accepted WP7 cohort-goldens route."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _metadata_fixture(root: Path) -> dict[str, object]:
    root.mkdir()
    case_ids = [f"CASE-{index:02d}" for index in range(1, 19)]
    def vials(run_cases: list[str], first_block: int) -> list[dict[str, object]]:
        return [
            {
                "case_id": case_id, "vial_ordinal": index + 1,
                "block_recipe_id": f"R-{first_block + index // 3}",
                "block_position": index % 3 + 1,
                "recipe_id": f"R-{first_block + index // 3}",
            }
            for index, case_id in enumerate(run_cases)
        ]

    sequence = {
        "schema_version": "carryover-recipe-sequence-v1",
        "split_role": "test",
        "split_version": "synthetic-v1",
        "runs": [
            {"run_id": "RUN-1", "vials": vials(case_ids[:9], 1)},
            {"run_id": "RUN-2", "vials": vials(case_ids[9:], 4)},
        ],
    }
    sequence_path = root / "recipe.json"
    sequence_sha = _write(sequence_path, sequence)
    run_specs = (("RUN-1", case_ids[:9]), ("RUN-2", case_ids[9:]))
    run_paths, run_shas, bindings = [], [], []
    for index, (run_id, run_cases) in enumerate(run_specs, 1):
        path = root / f"run-{index}.json"
        ranges = [(segment, segment * 3, segment * 3 + 1)
                  for segment in range(10)]
        frames = 28
        value = {
            "artifact_type": "rendered_transport_run_sequence_v4",
            "status": "pass",
            "run_id": run_id,
            "case_ids": run_cases,
            "output_video_sha256": f"{index:064x}",
            "output_frame_count": frames,
            "encoded_transition_evidence": [
                {
                    "segment_index": segment,
                    "start_frame_index": start,
                    "stop_frame_index": stop,
                    "frame_count": stop - start,
                    "decoded_bgr_sha256": [f"{segment + index:064x}"] * (stop - start),
                }
                for segment, start, stop in ranges
            ],
            "fill_clips": [
                {"case_id": case_id, "frame_count": 3}
                for case_id in run_cases
            ],
        }
        digest = _write(path, value)
        run_paths.append(path)
        run_shas.append(digest)
        bindings.append({
            "run_id": run_id, "case_ids": run_cases,
            "run_receipt_path": str(path.absolute()),
            "run_receipt_sha256": digest,
            "video_sha256": f"{index:064x}", "frame_count": frames,
        })
    video_sha = hashlib.sha256(b"synthetic-video").hexdigest()
    composition = {
        "artifact_type": "wp7_ordered_test_video_v1", "status": "pass",
        "output": {"sha256": video_sha, "frame_count": 56,
                   "path": str(root / "does-not-exist.mp4")},
        "ordered_runs": bindings,
        "verification": {"case_count": 18, "run_count": 2,
                         "source_order": ["RUN-1", "RUN-2"]},
        "provenance": {"prepared_recipe": {
            "path": str(sequence_path.absolute()), "sha256": sequence_sha,
            "run_order": ["RUN-1", "RUN-2"], "case_order": case_ids,
        }},
    }
    composition_path = root / "composition.json"
    composition_sha = _write(composition_path, composition)
    review = {
        "artifact_type": "fluidmix_wp7_ordered_test_video_final_output_review_v1",
        "status": "pass", "verdict": "pass",
        "identity": {"output_mp4_sha256": video_sha,
                     "output_receipt_sha256": composition_sha,
                     "recipe": {"path": str(sequence_path.absolute()),
                                "sha256": sequence_sha}},
        "case_contract": {"count": 18, "unique": True},
        "output_media": {"sha256": video_sha, "frame_count": 56},
    }
    review_path = root / "review.json"
    review_sha = _write(review_path, review)
    return {
        "recipe_sequence": sequence_path,
        "expected_recipe_sequence_sha256": sequence_sha,
        "composition_receipt": composition_path,
        "expected_composition_receipt_sha256": composition_sha,
        "final_review": review_path,
        "expected_final_review_sha256": review_sha,
        "run_receipts": tuple(run_paths),
        "expected_run_receipt_sha256s": tuple(run_shas),
        "video_sha": video_sha,
        "sequence": sequence,
        "case_ids": case_ids,
    }


def _synthetic_profile(values: dict[str, object]):
    from fluidmix.wp7_cohort_preparation import _AcceptedRunTestProfile

    sequence = values["sequence"]
    cohorts = []
    for run in sequence["runs"]:
        for start in range(0, 9, 3):
            rows = run["vials"][start:start + 3]
            cohorts.append((run["run_id"], rows[0]["recipe_id"],
                            tuple(row["case_id"] for row in rows)))
    return _AcceptedRunTestProfile(
        recipe_path=values["recipe_sequence"].absolute(),
        recipe_sha256=values["expected_recipe_sequence_sha256"],
        composition_path=values["composition_receipt"].absolute(),
        composition_sha256=values["expected_composition_receipt_sha256"],
        review_path=values["final_review"].absolute(),
        review_sha256=values["expected_final_review_sha256"],
        run_paths=tuple(path.absolute() for path in values["run_receipts"]),
        run_sha256s=values["expected_run_receipt_sha256s"],
        video_sha256=values["video_sha"], total_frame_count=56,
        run_frame_counts=(28, 28), fill_count=18, transition_count=19,
        boundary_transition=(9, 27, 29, 2), run2_first_fill_frame_count=3,
        first_transition=(0, 1), last_transition=(55, 56),
        cohorts=tuple(cohorts),
    )


def _arguments(values: dict[str, object]) -> dict[str, object]:
    return {key: values[key] for key in (
        "recipe_sequence", "expected_recipe_sequence_sha256", "composition_receipt",
        "expected_composition_receipt_sha256", "final_review",
        "expected_final_review_sha256", "run_receipts",
        "expected_run_receipt_sha256s",
    )}


def _refresh_receipt_chain(values: dict[str, object]) -> None:
    run_values = [json.loads(path.read_text()) for path in values["run_receipts"]]
    run_shas = tuple(_write(path, value) for path, value in zip(
        values["run_receipts"], run_values, strict=True
    ))
    values["expected_run_receipt_sha256s"] = run_shas
    composition = json.loads(values["composition_receipt"].read_text())
    for binding, path, digest, run in zip(
        composition["ordered_runs"], values["run_receipts"], run_shas,
        run_values, strict=True,
    ):
        binding["run_receipt_path"] = str(path.absolute())
        binding["run_receipt_sha256"] = digest
        binding["frame_count"] = run["output_frame_count"]
    composition_sha = _write(values["composition_receipt"], composition)
    values["expected_composition_receipt_sha256"] = composition_sha
    review = json.loads(values["final_review"].read_text())
    review["identity"]["output_receipt_sha256"] = composition_sha
    values["expected_final_review_sha256"] = _write(values["final_review"], review)


def test_public_gate_pins_exact_accepted_paths_and_hashes(tmp_path: Path) -> None:
    import fluidmix.wp7_cohort_preparation as preparation

    profile = preparation._ACCEPTED_RUN_TEST_PROFILE
    assert profile.video_sha256 == "1021878c646f1019ac941ab645520446441d6f401ff664d05d5a4e64a682760b"
    assert profile.run_frame_counts == (3574, 3748)
    assert profile.boundary_transition == (9, 3561, 3587, 26)
    accepted = {
        "recipe_sequence": profile.recipe_path,
        "expected_recipe_sequence_sha256": profile.recipe_sha256,
        "composition_receipt": profile.composition_path,
        "expected_composition_receipt_sha256": profile.composition_sha256,
        "final_review": profile.review_path,
        "expected_final_review_sha256": profile.review_sha256,
        "run_receipts": profile.run_paths,
        "expected_run_receipt_sha256s": profile.run_sha256s,
        "destination": tmp_path / "never",
    }
    for field, replacement in (
        ("recipe_sequence", tmp_path / "accepted-other-recipe.json"),
        ("expected_recipe_sequence_sha256", "0" * 64),
        ("composition_receipt", tmp_path / "other-composition.json"),
        ("run_receipts", (profile.run_paths[1], profile.run_paths[0])),
    ):
        changed = dict(accepted)
        changed[field] = replacement
        with pytest.raises(ValueError, match="immutable accepted profile"):
            preparation.prepare_run_test_inputs(**changed)


@pytest.mark.parametrize("binding", [
    "composition_path", "composition_hash", "composition_cases", "composition_runs",
    "review_path", "review_hash",
])
def test_recipe_cross_bindings_are_required(
    tmp_path: Path, binding: str
) -> None:
    from fluidmix.wp7_cohort_preparation import _prepare_run_test_inputs_for_profile

    values = _metadata_fixture(tmp_path / "source")
    profile = _synthetic_profile(values)
    if binding.startswith("composition"):
        receipt = json.loads(values["composition_receipt"].read_text())
        if binding == "composition_cases":
            receipt["provenance"]["prepared_recipe"]["case_order"].reverse()
        elif binding == "composition_runs":
            receipt["provenance"]["prepared_recipe"]["run_order"].reverse()
        else:
            key = "path" if binding.endswith("path") else "sha256"
            receipt["provenance"]["prepared_recipe"][key] = "wrong"
        _write(values["composition_receipt"], receipt)
    else:
        review = json.loads(values["final_review"].read_text())
        key = "path" if binding.endswith("path") else "sha256"
        review["identity"]["recipe"][key] = "wrong"
        _write(values["final_review"], review)
    _refresh_receipt_chain(values)
    with pytest.raises(ValueError, match="recipe binding|prepared-recipe"):
        _prepare_run_test_inputs_for_profile(
            **_arguments(values), destination=tmp_path / "never", profile=profile
        )


@pytest.mark.parametrize("mutation", ["alternate_geometry", "boundary", "run2_first_fill", "cohort"])
def test_exact_geometry_and_cohort_drift_fail(
    tmp_path: Path, mutation: str
) -> None:
    from fluidmix.wp7_cohort_preparation import _prepare_run_test_inputs_for_profile

    values = _metadata_fixture(tmp_path / "source")
    profile = _synthetic_profile(values)
    if mutation == "cohort":
        sequence = json.loads(values["recipe_sequence"].read_text())
        sequence["runs"][0]["vials"][1]["block_position"] = 1
        recipe_sha = _write(values["recipe_sequence"], sequence)
        values["expected_recipe_sequence_sha256"] = recipe_sha
        composition = json.loads(values["composition_receipt"].read_text())
        composition["provenance"]["prepared_recipe"]["sha256"] = recipe_sha
        _write(values["composition_receipt"], composition)
        review = json.loads(values["final_review"].read_text())
        review["identity"]["recipe"]["sha256"] = recipe_sha
        _write(values["final_review"], review)
    else:
        run2 = json.loads(values["run_receipts"][1].read_text())
        if mutation == "alternate_geometry":
            run2["output_frame_count"] = 29
            run2["encoded_transition_evidence"][-1]["stop_frame_index"] = 29
            run2["encoded_transition_evidence"][-1]["frame_count"] = 2
            run2["encoded_transition_evidence"][-1]["decoded_bgr_sha256"] *= 2
            composition = json.loads(values["composition_receipt"].read_text())
            composition["output"]["frame_count"] = 57
            _write(values["composition_receipt"], composition)
            review = json.loads(values["final_review"].read_text())
            review["output_media"]["frame_count"] = 57
            _write(values["final_review"], review)
        elif mutation == "boundary":
            run2["encoded_transition_evidence"][0]["stop_frame_index"] = 2
            run2["encoded_transition_evidence"][0]["frame_count"] = 2
            run2["encoded_transition_evidence"][0]["decoded_bgr_sha256"] *= 2
        else:
            run2["fill_clips"][0]["frame_count"] = 4
        _write(values["run_receipts"][1], run2)
    _refresh_receipt_chain(values)
    with pytest.raises(ValueError, match="geometry|timing|cohort"):
        _prepare_run_test_inputs_for_profile(
            **_arguments(values), destination=tmp_path / "never", profile=profile
        )


def test_declared_mp4_is_never_opened_read_statted_or_resolved(
    tmp_path: Path, monkeypatch
) -> None:
    from fluidmix.wp7_cohort_preparation import _prepare_run_test_inputs_for_profile

    values = _metadata_fixture(tmp_path / "source")
    profile = _synthetic_profile(values)
    originals = {
        "open": Path.open, "read_bytes": Path.read_bytes,
        "stat": Path.stat, "resolve": Path.resolve,
    }
    def guard(name):
        def wrapped(path, *args, **kwargs):
            if Path(path).suffix.lower() == ".mp4":
                pytest.fail(f"declared MP4 reached Path.{name}")
            return originals[name](path, *args, **kwargs)
        return wrapped
    for name in originals:
        monkeypatch.setattr(Path, name, guard(name))
    result = _prepare_run_test_inputs_for_profile(
        **_arguments(values), destination=tmp_path / "prepared", profile=profile
    )
    assert result.publication_receipt["inputs"]["declared_video"]["path_accessed"] is False
def test_prepare_capture_is_metadata_only_and_publishable(tmp_path: Path) -> None:
    from fluidmix.wp7_cohort_preparation import _prepare_run_test_inputs_for_profile
    from fluidmix.wp7_phase_capture import publish_phase_capture

    values = _metadata_fixture(tmp_path / "source")
    video = tmp_path / "does-not-exist.mp4"
    result = _prepare_run_test_inputs_for_profile(
        **{key: values[key] for key in (
            "recipe_sequence", "expected_recipe_sequence_sha256",
            "composition_receipt", "expected_composition_receipt_sha256",
            "final_review", "expected_final_review_sha256", "run_receipts",
            "expected_run_receipt_sha256s",
        )},
        destination=tmp_path / "prepared",
        profile=_synthetic_profile(values),
    )
    assert not video.exists()
    assert set(path.name for path in result.output_paths) == {
        "run-receipts.json", "source-01-composition.json", "source-02-review.json",
        "source-03-runs.json", "capture-plan.json", "route-decision.json",
        "execution-preflight.json", "publication-receipt.json",
    }
    assert json.loads((tmp_path / "prepared/capture-plan.json").read_text())[
        "encoded_transition_evidence"
    ][1]["decoded_bgr_sha256"] is None
    publication = json.loads((tmp_path / "prepared/publication-receipt.json").read_text())
    preflight = json.loads((tmp_path / "prepared/execution-preflight.json").read_text())
    assert publication["canonical_bindings"]["execution_preflight_sha256"] == preflight[
        "preflight_sha256"
    ]
    assert publication["outputs"]["execution-preflight.json"] == hashlib.sha256(
        (tmp_path / "prepared/execution-preflight.json").read_bytes()
    ).hexdigest()
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(56)]
    published = publish_phase_capture(
        plan_path=tmp_path / "prepared/capture-plan.json",
        expected_plan_sha256=hashlib.sha256(
            (tmp_path / "prepared/capture-plan.json").read_bytes()
        ).hexdigest(),
        expected_phase="run-test", expected_capture_origin="synthetic_camera",
        video_bytes=b"synthetic-video", recipe_sequence=values["sequence"],
        decoded_rgb_frames=frames, capture_receipt_out=tmp_path / "capture.json",
        provenance_out=tmp_path / "provenance.json",
    )
    assert published.capture_receipt["video_sha256"] == values["video_sha"]


def test_prepare_rejects_alternate_receipt_and_boundary_drift(tmp_path: Path) -> None:
    from fluidmix.wp7_cohort_preparation import _prepare_run_test_inputs_for_profile

    values = _metadata_fixture(tmp_path / "source")
    arguments = {key: values[key] for key in (
        "recipe_sequence", "expected_recipe_sequence_sha256", "composition_receipt",
        "expected_composition_receipt_sha256", "final_review",
        "expected_final_review_sha256", "run_receipts",
        "expected_run_receipt_sha256s",
    )}
    arguments["run_receipts"] = (values["run_receipts"][0], values["run_receipts"][0])
    arguments["expected_run_receipt_sha256s"] = (
        values["expected_run_receipt_sha256s"][0],
        values["expected_run_receipt_sha256s"][0],
    )
    with pytest.raises(ValueError, match="order|binding|identity"):
        _prepare_run_test_inputs_for_profile(
            **arguments, destination=tmp_path / "alternate",
            profile=_synthetic_profile(values),
        )

    arguments["run_receipts"] = values["run_receipts"]
    arguments["expected_run_receipt_sha256s"] = values["expected_run_receipt_sha256s"]
    run2 = json.loads(values["run_receipts"][1].read_text())
    run2["encoded_transition_evidence"][0]["start_frame_index"] = 1
    changed_sha = _write(values["run_receipts"][1], run2)
    arguments["expected_run_receipt_sha256s"] = (
        values["expected_run_receipt_sha256s"][0], changed_sha
    )
    composition = json.loads(values["composition_receipt"].read_text())
    composition["ordered_runs"][1]["run_receipt_sha256"] = changed_sha
    arguments["expected_composition_receipt_sha256"] = _write(
        values["composition_receipt"], composition
    )
    review = json.loads(values["final_review"].read_text())
    review["identity"]["output_receipt_sha256"] = arguments[
        "expected_composition_receipt_sha256"
    ]
    arguments["expected_final_review_sha256"] = _write(values["final_review"], review)
    with pytest.raises(ValueError, match="boundary|timing"):
        _prepare_run_test_inputs_for_profile(
            **arguments, destination=tmp_path / "drift",
            profile=_synthetic_profile(values),
        )


def test_prepare_is_no_clobber_and_emits_no_terminal_artifacts(tmp_path: Path) -> None:
    from fluidmix.wp7_cohort_preparation import _prepare_run_test_inputs_for_profile

    values = _metadata_fixture(tmp_path / "source")
    arguments = {key: values[key] for key in (
        "recipe_sequence", "expected_recipe_sequence_sha256", "composition_receipt",
        "expected_composition_receipt_sha256", "final_review",
        "expected_final_review_sha256", "run_receipts",
        "expected_run_receipt_sha256s",
    )}
    destination = tmp_path / "prepared"
    profile = _synthetic_profile(values)
    _prepare_run_test_inputs_for_profile(
        **arguments, destination=destination, profile=profile
    )
    with pytest.raises(FileExistsError):
        _prepare_run_test_inputs_for_profile(
            **arguments, destination=destination, profile=profile
        )
    assert not any(
        token in path.name for path in destination.iterdir()
        for token in ("claim", "journal", "capture-receipt", "evidence-store", "one-shot")
    )


def _freeze_fixture(root: Path, monkeypatch) -> dict[str, object]:
    import fluidmix.wp7_cohort_preparation as preparation
    from fluidmix.wp7_rgb_runtime import LoadedFrozenCNN

    root.mkdir()
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    replay = root / "replay.json"
    replay_value = {
        "artifact_type": "wp5_cnn_deterministic_bounds_replay",
        "status": "swap_review_ready", "execution_device": "cpu",
        "threshold_frozen": 0.5, "checkpoint_frozen": True,
        "replayed_rows": [
            {"run_id": f"RUN-{case_index // 3 + 1}",
             "case_id": f"CASE-{case_index}:stage-{stage}"}
            for case_index, count in enumerate((3, 3, 3, 2, 2, 2, 2, 2, 2))
            for stage in range(1, count + 1)
        ],
        "determinism_gate": {"matched_rows": 21, "mismatch_count": 0,
                             "status": "pass"},
        "scoring": {"denominators": {"all_positive_rows": 21}},
        "source_bindings": {"checkpoint": {"sha256": checkpoint_sha}},
    }
    replay_sha = _write(replay, replay_value)
    review = root / "review.md"
    review.write_text(f"Outcome: **PASS**\nReplay artifact SHA-256: `{replay_sha}`\nCPU\n")
    review_sha = hashlib.sha256(review.read_bytes()).hexdigest()
    monkeypatch.setattr(preparation, "CNN_CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(preparation, "CNN_REPLAY_SHA256", replay_sha)
    monkeypatch.setattr(preparation, "CNN_REVIEW_SHA256", review_sha)
    loaded = LoadedFrozenCNN(
        object(), {"model_sha256": checkpoint_sha}, {"probability_threshold": 0.5},
        {}, checkpoint_sha, {"checkpoint": str(checkpoint), "replay": str(replay),
                             "review": str(review)},
    )
    calls = []
    monkeypatch.setattr(
        preparation, "load_frozen_cnn",
        lambda **kwargs: calls.append(kwargs) or loaded,
    )
    return {
        "checkpoint": checkpoint, "expected_checkpoint_sha256": checkpoint_sha,
        "replay": replay, "expected_replay_sha256": replay_sha,
        "review": review, "expected_review_sha256": review_sha,
        "calls": calls,
    }


def test_freeze_prep_derives_runtime_and_truthful_denominators(
    tmp_path: Path, monkeypatch
) -> None:
    from fluidmix.carryover_pipeline import _load_freeze
    from fluidmix.wp7_cohort_preparation import prepare_validation_freeze

    values = _freeze_fixture(tmp_path / "freeze-source", monkeypatch)
    result = prepare_validation_freeze(
        **{key: value for key, value in values.items() if key != "calls"},
        destination=tmp_path / "freeze-prepared",
    )
    validation = json.loads(result.validation_results_path.read_text())
    assert values["calls"] and values["calls"][0]["expected_checkpoint_sha256"] == values[
        "expected_checkpoint_sha256"
    ]
    assert validation["validation_denominators"] == {
        "run_count": 3, "vial_count": 9, "predicted_stage_count": 21,
        "measured_stage_count": 21, "history_supported_vial_count": 0,
        "explained_vial_count": 0,
    }
    assert validation["decision"]["config"] == {
        "discard_threshold_delta_e76": 2.0, "keep_threshold_delta_e76": 1.0,
    }
    assert validation["explanation"]["method"] == "deterministic_guarded_template"
    assert validation["explanation"]["model_id"] == "none"
    assert len({
        validation["measurement"]["contract_sha256"],
        validation["explanation"]["method_sha256"],
        validation["explanation"]["guard_sha256"],
        validation["store"]["schema_sha256"],
    }) == 4
    freeze = json.loads(result.freeze_path.read_text())
    assert _load_freeze(result.freeze_path, freeze["freeze_sha256"]) == freeze


@pytest.mark.parametrize("mutation", ["hash", "status", "lineage"])
def test_freeze_prep_fails_wrong_hash_status_or_runtime_lineage(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    from fluidmix.wp7_cohort_preparation import prepare_validation_freeze

    values = _freeze_fixture(tmp_path / "freeze-source", monkeypatch)
    if mutation == "hash":
        values["expected_replay_sha256"] = "0" * 64
    else:
        replay = json.loads(values["replay"].read_text())
        if mutation == "status":
            replay["status"] = "failed"
        else:
            replay["source_bindings"]["checkpoint"]["sha256"] = "f" * 64
        digest = _write(values["replay"], replay)
        values["expected_replay_sha256"] = digest
        import fluidmix.wp7_cohort_preparation as preparation
        monkeypatch.setattr(preparation, "CNN_REPLAY_SHA256", digest)
    with pytest.raises(ValueError, match="SHA|status|lineage|replay"):
        prepare_validation_freeze(
            **{key: value for key, value in values.items() if key != "calls"},
            destination=tmp_path / "never-freeze",
        )


def test_cli_exposes_only_preparation_subcommands() -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_wp7_cohort", ROOT / "scripts/prepare_wp7_cohort.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = next(action for action in module.build_parser()._actions
                    if action.dest == "command").choices
    assert set(commands) == {"prepare-run-test", "prepare-freeze"}
    all_destinations = {
        action.dest for parser in commands.values() for action in parser._actions
    }
    assert not all_destinations & {
        "video", "video5", "claim", "journal", "capture_receipt",
        "evidence_store", "one_shot_receipt",
    }
