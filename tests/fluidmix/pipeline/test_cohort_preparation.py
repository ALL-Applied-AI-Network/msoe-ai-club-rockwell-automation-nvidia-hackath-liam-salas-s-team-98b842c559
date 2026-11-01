"""Metadata-only preparation for the accepted WP7 cohort-goldens route."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def _write(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_prepare_real_check_publishes_exact_video5_profile(tmp_path: Path) -> None:
    from fluidmix.pipeline.cohort_preparation import prepare_real_check_inputs

    result = prepare_real_check_inputs(destination=tmp_path / "prepared")
    prepared = tmp_path / "prepared"
    sequence_raw = (prepared / "recipe-sequence.json").read_bytes()
    sequence = json.loads(sequence_raw)
    plan = json.loads((prepared / "capture-plan.json").read_text())
    receipt = result.publication_receipt

    expected_text_bindings = {
        "dataset_manifest": "data/dataset_manifest.csv",
        "dataset_inventory": "data/dataset_inventory.csv",
        "video5_vial_recipes": "data/video5_vial_recipes.csv",
        "video5_transitions": "data/video5_transitions.csv",
    }
    assert {
        key: value["repository_relative_path"]
        for key, value in receipt["inputs"]["sources"].items()
    } == expected_text_bindings
    assert all(
        set(value) >= {"repository_relative_path", "raw_sha256"}
        and "path" not in value
        and not Path(value["repository_relative_path"]).is_absolute()
        for value in receipt["inputs"]["sources"].values()
    )

    assert sequence["split_role"] == "real_check"
    assert sequence["split_version"] == "video5-cohort-goldens-v1"
    assert [run["run_id"] for run in sequence["runs"]] == ["VIDEO5-RUN-01"]
    vials = sequence["runs"][0]["vials"]
    assert [vial["case_id"] for vial in vials] == [
        f"VIDEO5-VIAL-{index:02d}" for index in range(1, 31)
    ]
    assert [vial["block_position"] for vial in vials] == [1, 2, 3] * 10
    assert vials[0]["previous_recipe_id"] is None
    assert vials[0]["programmed_tank_ml"] == {
        "t100": 10, "t200": 10, "t400": 10,
    }
    assert vials[3]["dispense_order"] == ["t200", "t100"]
    assert vials[-1]["programmed_tank_ml"] == {
        "t100": 60, "t200": 0, "t400": 10,
    }

    assert plan["phase"] == "run-real-check"
    assert plan["capture_origin"] == "real_camera"
    assert plan["video_sha256"] == (
        "6f87cec1aeec3b2bae6dac639a317c077a3f175f27abd5fc8430d6953d9d05b9"
    )
    assert len(plan["fill_clips"]) == 30
    assert len(plan["encoded_transition_evidence"]) == 31
    expected_fill_ranges = [
        (0, 808), (1386, 2210), (2752, 3616), (4148, 4841),
        (5373, 6094), (6630, 7393), (8540, 9242), (10105, 10830),
        (11740, 12428), (13043, 14184), (14754, 15849),
        (16463, 17597), (18292, 19692), (20386, 21780),
        (22485, 23926), (24530, 25681), (26267, 27409),
        (27973, 29098), (29685, 30929), (31511, 32768),
        (33326, 34600), (35387, 36619), (37392, 38698),
        (39495, 40797), (41534, 42711), (43540, 44620),
        (45430, 46548), (47385, 48035), (48751, 49400),
        (50138, 50815),
    ]
    expected_transition_ranges = [
        (0, 1), (808, 1387), (2210, 2753), (3616, 4149),
        (4841, 5374), (6094, 6631), (7393, 8541), (9242, 10106),
        (10830, 11741), (12428, 13044), (14184, 14755),
        (15849, 16464), (17597, 18293), (19692, 20387),
        (21780, 22486), (23926, 24531), (25681, 26268),
        (27409, 27974), (29098, 29686), (30929, 31512),
        (32768, 33327), (34600, 35388), (36619, 37393),
        (38698, 39496), (40797, 41535), (42711, 43541),
        (44620, 45431), (46548, 47386), (48035, 48752),
        (49400, 50139), (50815, 50942),
    ]
    windows = json.loads(
        (prepared / "upstream-04-transition-windows.json").read_text()
    )
    assert windows["fill_ranges"] == [list(bounds) for bounds in expected_fill_ranges]
    assert [
        (row["start_frame_index"], row["stop_frame_index"])
        for row in plan["encoded_transition_evidence"]
    ] == expected_transition_ranges
    assert [
        (row["start_frame_index"], row["stop_frame_index"])
        for row in windows["transition_ranges"]
    ] == expected_transition_ranges
    assert plan["fill_clips"][0] == {
        "case_id": "VIDEO5-VIAL-01", "frame_count": 808,
    }
    assert plan["fill_clips"][-1] == {
        "case_id": "VIDEO5-VIAL-30", "frame_count": 677,
    }
    assert plan["encoded_transition_evidence"][0] == {
        "segment_index": 0, "start_frame_index": 0, "stop_frame_index": 1,
        "frame_count": 1, "decoded_bgr_sha256": None,
    }
    assert plan["encoded_transition_evidence"][1] == {
        "segment_index": 1, "start_frame_index": 808,
        "stop_frame_index": 1387, "frame_count": 579,
        "decoded_bgr_sha256": None,
    }
    assert plan["encoded_transition_evidence"][-1] == {
        "segment_index": 30, "start_frame_index": 50815,
        "stop_frame_index": 50942, "frame_count": 127,
        "decoded_bgr_sha256": None,
    }
    assert [source["role"] for source in plan["source_provenance"]] == [
        "video5_dataset_manifest", "video5_recipe_metadata",
        "video5_timing_metadata", "video5_transition_window_metadata",
    ]
    assert receipt["inputs"]["declared_video"] == {
        "dataset_relative_path": (
            "csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_5.mp4"
        ),
        "byte_count": 32178488,
        "sha256": plan["video_sha256"],
        "frame_count": 50942,
        "fps": 30,
        "width": 640,
        "height": 480,
        "path_accessed": False,
    }
    assert receipt["inputs"]["recipe_sequence"]["raw_sha256"] == hashlib.sha256(
        sequence_raw
    ).hexdigest()
    assert receipt["inputs"]["recipe_sequence"]["canonical_sha256"] == _hash(
        sequence
    )
    assert receipt["reference_contract"] == {
        "rule": "each_fill_start_frame_is_empty_vial_before_pour_including_frame_0",
        "reference_frame_indices": [
            0, 1386, 2752, 4148, 5373, 6630, 8540, 10105, 11740,
            13043, 14754, 16463, 18292, 20386, 22485, 24530, 26267,
            27973, 29685, 31511, 33326, 35387, 37392, 39495, 41534,
            43540, 45430, 47385, 48751, 50138,
        ],
        "evidence_status": "approved_metadata_assumption_not_physical_truth",
    }
    assert receipt["runtime_capacity_preflight"] == {
        "all_frames_raw_rgb_bytes": 46948147200,
        "execution_mode": "capacity_bounded_in_memory_existing_runtime",
        "projected_peak_memory_bytes": 52 * 1024 ** 3,
        "minimum_available_ram_bytes": 80 * 1024 ** 3,
        "minimum_free_disk_bytes": 50 * 1024 ** 3,
        "measurement_timing": "immediately_before_run_real_check_claim",
        "capacity_evidence_status": (
            "requirements_bound_not_measured_during_metadata_preparation"
        ),
    }


@pytest.mark.parametrize(
    ("source_field", "hash_field", "old", "new", "message"),
    [
        (
            "dataset_manifest_path", "dataset_manifest_sha256",
            "32178488,6f87cec1", "32178488,0f87cec1", "manifest row",
        ),
        (
            "dataset_inventory_path", "dataset_inventory_sha256",
            "vial_fill,5,vial_fill/videos/vial_fill_video_5.mp4,32178488,1698.067,50942,",
            "vial_fill,5,vial_fill/videos/vial_fill_video_5.mp4,32178488,1698.067,50941,",
            "inventory row",
        ),
        (
            "recipe_csv_path", "recipe_csv_sha256",
            "1,5,Triple,10,10,10,30,", "1,5,Triple,11,10,10,30,", "volume",
        ),
        (
            "transitions_csv_path", "transitions_csv_sha256",
            "5,2,5,T200>T400>T100", "5,2,6,T200>T400>T100", "recipe row",
        ),
    ],
)
def test_real_check_source_semantic_drift_fails_closed(
    tmp_path: Path, source_field: str, hash_field: str,
    old: str, new: str, message: str,
) -> None:
    import fluidmix.pipeline.cohort_preparation as preparation

    profile = preparation._ACCEPTED_REAL_CHECK_PROFILE
    source = Path(getattr(profile, source_field))
    changed = source.read_text().replace(old, new, 1)
    assert changed != source.read_text()
    changed_path = tmp_path / source.name
    changed_path.write_text(changed)
    changed_profile = replace(
        profile,
        **{
            source_field: changed_path,
            hash_field: hashlib.sha256(changed_path.read_bytes()).hexdigest(),
        },
    )
    with pytest.raises(ValueError, match=message):
        preparation._prepare_real_check_inputs_for_profile(
            destination=tmp_path / "never", profile=changed_profile
        )


def test_real_check_preparation_never_touches_declared_media(
    tmp_path: Path, monkeypatch
) -> None:
    import fluidmix.pipeline.cohort_preparation as preparation

    methods = ("lstat", "open", "read_bytes", "stat", "resolve")
    originals = {name: getattr(Path, name) for name in methods}

    def guard(name):
        def wrapped(path, *args, **kwargs):
            if str(path).lower().endswith(".mp4"):
                pytest.fail(f"declared media reached Path.{name}")
            return originals[name](path, *args, **kwargs)
        return wrapped

    for name in methods:
        monkeypatch.setattr(Path, name, guard(name))
    result = preparation.prepare_real_check_inputs(destination=tmp_path / "prepared")
    assert result.publication_receipt["access_policy"] == "metadata_only_no_media_access"


def test_real_check_preparation_is_fresh_and_emits_no_terminal_artifacts(
    tmp_path: Path,
) -> None:
    from fluidmix.pipeline.cohort_preparation import prepare_real_check_inputs

    destination = tmp_path / "prepared"
    prepare_real_check_inputs(destination=destination)
    with pytest.raises(FileExistsError):
        prepare_real_check_inputs(destination=destination)
    assert not any(
        token in path.name
        for path in destination.iterdir()
        for token in ("claim", "journal", "capture-receipt", "evidence-store", "one-shot")
    )


@pytest.mark.parametrize(
    ("available_ram", "free_disk", "message"),
    [
        (80 * 1024 ** 3 - 1, 50 * 1024 ** 3, "available RAM"),
        (80 * 1024 ** 3, 50 * 1024 ** 3 - 1, "destination disk"),
    ],
)
def test_video5_runtime_capacity_preflight_fails_closed(
    tmp_path: Path, monkeypatch, available_ram: int, free_disk: int, message: str,
) -> None:
    import fluidmix.pipeline.cohort_preparation as preparation

    monkeypatch.setattr(preparation, "_available_memory_bytes", lambda: available_ram)
    monkeypatch.setattr(preparation, "_free_disk_bytes", lambda _path: free_disk)
    with pytest.raises(ValueError, match=message):
        preparation.require_video5_runtime_capacity(output_path=tmp_path / "receipt.json")


def test_video5_runtime_capacity_preflight_accepts_exact_minima(
    tmp_path: Path, monkeypatch,
) -> None:
    import fluidmix.pipeline.cohort_preparation as preparation

    ram = 80 * 1024 ** 3
    disk = 50 * 1024 ** 3
    monkeypatch.setattr(preparation, "_available_memory_bytes", lambda: ram)
    monkeypatch.setattr(preparation, "_free_disk_bytes", lambda _path: disk)
    assert preparation.require_video5_runtime_capacity(
        output_path=tmp_path / "receipt.json"
    )["status"] == "pass"


def test_real_check_cli_capacity_gate_runs_before_path_resolution_or_claim(
    tmp_path: Path, monkeypatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "video5_capacity_cli", ROOT / "cli/fluidmix/pipeline/run_carryover_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arguments = SimpleNamespace(
        phase="run-real-check", selected_mode="same_camera_cohort_goldens",
        development_reference_store=None, development_reference_store_sha256=None,
        out=tmp_path / "receipt.json",
    )
    parser = SimpleNamespace(parse_args=lambda _argv: arguments)
    monkeypatch.setattr(module, "build_parser", lambda: parser)
    monkeypatch.setattr(
        module, "require_video5_runtime_capacity",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("capacity stop")),
    )
    monkeypatch.setattr(
        module, "_reject_preflight_path_alias",
        lambda *_args: pytest.fail("path resolution ran before capacity gate"),
    )
    with pytest.raises(RuntimeError, match="capacity stop"):
        module.main([])


def test_real_check_sources_match_post_claim_capture_schemas(tmp_path: Path) -> None:
    from fluidmix.pipeline.cohort_preparation import prepare_real_check_inputs
    from fluidmix.pipeline.phase_capture import _validate_source_metadata

    prepare_real_check_inputs(destination=tmp_path / "prepared")
    root = tmp_path / "prepared"
    sequence = json.loads((root / "recipe-sequence.json").read_text())
    plan = json.loads((root / "capture-plan.json").read_text())
    receipt = json.loads((root / "publication-receipt.json").read_text())
    case_ids = [vial["case_id"] for vial in sequence["runs"][0]["vials"]]
    for source in plan["source_provenance"]:
        adapter = json.loads(Path(source["path"]).read_text())
        upstream = json.loads(Path(adapter["upstream_path"]).read_text())
        _validate_source_metadata(
            str(source["role"]), adapter,
            phase="run-real-check", video_sha256=plan["video_sha256"],
            recipe_sequence_sha256=_hash(sequence), case_ids=case_ids,
            capture_timing_sha256=receipt["canonical_bindings"][
                "capture_timing_sha256"
            ],
            decoded_frame_count=50942, composition_receipt_sha256=None,
            composition_run_bindings=[],
        )
        assert upstream["accepted_metadata_commit"] == (
            "c9a4a86e551f5d1b18676793cbdff04bcf6d38b8"
        )


@pytest.mark.parametrize("mutation", ["path", "hash"])
def test_post_claim_video5_source_binding_drift_fails(
    tmp_path: Path, mutation: str,
) -> None:
    from fluidmix.pipeline.cohort_preparation import prepare_real_check_inputs
    from fluidmix.pipeline.phase_capture import PhaseCaptureError, _validate_source_metadata

    prepare_real_check_inputs(destination=tmp_path / "prepared")
    root = tmp_path / "prepared"
    sequence = json.loads((root / "recipe-sequence.json").read_text())
    plan = json.loads((root / "capture-plan.json").read_text())
    receipt = json.loads((root / "publication-receipt.json").read_text())
    source = plan["source_provenance"][0]
    adapter_path = Path(source["path"])
    adapter = json.loads(adapter_path.read_text())
    upstream_path = Path(adapter["upstream_path"])
    upstream = json.loads(upstream_path.read_text())
    binding = upstream["source_binding"]
    binding[
        "repository_relative_path" if mutation == "path" else "raw_sha256"
    ] = (
        "data/other.csv" if mutation == "path" else "0" * 64
    )
    adapter["upstream_sha256"] = _write(upstream_path, upstream)
    case_ids = [vial["case_id"] for vial in sequence["runs"][0]["vials"]]
    with pytest.raises(PhaseCaptureError, match="source binding"):
        _validate_source_metadata(
            str(source["role"]), adapter,
            phase="run-real-check", video_sha256=plan["video_sha256"],
            recipe_sequence_sha256=_hash(sequence), case_ids=case_ids,
            capture_timing_sha256=receipt["canonical_bindings"][
                "capture_timing_sha256"
            ],
            decoded_frame_count=50942, composition_receipt_sha256=None,
            composition_run_bindings=[],
        )


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
    from fluidmix.pipeline.cohort_preparation import _AcceptedRunTestProfile

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
    import fluidmix.pipeline.cohort_preparation as preparation

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
    from fluidmix.pipeline.cohort_preparation import _prepare_run_test_inputs_for_profile

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
    from fluidmix.pipeline.cohort_preparation import _prepare_run_test_inputs_for_profile

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
    from fluidmix.pipeline.cohort_preparation import _prepare_run_test_inputs_for_profile

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
    from fluidmix.pipeline.cohort_preparation import _prepare_run_test_inputs_for_profile
    from fluidmix.pipeline.phase_capture import publish_phase_capture

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
    from fluidmix.pipeline.cohort_preparation import _prepare_run_test_inputs_for_profile

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
    from fluidmix.pipeline.cohort_preparation import _prepare_run_test_inputs_for_profile

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
    import fluidmix.pipeline.cohort_preparation as preparation
    from fluidmix.pipeline.rgb_runtime import LoadedFrozenCNN

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
    from fluidmix.pipeline.carryover_pipeline import _load_freeze
    from fluidmix.pipeline.cohort_preparation import prepare_validation_freeze

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
    from fluidmix.pipeline.cohort_preparation import prepare_validation_freeze

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
        import fluidmix.pipeline.cohort_preparation as preparation
        monkeypatch.setattr(preparation, "CNN_REPLAY_SHA256", digest)
    with pytest.raises(ValueError, match="SHA|status|lineage|replay"):
        prepare_validation_freeze(
            **{key: value for key, value in values.items() if key != "calls"},
            destination=tmp_path / "never-freeze",
        )


def test_cli_exposes_only_preparation_subcommands() -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_wp7_cohort", ROOT / "cli/fluidmix/pipeline/prepare_wp7_cohort.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = next(action for action in module.build_parser()._actions
                    if action.dest == "command").choices
    assert set(commands) == {
        "prepare-run-test", "prepare-real-check", "prepare-freeze",
    }
    real_check = commands["prepare-real-check"]
    assert {action.dest for action in real_check._actions} == {"help", "destination"}
    all_destinations = {
        action.dest for parser in commands.values() for action in parser._actions
    }
    assert not all_destinations & {
        "video", "video5", "claim", "journal", "capture_receipt",
        "evidence_store", "one_shot_receipt",
    }
