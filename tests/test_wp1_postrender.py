from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.full_transition_manifest import (  # noqa: E402
    build_default_full_transition_source_spec,
    build_full_transition_artifacts,
)
from fluidmix.wp1_postrender import (  # noqa: E402
    canonical_sha256,
    decide_wp1_visibility,
    file_sha256,
    package_wp1_test_videos,
    validate_wp1_acceptance,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_png(path: Path, rgb: tuple[int, int, int], *, mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask:
        pixels = np.zeros((8, 8), dtype=np.uint8)
        pixels[2:7, 2:7] = 255
        Image.fromarray(pixels, mode="L").save(path)
    else:
        pixels = np.zeros((8, 8, 3), dtype=np.uint8)
        pixels[:, :] = rgb
        Image.fromarray(pixels, mode="RGB").save(path)


def _manifest_bundle(root: Path) -> dict[str, object]:
    artifacts = build_full_transition_artifacts(
        build_default_full_transition_source_spec(repo_root=ROOT), repo_root=ROOT
    )
    for relative, value in artifacts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            _write_json(path, value)
    return artifacts


def _minimal_case(
    case_root: Path,
    *,
    case_id: str,
    schedule: dict[str, object],
    model_row: dict[str, object],
    render_plan: dict[str, object],
    plan_path: Path,
    runtime: dict[str, str],
    colours: list[tuple[int, int, int]] | None = None,
) -> None:
    app = case_root / "app-output"
    app.mkdir(parents=True)
    video = app / "fill.mp4"
    video.write_bytes((case_id + "-accepted-fill").encode())
    pairs = []
    stage_paths = []
    stage_hashes = []
    stages = schedule["physical_sequence"]["dispense_order"]
    mask = app / "masks" / "liquid.png"
    _write_png(mask, (0, 0, 0), mask=True)
    for index, tank in enumerate(stages):
        rgb = app / "stages" / f"stage-{index + 1}-{tank}.png"
        colour = (colours or [(70 + index * 20, 80, 90) for index in range(3)])[index]
        _write_png(rgb, colour)
        relative = rgb.relative_to(app).as_posix()
        stage_paths.append(relative)
        stage_hashes.append(file_sha256(rgb))
        pairs.append(
            {
                "physics_step": 100 + index,
                "phase": f"{tank}-stage",
                "visible_stream_tanks": [],
                "liquid_representation": "vial_proxy",
                "paired_capture_state_passed": True,
                "rgb_path": relative,
                "rgb_sha256": file_sha256(rgb),
                "mask_path": mask.relative_to(app).as_posix(),
                "mask_sha256": file_sha256(mask),
            }
        )
    receipt = {
        "status": "pass",
        "errors": [],
        "warnings": [],
        "arguments": {
            "video_fps": 10.0,
            "train_render_plan_sha256": canonical_sha256(render_plan),
            "train_case_id": case_id,
            "accepted_qualification_receipt_sha256": runtime["qualification_receipt_sha256"],
            "nuisance_receipt_sha256": runtime["nuisance_receipt_sha256"],
        },
        "versions": {"isaac": runtime["isaac_version"], "kit": runtime["kit_version"]},
        "provenance": {
            "requested_stage_sha256": runtime["stage_sha256"],
            "script_sha256": runtime["renderer_script_sha256"],
        },
        "render": {
            "video_path": "fill.mp4",
            "video_sha256": file_sha256(video),
            "video_decodable": True,
            "video_renderer": "RaytracedLighting",
            "stage_still_renderer": "PathTracing",
            "stage_frames_are_video_subsets": False,
            "stage_frame_paths": stage_paths,
            "stage_frame_sha256": stage_hashes,
            "stage_frame_physics_steps": [
                pour["stage_capture_step"] for pour in schedule["pours"]
            ],
        },
        "semantics": {"semantic_gate_passed": True, "pairs": pairs},
        "tank_colour_resolution": deepcopy(render_plan["tank_colour_resolution"]),
        "execution_schedule": {
            "case_id": case_id,
            "cohort_plan_sha256": canonical_sha256(render_plan),
            "schedule_file_sha256": file_sha256(plan_path),
            "source_run_spec_sha256": render_plan["source_run_spec_sha256"],
            "schedule_sha256": render_plan["execution_schedule_sha256_by_case"][case_id],
            "vial_record_sha256": canonical_sha256(model_row),
        },
    }
    _write_json(app / "receipt.json", receipt)
    result = {
        "case_id": case_id,
        "outcome": "passed",
        "return_code": 0,
        "timed_out": False,
        "evidence": {
            "failures": [],
            "launch_failures": [],
            "receipt_failures": [],
            "artifact_validation": {"status": "pass", "failures": [], "warnings": []},
        },
    }
    _write_json(case_root / "case-result.json", result)
    _write_json(case_root / "launch-receipt.json", {"case_id": case_id})
    (case_root / "exit-code.txt").write_text("0\n", encoding="utf-8")


def _runtime(host: str) -> dict[str, str]:
    return {
        "execution_host": host,
        "gpu_model": "NVIDIA L40S",
        "isaac_version": "6.2.0.0",
        "kit_version": "110.4.0",
        "stage_sha256": "1" * 64,
        "renderer_script_sha256": "2" * 64,
        "qualification_receipt_sha256": "3" * 64,
        "nuisance_receipt_sha256": "4" * 64,
        "python_runtime_sha256": "5" * 64,
        "source_recipe_manifest_sha256": "49ab190bf018f61c868f48ddb83dd96d7e1d0ff77080e68d62953d8f8f9ef49b",
        "source_split_spec_sha256": "e6e4b3b8d2c0932629ffae5872a58df9f4fb1b03612b0e842e53339a4d071bfb",
    }


def _write_transport(
    root: Path,
    *,
    binding: dict[str, object],
    fill_sha_by_case: dict[str, str],
) -> Path:
    run_id = binding["run_id"]
    case_ids = binding["case_ids"]
    products = root / run_id
    products.mkdir(parents=True)
    segments = []
    for index in range(10):
        path = products / f"segment-{index:02d}.mp4"
        path.write_bytes(f"{run_id}-segment-{index}".encode())
        segments.append(
            {
                "ordinal": index + 1,
                "kind": "entry" if index == 0 else "exit" if index == 9 else "between_fill",
                "product_path": str(path),
                "product_sha256": file_sha256(path),
                "outgoing_fully_left_frame": True,
                "incoming_center_error_px": 2.0,
                "background_max_abs_delta": 2,
                "seam_mean_abs_difference": 1.0,
                "fill_interframe_p99": 1.0,
                "passed": True,
            }
        )
    vial_products = []
    for ordinal, case_id in enumerate(case_ids, start=1):
        path = products / f"vial-{ordinal:02d}.mp4"
        path.write_bytes(f"{run_id}-vial-{ordinal}".encode())
        vial_products.append(
            {
                "vial_ordinal": ordinal,
                "case_id": case_id,
                "source_fill_sha256": fill_sha_by_case[case_id],
                "product_path": str(path),
                "product_sha256": file_sha256(path),
            }
        )
    run_video = products / "run.mp4"
    run_video.write_bytes(f"{run_id}-run".encode())
    receipt = {
        "schema_version": 1,
        "artifact_type": "fluidmix-wp1-transport-acceptance-v1",
        "status": "pass",
        "run_id": run_id,
        "case_ids": case_ids,
        "segments": segments,
        "vial_products": vial_products,
        "run_product": {"path": str(run_video), "sha256": file_sha256(run_video)},
        "camera_transform_changed": False,
        "qualified_fill_receipts_changed": False,
        "review_status": "accepted",
    }
    path = root / f"{run_id}.transport.json"
    _write_json(path, receipt)
    return path


@pytest.fixture
def accepted_fixture(tmp_path: Path) -> dict[str, object]:
    manifest_root = tmp_path / "manifests"
    artifacts = _manifest_bundle(manifest_root)
    collection = artifacts["full-transition-manifest.json"]
    run_roots = []
    fill_sha: dict[str, str] = {}
    runtimes = [_runtime("l40s-host-a"), _runtime("l40s-host-b")]
    for index, binding in enumerate(collection["run_bindings"], start=1):
        run_root = tmp_path / f"merged-{index:02d}"
        run_roots.append(run_root)
        plan_path = manifest_root / binding["render_plan_path"]
        plan = artifacts[binding["render_plan_path"]]
        model = artifacts[binding["model_input_plan_path"]]
        model_by_case = {row["case_id"]: row for row in model["vial_records"]}
        schedules = plan["execution_schedules"]
        for schedule in schedules:
            case_id = schedule["case_binding"]["case_id"]
            _minimal_case(
                run_root / "cases" / case_id,
                case_id=case_id,
                schedule=schedule,
                model_row=model_by_case[case_id],
                render_plan=plan,
                plan_path=plan_path,
                runtime=runtimes[index - 1],
                colours=(
                    [(180, 60, 60), (90, 100, 110), (100, 110, 120)]
                    if case_id == "CASE_d52026e7df33b0cb"
                    else None
                ),
            )
            fill_sha[case_id] = file_sha256(
                run_root / "cases" / case_id / "app-output" / "fill.mp4"
            )
        _write_json(
            run_root / "shard-merge.json",
            {
                "schema_version": 1,
                "artifact_type": "physx_train_residual_shard_merge",
                "status": "pass",
                "plan_sha256": canonical_sha256(plan),
                "merged_case_count": 9,
                "case_ids": binding["case_ids"],
                "source_shards": [{"execution_host": runtimes[index - 1]["execution_host"]}],
                "full_resume_summary": {
                    "status": "pass",
                    "planned_case_count": 9,
                    "validated_completed_case_count": 9,
                    "case_results": [
                        {
                            "case_id": case_id,
                            "outcome": "recognized_completed",
                            "evidence": {
                                "artifact_validation": {
                                    "status": "pass", "failures": [], "warnings": []
                                }
                            },
                        }
                        for case_id in binding["case_ids"]
                    ],
                },
            },
        )
        _write_json(
            run_root / "runtime-profile.json",
            {
                "schema_version": 1,
                "artifact_type": "fluidmix-wp1-runtime-profile-v1",
                "status": "pass",
                **runtimes[index - 1],
                "plan_canonical_sha256": canonical_sha256(plan),
                "plan_file_sha256": file_sha256(plan_path),
                "case_receipt_sha256": {
                    case_id: file_sha256(
                        run_root / "cases" / case_id / "app-output" / "receipt.json"
                    )
                    for case_id in binding["case_ids"]
                },
            },
        )

    leakage = manifest_root / "leakage-check-receipt.json"
    _write_json(
        leakage,
        {
            "artifact_type": "fluidmix-wp1-leakage-check-v1",
            "status": "pass",
            "split_version": "inn-7-splits-v3",
            "case_ids": [case for b in collection["run_bindings"] for case in b["case_ids"]],
            "recipe_ids": [recipe for b in collection["run_bindings"] for recipe in b["recipe_ids"]],
            "near_video5_count": 0,
            "leakage_failure_count": 0,
            "leakage_manifest_sha256": file_sha256(manifest_root / "leakage-manifest.csv"),
        },
    )
    boundary = manifest_root / "input-boundary-receipt.json"
    _write_json(
        boundary,
        {
            "artifact_type": "fluidmix-wp1-input-boundary-check-v1",
            "status": "pass",
            "forbidden_field_count": 0,
            "model_plan_canonical_sha256": [
                b["model_input_plan_canonical_sha256"] for b in collection["run_bindings"]
            ],
        },
    )
    transport_root = tmp_path / "transport"
    transport_paths = [
        _write_transport(transport_root, binding=binding, fill_sha_by_case=fill_sha)
        for binding in collection["run_bindings"]
    ]
    review_video = tmp_path / "review.mp4"
    review_video.write_bytes(b"lead-reviewed-all-runs")
    review = tmp_path / "review.json"
    _write_json(
        review,
        {
            "artifact_type": "fluidmix-wp1-review-acceptance-v1",
            "status": "accepted",
            "reviewer_role": "lead",
            "all_runs_viewed": True,
            "visually_distinct_presets": True,
            "review_video_path": str(review_video),
            "review_video_sha256": file_sha256(review_video),
            "run_product_sha256": [
                json.loads(path.read_text())["run_product"]["sha256"]
                for path in transport_paths
            ],
        },
    )
    return {
        "manifest_root": manifest_root,
        "run_roots": run_roots,
        "transport_paths": transport_paths,
        "review": review,
        "collection": collection,
    }


def test_visibility_decision_uses_all_three_receipt_bound_stages_and_keeps_plan_hashes_distinct(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    root = accepted_fixture["manifest_root"]
    run_root = accepted_fixture["run_roots"][0]
    binding = accepted_fixture["collection"]["run_bindings"][0]
    plan = json.loads((root / binding["render_plan_path"]).read_text())
    by_case = {row["case_binding"]["case_id"]: row for row in plan["execution_schedules"]}
    model = json.loads((root / binding["model_input_plan_path"]).read_text())
    model_by_case = {row["case_id"]: row for row in model["vial_records"]}
    for case_id, colours in (
        ("CASE_d52026e7df33b0cb", [(180, 60, 60), (90, 100, 110), (100, 110, 120)]),
        ("CASE_ab6449c995c4b07d", [(60, 60, 60), (90, 100, 110), (100, 110, 120)]),
    ):
        shutil.rmtree(run_root / "cases" / case_id)
        _minimal_case(
            run_root / "cases" / case_id,
            case_id=case_id,
            schedule=by_case[case_id],
            model_row=model_by_case[case_id],
            render_plan=plan,
            plan_path=root / binding["render_plan_path"],
            runtime=_runtime("l40s-host-a"),
            colours=colours,
        )
    output = tmp_path / "visibility.json"
    decision = decide_wp1_visibility(
        manifest_root=root, run_root=run_root, output_path=output
    )
    assert decision["status"] == "pass"
    assert decision["decision"] == "original_1_5ml_visible"
    assert [row["stage"] for row in decision["stage_measurements"]] == ["t200", "t400", "t100"]
    assert decision["maximum_delta_e76"] >= 2.0
    assert decision["plan_binding"]["canonical_sha256"] == canonical_sha256(plan)
    assert decision["plan_binding"]["file_sha256"] == file_sha256(root / binding["render_plan_path"])
    assert decision["decision_sha256"] == canonical_sha256(
        {key: value for key, value in decision.items() if key != "decision_sha256"}
    )


def test_visibility_rejects_malformed_or_mismatched_masks(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    root = accepted_fixture["manifest_root"]
    run_root = accepted_fixture["run_roots"][0]
    target = run_root / "cases" / "CASE_d52026e7df33b0cb" / "app-output"
    receipt_path = target / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    bad_mask = target / "masks" / "bad.png"
    Image.fromarray(np.full((7, 8), 1, dtype=np.uint8), mode="L").save(bad_mask)
    receipt["semantics"]["pairs"][0]["mask_path"] = "masks/bad.png"
    receipt["semantics"]["pairs"][0]["mask_sha256"] = file_sha256(bad_mask)
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="mask"):
        decide_wp1_visibility(
            manifest_root=root, run_root=run_root, output_path=tmp_path / "never.json"
        )


def test_aggregate_acceptance_and_package_exact_wp1_names(
    accepted_fixture: dict[str, object], tmp_path: Path
) -> None:
    acceptance_path = tmp_path / "wp1-acceptance.json"
    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=accepted_fixture["run_roots"],
        transport_receipts=accepted_fixture["transport_paths"],
        review_receipt=accepted_fixture["review"],
        output_path=acceptance_path,
        repo_root=ROOT,
    )
    assert result["status"] == "pass"
    assert result["counts"] == {"runs": 2, "cases": 18, "recipes": 6, "keep": 9, "discard": 9}
    assert result["runtime"]["gpu_model"] == "NVIDIA L40S"
    assert result["runtime"]["hosts"] == ["l40s-host-a", "l40s-host-b"]
    assert result["transport"]["segment_count"] == 20

    package = tmp_path / "package"
    receipt = package_wp1_test_videos(
        acceptance_path=acceptance_path,
        output_root=package,
    )
    expected_runs = {
        "test-run01-S0439-S0440-S0441-amber_teal.mp4",
        "test-run02-S0442-S0443-S0444-crimson_indigo.mp4",
    }
    assert {path.name for path in (package / "videos" / "test").glob("test-run??-*.mp4") if "-vial" not in path.name} == expected_runs
    assert len(list((package / "videos" / "test").glob("test-run??-vial??-*.mp4"))) == 18
    assert (package / "videos" / "test" / "INDEX.csv").is_file()
    assert (package / "videos" / "test" / "README.md").is_file()
    assert receipt["status"] == "pass"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("mixed_gpu", "runtime_profile_mismatch"),
        ("mixed_runtime_hash", "runtime_profile_mismatch"),
        ("mixed_source_hash", "runtime_profile_mismatch"),
        ("missing_case", "case_set"),
        ("mismatched_video", "source_fill"),
        ("relabelled", "label"),
        ("failed_transport", "transport"),
        ("absent_transport", "transport"),
    ],
)
def test_acceptance_fails_closed_for_adversarial_evidence(
    accepted_fixture: dict[str, object], tmp_path: Path, mutation: str, expected: str
) -> None:
    roots = list(accepted_fixture["run_roots"])
    transports = list(accepted_fixture["transport_paths"])
    if mutation in {"mixed_gpu", "mixed_runtime_hash", "mixed_source_hash"}:
        path = roots[1] / "runtime-profile.json"
        value = json.loads(path.read_text())
        field = {
            "mixed_gpu": "gpu_model",
            "mixed_runtime_hash": "renderer_script_sha256",
            "mixed_source_hash": "source_recipe_manifest_sha256",
        }[mutation]
        value[field] = "NVIDIA GB10" if field == "gpu_model" else "f" * 64
        _write_json(path, value)
    elif mutation == "missing_case":
        shutil.rmtree(roots[0] / "cases" / accepted_fixture["collection"]["run_bindings"][0]["case_ids"][0])
    elif mutation == "mismatched_video":
        receipt = json.loads(transports[0].read_text())
        receipt["vial_products"][0]["source_fill_sha256"] = "f" * 64
        _write_json(transports[0], receipt)
    elif mutation == "relabelled":
        label = accepted_fixture["manifest_root"] / "run-01" / "label-provenance.json"
        value = json.loads(label.read_text())
        value["vial_labels"][0]["expected_disposition"] = "discard"
        _write_json(label, value)
    elif mutation == "failed_transport":
        receipt = json.loads(transports[0].read_text())
        receipt["segments"][0]["passed"] = False
        _write_json(transports[0], receipt)
    elif mutation == "absent_transport":
        transports.pop()

    result = validate_wp1_acceptance(
        manifest_root=accepted_fixture["manifest_root"],
        run_roots=roots,
        transport_receipts=transports,
        review_receipt=accepted_fixture["review"],
        output_path=tmp_path / f"{mutation}-wp1-acceptance.json",
        repo_root=ROOT,
    )
    assert result["status"] == "failed"
    assert any(expected in failure for failure in result["failures"])
