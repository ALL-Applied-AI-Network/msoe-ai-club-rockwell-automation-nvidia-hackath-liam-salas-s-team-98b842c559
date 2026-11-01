"""CPU contracts for the frozen Brev validation-video cohort."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_BATCHES = (
    "BATCH_72ADF5DC9C45",
    "BATCH_A51288D2AE92",
    "BATCH_907E15C8E9B1",
    "BATCH_D7A0146E9E09",
    "BATCH_F77F6F81E31A",
    "BATCH_D5944B023021",
    "BATCH_A583FC4BC148",
    "BATCH_E4033277D748",
)


def _write_frozen_plan(path: Path) -> None:
    from fluidmix.memory.history.vision_dataset_plan import CONFIG_PATH, build_vision_artifacts

    payload = build_vision_artifacts(CONFIG_PATH, repo_root=ROOT)["render-plan.json"]
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_twin(root: Path) -> None:
    from tests.twin.scene.test_twin_stage_appearance import _write_twin

    _write_twin(root)
    camera = root / "cameras/FillCam.json"
    camera.parent.mkdir(parents=True)
    camera.write_text('{"render_px":[640,480]}\n', encoding="utf-8")


def _materialize(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    from twin.render.brev_video_cohort import materialize_brev_video_cohort

    plan = tmp_path / "render-plan.json"
    _write_frozen_plan(plan)
    twin = tmp_path / "twin"
    _write_twin(twin)
    runtime_lock = tmp_path / "runtime-lock.txt"
    runtime_lock.write_text("isaacsim==6.0.1.0\n", encoding="utf-8")
    output = tmp_path / "cohort"
    graph = materialize_brev_video_cohort(
        render_plan_path=plan,
        twin_root=twin,
        output_dir=output,
        runtime_lock=runtime_lock,
        repo_root=ROOT,
        implementation_commit="1" * 40,
        isaac_python="/home/ubuntu/isaac/.venv/bin/python",
    )
    return output, graph


def test_materializer_emits_exact_validation_video_graph_layers_and_manifests(tmp_path):
    from twin.render.brev_video_cohort import (
        SOURCE_RENDER_PLAN_SHA256,
        STAGE_TIME_CODES,
        VIDEO_TIME_CODES,
    )
    from twin.render.render_process import load_experiment_manifest
    from twin.scene.twin_stage_appearance import inspect_stage_appearance_layer

    output, graph = _materialize(tmp_path)

    assert len(VIDEO_TIME_CODES) == 133
    assert tuple(VIDEO_TIME_CODES[index - 1] for index in (29, 64, 133)) == STAGE_TIME_CODES
    assert graph["source_render_plan_sha256"] == SOURCE_RENDER_PLAN_SHA256
    assert graph["family"] == {
        "family_id": "VA",
        "partition": "val",
        "previous_sim_recipe_id": "S0160",
        "current_sim_recipe_id": "S0380",
        "synthetic_onset_id": "after_t400",
    }
    assert graph["render_runtime"] == {
        "provider": "Brev",
        "instance_name": "claude-isaac-smoke-1024",
        "instance_id": "m1vgz4tbs",
        "machine_type": "g6e.xlarge",
        "gpu": "L40S",
        "machine": "x86_64",
        "runtime_name": "NVIDIA Isaac Sim",
        "isaac_sim_version": "6.0.1.0",
        "environment_profile": "generic",
        "isaac_python": "/home/ubuntu/isaac/.venv/bin/python",
    }
    assert graph["denominators"] == {
        "batches": 8,
        "vials_and_render_launches": 24,
        "frames_per_launch": 133,
        "total_frames": 3192,
        "stage_frames_per_launch": 3,
        "stage_frames_total": 72,
        "videos": 24,
        "maximum_process_attempts_per_launch": 1,
    }
    assert graph["render_controls"]["time_codes"] == list(VIDEO_TIME_CODES)
    assert graph["render_controls"]["stage_time_codes"] == list(STAGE_TIME_CODES)
    assert graph["video_encoding"] == {
        "codec_request": "mp4v",
        "container": "mp4",
        "fps": 6.0,
        "expected_decoded_frames": 133,
        "timing": "constant_fps_with_inserted_time_code_342_not_source_timing_exact",
        "use": "lossy_review_visualization_only",
        "measurement_source": "hash_bound_capture_png_and_liquid_mask_only",
    }
    rows = graph["launches"]
    assert len(rows) == 24
    assert tuple(dict.fromkeys(row["batch_id"] for row in rows)) == EXPECTED_BATCHES
    assert [row["launch_id"] for row in rows[:3]] == [
        "BATCH_72ADF5DC9C45-V1",
        "BATCH_72ADF5DC9C45-V2",
        "BATCH_72ADF5DC9C45-V3",
    ]
    assert [row["seed"] for row in rows[:3]] == [1310001, 1310002, 1310003]
    assert all(row["max_process_attempts"] == 1 for row in rows)
    assert all(row["replacement_attempts"] == 0 for row in rows)
    for row in rows:
        layer = inspect_stage_appearance_layer(row["appearance_layer"], tmp_path / "twin")
        assert [sample["time_code"] for sample in layer["samples"]] == list(STAGE_TIME_CODES)
        manifest = load_experiment_manifest(row["experiment_manifest"], row["launch_id"])["payload"]
        assert manifest["seed"] == row["seed"]
        assert manifest["denominators"]["requested_frames"] == 133
        assert manifest["denominators"]["maximum_process_attempts"] == 1
        assert manifest["command_plan"]["appearance_layer"]["stage_sha256"] == layer["stage_sha256"]
        assert manifest["render_runtime"]["isaac_sim_version"] == "6.0.1.0"
        assert manifest["render_runtime"]["environment_profile"] == "generic"
    assert json.loads((output / "cohort.json").read_text(encoding="utf-8")) == graph


def test_materializer_rejects_plan_hash_drift_before_creating_output(tmp_path):
    from twin.render.brev_video_cohort import materialize_brev_video_cohort

    plan = tmp_path / "render-plan.json"
    _write_frozen_plan(plan)
    plan.write_bytes(plan.read_bytes() + b" ")
    twin = tmp_path / "twin"
    _write_twin(twin)
    runtime_lock = tmp_path / "runtime-lock.txt"
    runtime_lock.write_text("isaacsim==6.0.1.0\n", encoding="utf-8")
    output = tmp_path / "cohort"

    with pytest.raises(ValueError, match="render plan SHA-256"):
        materialize_brev_video_cohort(
            render_plan_path=plan,
            twin_root=twin,
            output_dir=output,
            runtime_lock=runtime_lock,
            repo_root=ROOT,
            implementation_commit="1" * 40,
            isaac_python="/home/ubuntu/isaac/.venv/bin/python",
        )

    assert not output.exists()


def test_runner_stops_after_first_failed_render_without_replacement(tmp_path):
    from twin.render.brev_video_cohort import run_brev_video_cohort

    output, graph = _materialize(tmp_path)
    calls = tmp_path / "calls.txt"
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        """import argparse,json,pathlib,sys
p=argparse.ArgumentParser()
p.add_argument('--run-id'); p.add_argument('--out-parent'); p.add_argument('--seed')
args,_=p.parse_known_args()
calls=pathlib.Path(__CALLS__); calls.open('a').write('render '+args.run_id+'\\n')
if args.run_id.endswith('-V2'): raise SystemExit(7)
root=pathlib.Path(args.out_parent)/args.run_id; (root/'capture').mkdir(parents=True)
(root/'attempt.json').write_text(json.dumps({'status':'mechanical_gates_passed','failure_reasons':[],'capture':{'verified':True,'root':str(root/'capture')}})+'\\n')
""".replace("__CALLS__", repr(str(calls))),
        encoding="utf-8",
    )
    encoder = tmp_path / "encoder.py"
    encoder.write_text(
        """import argparse,hashlib,json,pathlib,sys
p=argparse.ArgumentParser(); p.add_argument('--capture-root'); p.add_argument('--output-dir')
args=p.parse_args(); calls=pathlib.Path(__CALLS__); calls.open('a').write('encode '+pathlib.Path(args.capture_root).parent.name+'\\n')
out=pathlib.Path(args.output_dir); out.mkdir(parents=True); video=out/'sequence.mp4'; video.write_bytes(b'video')
(out/'video.json').write_text(json.dumps({'status':'encoded_verified','capture':{'root':str(pathlib.Path(args.capture_root).resolve())},'video':{'path':'sequence.mp4','sha256':hashlib.sha256(b'video').hexdigest(),'bytes':5}})+'\\n')
""".replace("__CALLS__", repr(str(calls))),
        encoding="utf-8",
    )
    controller = tmp_path / "controller-python"
    controller.symlink_to(sys.executable)

    result = run_brev_video_cohort(
        output / "cohort.json",
        controller_python=controller,
        launcher_script=launcher,
        encoder_script=encoder,
    )

    assert result["status"] == "failed"
    assert result["attempted_launches"] == 2
    assert result["completed_launches"] == 1
    assert result["failed_launch_id"] == graph["launches"][1]["launch_id"]
    assert [line for line in calls.read_text().splitlines()] == [
        f"render {graph['launches'][0]['launch_id']}",
        f"encode {graph['launches'][0]['launch_id']}",
        f"render {graph['launches'][1]['launch_id']}",
    ]
    assert len(result["rows"]) == 2
    assert result["rows"][0]["render"]["command"][0] == str(controller)
    assert result["rows"][0]["outcome"] == "passed"
    assert result["rows"][1]["outcome"] == "render_failed"
    assert json.loads((output / "cohort-run.json").read_text()) == result
    assert not (output / graph["launches"][2]["launch_id"]).exists()
