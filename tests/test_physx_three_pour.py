"""CPU contracts for the bounded F2 one-system three-pour experiment."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fluidmix.physx_three_pour import (
    build_three_pour_plan,
    distribute_particle_count,
    validate_three_pour_receipt,
)
import fluidmix.physx_three_pour as physx_three_pour


SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "physx_three_pour_sequence", ROOT / "scripts" / "physx_three_pour_sequence.py"
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
sequence = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(sequence)


def test_frozen_plan_has_exact_recipe_order_counts_and_nominal_volume_limit():
    plan = build_three_pour_plan()

    assert plan["schema_version"] == 1
    assert plan["experiment"] == "physx_three_pour_f2"
    assert plan["seed"] == 20261025
    assert plan["fluid_rest_offset_m"] == 0.0015
    assert plan["nominal_particle_volume_ml"] == pytest.approx(0.027)
    assert [row["tank_id"] for row in plan["pours"]] == ["t200", "t400", "t100"]
    assert [row["programmed_volume_ml"] for row in plan["pours"]] == [10, 10, 40]
    assert [row["particle_count"] for row in plan["pours"]] == [370, 370, 1481]
    assert [row["emission_chunk_count"] for row in plan["pours"]] == [20, 20, 48]
    assert plan["totals"] == {
        "programmed_volume_ml": 60,
        "particle_count": 2221,
        "nominal_mass_convention_equivalent_ml": pytest.approx(59.967),
        "nominal_error_ml": pytest.approx(-0.033),
    }
    assert plan["mixing_assumption"] == (
        "fully_mixed_after_each_settled_pour_uncalibrated_render_parameter"
    )
    assert plan["claims"]["physically_displaced_volume"] is False
    assert plan["claims"]["absorbance_fitted"] is False


@pytest.mark.parametrize("total,chunks", [(370, 20), (1481, 48), (1, 1)])
def test_emission_chunk_distribution_is_positive_ordered_and_exact(total, chunks):
    counts = distribute_particle_count(total, chunks)

    assert len(counts) == chunks
    assert sum(counts) == total
    assert all(type(value) is int and value > 0 for value in counts)
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("total,chunks", [(0, 1), (1, 0), (3, 4), (True, 1), (1, False)])
def test_emission_chunk_distribution_rejects_invalid_or_empty_chunks(total, chunks):
    with pytest.raises((TypeError, ValueError)):
        distribute_particle_count(total, chunks)


def _passing_receipt() -> dict[str, object]:
    plan = build_three_pour_plan()
    stages = []
    cumulative = 0
    frame_hashes = []
    for ordinal, pour in enumerate(plan["pours"], start=1):
        cumulative += pour["particle_count"]
        digest = f"{ordinal:064x}"
        frame_hashes.append(digest)
        stages.append(
            {
                "ordinal": ordinal,
                "tank_id": pour["tank_id"],
                "programmed_volume_ml": pour["programmed_volume_ml"],
                "emitted_particle_count": pour["particle_count"],
                "cumulative_particle_count": cumulative,
                "finite_positions": True,
                "escaped_particle_count": 0,
                "stable": True,
                "stable_frame_index": ordinal * 10,
                "stable_frame_sha256": digest,
                "isosurface_nonempty": True,
                "used_gpu_memory_mib_before": 1000,
                "used_gpu_memory_mib_after": 1100,
                "memory_recovered_during_settle": True,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "physx_three_pour_f2",
        "status": "pass",
        "versions": {
            "isaac": "6.0.1.0",
            "kit": "110.1.2",
            "physx": "110.1.13",
        },
        "gpu_simulation": {
            "cuda_device_available": True,
            "scene_enable_gpu_dynamics": True,
            "broadphase": "GPU",
        },
        "particle_system": {
            "count": 1,
            "exactly_one": True,
            "mass_api_count": 0,
            "all_particle_sets_target_one_system": True,
            "collider_types": ["Cube"],
            "pbd_material_binding": True,
            "render_material_binding": True,
        },
        "render_guides": {
            "expected_count": 8,
            "authored_count": 8,
            "collision_api_count": 0,
            "paths": [f"/World/RenderGuides/Guide{index}" for index in range(8)],
        },
        "camera": {
            "path": "/World/Camera",
            "clipping_range_requested": [0.01, 10000.0],
            "clipping_range_authored": [0.01, 10000.0],
            "clipping_range_effective": [0.01, 10000.0],
        },
        "pours": stages,
        "totals": {
            "programmed_volume_ml": 60,
            "emitted_particle_count": 2221,
            "final_particle_count": 2221,
            "nominal_mass_convention_equivalent_ml": 59.967,
            "nominal_error_ml": -0.033,
        },
        "isosurface": {
            "enabled": True,
            "mesh_point_count": 900,
            "mesh_face_count": 1700,
            "mesh_finite_points": True,
            "mesh_extent_nonempty": True,
        },
        "render": {
            "frame_count": 31,
            "frame_sha256": [f"{index:064x}" for index in range(1, 32)],
            "stage_frame_sha256": frame_hashes,
            "video_sha256": "f" * 64,
            "video_decodable": True,
            "video_decoded_frame_count": 31,
            "stage_frames_are_video_subsets": True,
        },
        "memory": {
            "hard_limit_mib": 40000,
            "peak_used_mib": 1400,
            "hard_limit_exceeded": False,
            "isosurface_enabled_seconds": 10.0,
            "isosurface_time_limit_seconds": 2700,
        },
        "timing": {"wall_seconds": 100.0, "wall_limit_seconds": 5400},
        "errors": [],
    }


def test_substantive_receipt_accepts_observed_three_pour_run():
    assert validate_three_pour_receipt(_passing_receipt()) == ()


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda r: r["particle_system"].update(count=2, exactly_one=False), "particle_system_count"),
        (lambda r: r["particle_system"].update(mass_api_count=1), "mass_api_present"),
        (lambda r: r["particle_system"].update(collider_types=["Cylinder"]), "collider_type_unsupported"),
        (lambda r: r["pours"][1].update(tank_id="t100"), "pour_order_or_volume"),
        (lambda r: r["pours"][0].update(escaped_particle_count=1), "particles_escaped"),
        (lambda r: r["pours"][2].update(stable=False), "stage_not_stable"),
        (lambda r: r["isosurface"].update(mesh_point_count=0), "isosurface_empty"),
        (lambda r: r["render"].update(video_decodable=False), "video_not_decodable"),
        (lambda r: r["render"].update(stage_frames_are_video_subsets=False), "stage_frame_not_subset"),
        (lambda r: r["memory"].update(hard_limit_exceeded=True), "memory_limit_exceeded"),
        (lambda r: r["timing"].update(wall_seconds=5400.1), "wall_time_exceeded"),
        (
            lambda r: r["render_guides"].update(collision_api_count=1),
            "render_guide_collision_api",
        ),
        (
            lambda r: r["render_guides"].update(authored_count=7),
            "render_guide_count",
        ),
        (
            lambda r: r["camera"].update(
                clipping_range_authored=None,
                clipping_range_effective=[1.0, 1000000.0],
            ),
            "camera_clipping_range",
        ),
    ],
)
def test_substantive_receipt_rejects_real_failures(mutator, expected):
    receipt = deepcopy(_passing_receipt())
    mutator(receipt)

    assert expected in validate_three_pour_receipt(receipt)


def test_version_metadata_is_not_a_substantive_gate_failure():
    receipt = _passing_receipt()
    receipt["versions"]["physx"] = None

    assert validate_three_pour_receipt(receipt) == ()


def test_runner_plan_only_path_needs_no_isaac_and_refuses_overwrite(tmp_path):
    script = ROOT / "scripts" / "physx_three_pour_sequence.py"
    output = tmp_path / "plan"
    command = [sys.executable, str(script), "--out", str(output), "--plan-only"]

    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert json.loads((output / "plan.json").read_text(encoding="utf-8")) == build_three_pour_plan()
    repeated = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "output path already exists" in repeated.stderr


def test_runner_cli_freezes_capture_retry_default_and_reports_invalid_value(tmp_path):
    script = ROOT / "scripts" / "physx_three_pour_sequence.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--max-capture-tries MAX_CAPTURE_TRIES" in help_result.stdout
    assert "(default: 20)" in help_result.stdout

    invalid = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out",
            str(tmp_path / "bad"),
            "--plan-only",
            "--max-capture-tries",
            "0",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert invalid.stderr.rstrip().endswith(
        "physx_three_pour_sequence.py: error: max_capture_tries must be a positive integer"
    )


def test_kit_start_disables_async_throttling_for_stopped_timeline(monkeypatch):
    started = []
    running_app = object()

    isaacsim = types.ModuleType("isaacsim")

    def simulation_app(config):
        started.append(config)
        return object()

    isaacsim.SimulationApp = simulation_app
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_kit = types.ModuleType("omni.kit")
    omni_kit.__path__ = []
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: None if not started else running_app
    omni.kit = omni_kit
    omni_kit.app = omni_kit_app
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.kit", omni_kit)
    monkeypatch.setitem(sys.modules, "omni.kit.app", omni_kit_app)

    app, owned = sequence._start_or_reuse_kit()

    assert app is running_app
    assert owned is not None
    assert started == [
        {
            "headless": True,
            "hide_ui": True,
            "renderer": "PathTracing",
            "extra_args": [
                "--/exts/isaacsim.core.throttling/enable_async=false"
            ],
        }
    ]


def test_replicator_capture_contract_waits_for_render_without_advancing_physics():
    calls = []

    class Orchestrator:
        def set_capture_on_play(self, value):
            calls.append(("set_capture_on_play", value))

        def step(self, **kwargs):
            calls.append(("step", kwargs))

    rep = types.SimpleNamespace(orchestrator=Orchestrator())

    sequence._configure_replicator_capture(rep)
    sequence._replicator_same_state_step(rep, rt_subframes=8)

    assert calls == [
        ("set_capture_on_play", False),
        (
            "step",
            {
                "delta_time": 0.0,
                "pause_timeline": True,
                "rt_subframes": 8,
                "wait_for_render": True,
            },
        ),
    ]


def test_receipt_preserves_the_renderer_compatibility_contract():
    args = sequence.build_parser().parse_args(["--out", "unused"])

    receipt = sequence._empty_receipt(args)

    assert receipt["render"]["capture_on_play"] is False
    assert receipt["render"]["wait_for_render"] is True
    assert receipt["render"]["same_state_delta_time"] == 0.0
    assert receipt["render"]["async_throttling_enabled"] is None
    assert receipt["render"]["rejected_rgb_diagnostics"] == []
    assert receipt["render_guides"] == {
        "expected_count": 8,
        "authored_count": 0,
        "collision_api_count": None,
        "paths": [],
    }
    assert receipt["camera"] == {
        "path": "/World/Camera",
        "clipping_range_requested": [0.01, 10000.0],
        "clipping_range_authored": None,
        "clipping_range_effective": None,
    }


def test_camera_clipping_helper_authors_and_attests_nvidia_example_range():
    class Attribute:
        def __init__(self):
            self.value = (1.0, 1000000.0)
            self.authored = False

        def Set(self, value):
            self.value = tuple(value)
            self.authored = True
            return True

        def Get(self):
            return self.value

        def HasAuthoredValueOpinion(self):
            return self.authored

    attribute = Attribute()

    class Camera:
        def CreateClippingRangeAttr(self):
            return attribute

        def GetClippingRangeAttr(self):
            return attribute

    class Gf:
        Vec2f = staticmethod(lambda near, far: (near, far))

    receipt = {"camera": {}}
    sequence._author_and_attest_camera_clipping(Camera(), Gf, receipt)

    assert receipt["camera"] == {
        "clipping_range_requested": [0.01, 10000.0],
        "clipping_range_authored": [0.01, 10000.0],
        "clipping_range_effective": [0.01, 10000.0],
    }


def test_camera_clipping_helper_rejects_unauthored_usd_default_range():
    class DefaultAttribute:
        def Set(self, value):
            return True

        def Get(self):
            return (1.0, 1000000.0)

        def HasAuthoredValueOpinion(self):
            return False

    attribute = DefaultAttribute()

    class Camera:
        def CreateClippingRangeAttr(self):
            return attribute

        def GetClippingRangeAttr(self):
            return attribute

    class Gf:
        Vec2f = staticmethod(lambda near, far: (near, far))

    receipt = {"camera": {}}
    with pytest.raises(RuntimeError, match="camera clipping range"):
        sequence._author_and_attest_camera_clipping(Camera(), Gf, receipt)

    assert receipt["camera"]["clipping_range_requested"] == [0.01, 10000.0]
    assert receipt["camera"]["clipping_range_authored"] is None
    assert receipt["camera"]["clipping_range_effective"] == [1.0, 1000000.0]


def test_render_guide_specs_are_exact_bounded_and_outside_the_fluid_interior():
    specs = physx_three_pour.build_render_guide_specs()

    assert specs == (
        {"name": "BaseXMin", "center_m": (-0.018, -0.003, 0.0), "size_m": (0.001, 0.001, 0.038)},
        {"name": "BaseXMax", "center_m": (0.018, -0.003, 0.0), "size_m": (0.001, 0.001, 0.038)},
        {"name": "BaseZMin", "center_m": (0.0, -0.003, -0.018), "size_m": (0.038, 0.001, 0.001)},
        {"name": "BaseZMax", "center_m": (0.0, -0.003, 0.018), "size_m": (0.038, 0.001, 0.001)},
        {"name": "EdgeXMinZMin", "center_m": (-0.018, 0.073, -0.018), "size_m": (0.001, 0.146, 0.001)},
        {"name": "EdgeXMinZMax", "center_m": (-0.018, 0.073, 0.018), "size_m": (0.001, 0.146, 0.001)},
        {"name": "EdgeXMaxZMin", "center_m": (0.018, 0.073, -0.018), "size_m": (0.001, 0.146, 0.001)},
        {"name": "EdgeXMaxZMax", "center_m": (0.018, 0.073, 0.018), "size_m": (0.001, 0.146, 0.001)},
    )
    for spec in specs:
        center = spec["center_m"]
        size = spec["size_m"]
        bounds = tuple(
            (center[axis] - size[axis] / 2.0, center[axis] + size[axis] / 2.0)
            for axis in range(3)
        )
        assert (
            bounds[1][1] <= 0.0
            or bounds[0][1] <= -0.015
            or bounds[0][0] >= 0.015
            or bounds[2][1] <= -0.015
            or bounds[2][0] >= 0.015
        )


def test_rejected_full_shape_rgb_is_saved_only_as_diagnostic_evidence(tmp_path):
    rgb = np.full((2, 3, 3), 17, dtype=np.uint8)

    class FakeCv2:
        COLOR_RGB2BGR = 7

        @staticmethod
        def cvtColor(value, code):
            assert code == FakeCv2.COLOR_RGB2BGR
            return value[..., ::-1]

        @staticmethod
        def imwrite(path, value):
            assert value.shape == (2, 3, 3)
            Path(path).write_bytes(b"diagnostic png fixture")
            return True

    record = sequence._write_rejected_rgb_diagnostic(
        FakeCv2,
        np,
        tmp_path,
        rgb,
        step=0,
        phase="prefill",
    )

    artifact = tmp_path / record["path"]
    assert artifact.read_bytes() == b"diagnostic png fixture"
    assert record == {
        "path": "diagnostics/rejected-rgb-step-00000-prefill.png",
        "sha256": hashlib.sha256(b"diagnostic png fixture").hexdigest(),
        "shape": [2, 3, 3],
        "dtype": "uint8",
        "channel_mean": [17.0, 17.0, 17.0],
        "channel_std": [0.0, 0.0, 0.0],
        "spatial_rgb_std": 0.0,
        "minimum": 17,
        "maximum": 17,
        "spatially_constant": True,
        "included_in_frame_count": False,
        "included_in_video": False,
        "included_as_stage": False,
    }
