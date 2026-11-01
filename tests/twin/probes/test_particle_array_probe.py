from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from twin.probes.particle_array_probe import (
    ARRAY_NAMES,
    CANONICAL_TWIN_SHA256,
    CAPTURE_STEPS,
    PHYSICS_HZ,
    SEED,
    build_probe_plan,
    center_support_diagnostic,
    validate_probe_receipt,
)

SCRIPTS_ROOT = Path(__file__).parents[3] / "cli/twin/probes"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
import cli.twin.probes.physx_particle_array_probe as runtime_probe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _passing_receipt(tmp_path: Path) -> dict[str, object]:
    plan = build_probe_plan()
    stages: list[dict[str, object]] = []
    cumulative_counts = (370, 740)
    for step, count in zip(CAPTURE_STEPS, cumulative_counts):
        stage_dir = tmp_path / f"step-{step:05d}"
        stage_dir.mkdir(parents=True)
        arrays: dict[str, dict[str, object]] = {}
        for name in ARRAY_NAMES:
            if name in {"ids", "proto_indices"}:
                values = np.arange(count, dtype=np.int64)
                if name == "proto_indices":
                    values %= 10
            elif name == "compositions":
                values = np.zeros((count, 3), dtype=np.float64)
                values[:, 0] = 1.0
            else:
                values = np.zeros((count, 3), dtype=np.float64)
                if name == "scales":
                    values[:] = 1.0
            path = stage_dir / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            arrays[name] = {
                "path": str(path.relative_to(tmp_path)),
                "sha256": _sha256(path),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "finite": True,
            }
        stages.append(
            {
                "physics_step": step,
                "expected_particle_count": count,
                "emitted_particle_count": count,
                "inside_vial_interior_count": count,
                "outside_vial_interior_count": 0,
                "escaped_particle_count": 0,
                "ids_exact": True,
                "arrays": arrays,
                "diagnostics": {
                    "raw_simulation_positions": {
                        "q05_z_m": 0.98,
                        "q95_z_m": 0.99,
                        "center_support_width_m": 0.01 / 0.9,
                        "wall_to_center_support_m": 0.0029 + 0.01 / 0.9,
                        "visual_prototype_support_m": 0.01 / 0.9 + 2 * 0.0015,
                        "physical_radius_interpretation": False,
                    },
                    "smoothed_instancer_positions": {
                        "q05_z_m": 0.98,
                        "q95_z_m": 0.99,
                        "center_support_width_m": 0.01 / 0.9,
                        "wall_to_center_support_m": 0.0029 + 0.01 / 0.9,
                        "visual_prototype_support_m": 0.01 / 0.9 + 2 * 0.0015,
                        "physical_radius_interpretation": False,
                    },
                },
            }
        )
    callback_trace = [
        {
            "callback_index": index,
            "dt_s": 1.0 / PHYSICS_HZ,
            "timeline_time_s": index / PHYSICS_HZ,
            "app_update_index": index,
        }
        for index in range(1, CAPTURE_STEPS[-1] + 1)
    ]
    trace_path = tmp_path / "callback-trace.json"
    trace_path.write_text(
        json.dumps(callback_trace, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_records = [
        {"path": record["path"], "sha256": record["sha256"]}
        for stage in stages
        for record in stage["arrays"].values()
    ]
    manifest_records.append(
        {"path": "callback-trace.json", "sha256": _sha256(trace_path)}
    )
    manifest_path = tmp_path / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps({"records": manifest_records}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "experiment": "a36_physx_first_two_stage_particle_array_probe",
        "status": "pass",
        "errors": [],
        "plan": plan,
        "command": {
            "argv": ["cli/twin/probes/physx_particle_array_probe.py", "--seed", str(SEED)],
            "seed": SEED,
            "denominator": 1,
            "script_sha256": "c" * 64,
        },
        "source": {
            "stage_sha256": CANONICAL_TWIN_SHA256,
            "canonical_stage_sha256": CANONICAL_TWIN_SHA256,
            "layers_before": [{"path": "fill_station.usda", "sha256": "d" * 64}],
            "layers_after": [{"path": "fill_station.usda", "sha256": "d" * 64}],
            "source_layers_unchanged": True,
        },
        "runtime": {
            "isaac_sim": "6.2.0.0",
            "kit": "110.4.0",
            "physx": "110.4.2",
            "render_product_created": False,
            "rgb_captured": False,
            "video_created": False,
        },
        "particle_system": {
            "particle_contact_offset_m": 0.003,
            "fluid_rest_offset_m": 0.0015,
            "contact_offset_m": 0.003,
            "rest_offset_m": 0.0029,
            "solid_rest_offset_m": 0.0029,
            "max_velocity_m_s": 2.5,
            "ccd_enabled": True,
            "smoothing_enabled": True,
            "smoothing_strength": 1.0,
            "anisotropy_enabled": True,
            "anisotropy_scale": 1.0,
            "pbd_material_readback": {
                "density_kg_m3": 1000.0,
                "friction": 0.1,
                "cohesion": 0.01,
                "damping": 0.0,
                "surface_tension": 0.0074,
                "viscosity": 1.7e-6,
                "vorticity_confinement": 0.0,
            },
            "mass_density_readback": {
                "particle_mass_argument_kg": 0.001,
                "density_argument_kg_m3": 1000.0,
                "total_mass_attr_kg": 0.001,
                "density_attr_kg_m3": 1000.0,
            },
        },
        "point_instancer": {
            "one_shared_point_instancer": True,
            "simulation_points_authored": True,
            "prototype_count": 15,
            "max_particles": 2221,
        },
        "colliders": {
            "specs": plan["collider_specs"],
            "floor_path": "/FillStation/FluidmixF2/Colliders/Floor",
            "floor_top_z_m": 0.9762065645,
            "visible_inner_bottom_z_m": 0.9762065645,
            "floor_matches_visible_inner_bottom": True,
            "all_invisible": True,
        },
        "timing": {
            "physics_hz": PHYSICS_HZ,
            "executed_physics_steps": CAPTURE_STEPS[-1],
            "callbacks_observed": CAPTURE_STEPS[-1],
            "callback_indices_exact": True,
            "callback_dts_exact": True,
            "callback_times_exact": True,
            "callback_app_update_indices_exact": True,
            "trace_path": "callback-trace.json",
            "trace_sha256": _sha256(trace_path),
            "timeline_looping": False,
        },
        "emission": {
            "seed": SEED,
            "emitted_particle_count": 740,
            "event_count": 219,
        },
        "stages": stages,
        "artifacts": {
            "manifest_path": "artifact-manifest.json",
            "manifest_sha256": _sha256(manifest_path),
            "record_count": len(manifest_records),
        },
    }


def test_probe_plan_freezes_exact_a34_first_two_stage_contract() -> None:
    plan = build_probe_plan()

    assert plan["seed"] == SEED == 20261025
    assert plan["physics_hz"] == PHYSICS_HZ == 120
    assert plan["capture_steps"] == [1412, 2622]
    assert plan["stop_after_physics_step"] == 2622
    assert plan["programmed_pours_ml"] == [10, 10, 40]
    assert plan["tank_order"] == ["t200", "t400", "t100"]
    assert plan["expected_particle_counts"] == {"1412": 370, "2622": 740}
    assert plan["render_product"] is False
    assert plan["rgb"] is False
    assert plan["video"] is False
    assert plan["canonical_twin_sha256"] == CANONICAL_TWIN_SHA256
    assert plan["timeline"]["looping"] is False
    assert plan["timeline"]["end_margin_callbacks"] == 1
    assert plan["timeline"]["required_end_time_s"] == pytest.approx(2623 / 120)
    assert plan["array_names"] == list(ARRAY_NAMES)
    assert plan["diagnostic"]["center_support_formula"] == "(Q95_z-Q05_z)/0.90"
    assert plan["diagnostic"]["wall_to_center_formula"] == (
        "particle_to_nonparticle_rest_offset_m+center_support_width_m"
    )
    assert plan["diagnostic"]["visual_prototype_formula"] == (
        "center_support_width_m+2*prototype_radius_m*prototype_scale_z"
    )
    assert plan["diagnostic"]["fluid_rest_offset_is_physical_radius"] is False
    assert plan["diagnostic"]["a34_values_used_to_tune_thresholds"] is False
    assert plan["diagnostic"]["rescores_a34"] is False
    assert len(plan["a34_physics_plan_sha256"]) == 64


def test_center_support_diagnostic_keeps_physical_and_visual_support_separate() -> None:
    z = np.linspace(0.100, 0.190, 101, dtype=np.float64)

    result = center_support_diagnostic(
        z,
        particle_to_nonparticle_rest_offset_m=0.0029,
        prototype_radius_m=0.0015,
    )

    assert result["q05_z_m"] == pytest.approx(0.1045)
    assert result["q95_z_m"] == pytest.approx(0.1855)
    assert result["center_support_width_m"] == pytest.approx(0.09)
    assert result["wall_to_center_support_m"] == pytest.approx(0.0929)
    assert result["visual_prototype_support_m"] == pytest.approx(0.093)
    assert result["physical_radius_interpretation"] is False
    assert result["visual_prototype_support_is_physical"] is False


@pytest.mark.parametrize(
    "z,rest_offset,prototype_radius",
    [
        ([], 0.0029, 0.0015),
        ([0.1, np.nan], 0.0029, 0.0015),
        ([0.1], -0.1, 0.0015),
        ([0.1], 0.0029, -0.1),
    ],
)
def test_center_support_diagnostic_rejects_invalid_input(
    z: list[float], rest_offset: float, prototype_radius: float
) -> None:
    with pytest.raises(ValueError):
        center_support_diagnostic(
            z,
            particle_to_nonparticle_rest_offset_m=rest_offset,
            prototype_radius_m=prototype_radius,
        )


def test_validate_probe_receipt_accepts_complete_artifacts(tmp_path: Path) -> None:
    receipt = _passing_receipt(tmp_path)

    assert validate_probe_receipt(receipt, artifact_root=tmp_path) == ()


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (lambda r: r["runtime"].__setitem__("render_product_created", True), "render_product_forbidden"),
        (lambda r: r["timing"].__setitem__("callbacks_observed", 2621), "callback_trace_mismatch"),
        (lambda r: r["stages"][0].__setitem__("outside_vial_interior_count", 1), "containment_mismatch"),
        (lambda r: r["stages"][1].__setitem__("ids_exact", False), "particle_identity_mismatch"),
        (lambda r: r["stages"][0]["arrays"].pop("velocities"), "stage_arrays_incomplete"),
        (lambda r: r["source"].__setitem__("stage_sha256", "0" * 64), "source_contract_mismatch"),
        (lambda r: r["point_instancer"].__setitem__("one_shared_point_instancer", False), "point_instancer_contract_mismatch"),
        (lambda r: r["emission"].__setitem__("emitted_particle_count", 739), "emission_contract_mismatch"),
        (lambda r: r["particle_system"]["pbd_material_readback"].__setitem__("viscosity", 0.1), "pbd_material_readback_mismatch"),
        (lambda r: r["colliders"].__setitem__("floor_matches_visible_inner_bottom", False), "collider_contract_mismatch"),
    ],
)
def test_validate_probe_receipt_fails_closed(
    tmp_path: Path, mutation, failure: str
) -> None:
    receipt = _passing_receipt(tmp_path)
    mutation(receipt)

    assert failure in validate_probe_receipt(receipt, artifact_root=tmp_path)


def test_validate_probe_receipt_rejects_bad_hash_and_nonfinite_array(
    tmp_path: Path,
) -> None:
    receipt = _passing_receipt(tmp_path)
    array_record = receipt["stages"][0]["arrays"]["velocities"]
    path = tmp_path / array_record["path"]
    values = np.load(path, allow_pickle=False)
    values[0, 0] = np.nan
    np.save(path, values, allow_pickle=False)

    failures = validate_probe_receipt(receipt, artifact_root=tmp_path)

    assert "artifact_hash_mismatch" in failures
    assert "array_nonfinite" in failures


def test_validate_probe_receipt_rejects_unreadable_callback_trace_and_manifest(
    tmp_path: Path,
) -> None:
    receipt = _passing_receipt(tmp_path)
    (tmp_path / receipt["timing"]["trace_path"]).unlink()
    (tmp_path / receipt["artifacts"]["manifest_path"]).unlink()

    failures = validate_probe_receipt(receipt, artifact_root=tmp_path)

    assert "callback_trace_artifact_invalid" in failures
    assert "manifest_missing_or_invalid" in failures


def test_float32_physx_offset_readbacks_are_accepted(tmp_path: Path) -> None:
    receipt = _passing_receipt(tmp_path)
    receipt["particle_system"].update(
        {
            "particle_contact_offset_m": 0.003000000026077032,
            "fluid_rest_offset_m": 0.001500000013038516,
            "contact_offset_m": 0.003000000026077032,
            "rest_offset_m": 0.002899999963119626,
            "solid_rest_offset_m": 0.002899999963119626,
        }
    )

    assert validate_probe_receipt(receipt, artifact_root=tmp_path) == ()


def test_runtime_pbd_readback_includes_values_and_authored_flags() -> None:
    class Attr:
        def __init__(self, value: float) -> None:
            self.value = value

        def Get(self) -> float:
            return self.value

        def HasAuthoredValueOpinion(self) -> bool:
            return True

    class Pbd:
        def __getattr__(self, name: str):
            values = {
                "GetDensityAttr": 1000.0,
                "GetFrictionAttr": 0.1,
                "GetCohesionAttr": 0.01,
                "GetDampingAttr": 0.0,
                "GetSurfaceTensionAttr": 0.0074,
                "GetViscosityAttr": 1.7e-6,
                "GetVorticityConfinementAttr": 0.0,
            }
            if name not in values:
                raise AttributeError(name)
            return lambda: Attr(values[name])

    assert runtime_probe._pbd_material_readback(Pbd()) == {
        "density_kg_m3": 1000.0,
        "density_authored": True,
        "friction": 0.1,
        "friction_authored": True,
        "cohesion": 0.01,
        "cohesion_authored": True,
        "damping": 0.0,
        "damping_authored": True,
        "surface_tension": 0.0074,
        "surface_tension_authored": True,
        "viscosity": 1.7e-6,
        "viscosity_authored": True,
        "vorticity_confinement": 0.0,
        "vorticity_confinement_authored": True,
    }


def test_failed_stage_containment_preserves_arrays_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    count = 370
    ids = list(range(count))
    raw_positions = np.zeros((count, 3), dtype=np.float64)
    smoothed_positions = np.zeros((count, 3), dtype=np.float64)
    vectors = np.zeros((count, 3), dtype=np.float64)
    scales = np.ones((count, 3), dtype=np.float64)
    proto_indices = np.zeros(count, dtype=np.int64)
    compositions = np.zeros((count, 3), dtype=np.float64)
    compositions[:, 0] = 1.0

    class Attr:
        def __init__(self, value: object) -> None:
            self.value = value

        def Get(self) -> object:
            return self.value

    class Points:
        def GetSimulationPointsAttr(self) -> Attr:
            return Attr(raw_positions)

        def GetVelocitiesAttr(self) -> Attr:
            return Attr(vectors)

        def GetScalesAttr(self) -> Attr:
            return Attr(scales)

        def GetProtoIndicesAttr(self) -> Attr:
            return Attr(proto_indices.tolist())

        def GetPrim(self) -> object:
            return object()

    monkeypatch.setattr(
        runtime_probe.f2,
        "_read_runtime_particle_identity",
        lambda _points, _expected: ids,
    )
    monkeypatch.setattr(
        runtime_probe.f2,
        "_positions",
        lambda _prim, *, instancer: smoothed_positions,
    )

    with pytest.raises(RuntimeError, match="stage raw particle containment mismatch"):
        runtime_probe._capture_stage_arrays(
            output=tmp_path,
            step=1412,
            particle_points=Points(),
            compositions=compositions,
            expected_ids=ids,
            particle_to_nonparticle_rest_offset_m=0.0016666666666666668,
        )

    stage = tmp_path / "step-01412"
    assert {path.name for path in stage.glob("*.npy")} == {
        f"{name}.npy" for name in ARRAY_NAMES
    }
    failure = json.loads(
        (stage / "failure-diagnostic.json").read_text(encoding="utf-8")
    )
    assert failure["failure"] == "stage raw particle containment mismatch"
    assert failure["expected_particle_count"] == count
    assert failure["inside_vial_interior_count"] == 0
    assert failure["outside_vial_interior_count"] == count
    assert failure["escaped_particle_count"] == count

    (tmp_path / "plan.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "command.json").write_text("{}\n", encoding="utf-8")
    manifest = runtime_probe._build_manifest(
        tmp_path,
        {"status": "insufficient_evidence"},
        [],
    )
    paths = {record["path"] for record in manifest["records"]}
    assert "step-01412/failure-diagnostic.json" in paths
    assert "step-01412/raw_simulation_positions.npy" in paths


def test_plan_only_writes_frozen_plan_without_isaac(tmp_path: Path) -> None:
    script = Path(__file__).parents[3] / "cli/twin/probes" / "physx_particle_array_probe.py"
    output = tmp_path / "plan"
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out",
            str(output),
            "--stage",
            str(stage),
            "--plan-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((output / "plan.json").read_text(encoding="utf-8")) == build_probe_plan()
