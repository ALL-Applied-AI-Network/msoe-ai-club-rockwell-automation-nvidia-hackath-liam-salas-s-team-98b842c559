"""CPU contracts for measured, provenance-only F2 nuisance authoring."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fluidmix.paint_ball_emitter as runtime_contract  # noqa: E402
from fluidmix.paint_ball_emitter import bind_external_nuisance_receipt  # noqa: E402
from fluidmix.nuisance_authoring import EXPECTED_LIGHT_PATHS as AUTHORING_LIGHT_PATHS  # noqa: E402
from fluidmix.nuisance_authoring import LIGHT_SPECS  # noqa: E402


EXPECTED_LIGHT_PATHS = tuple(
    sorted(
        {
            "/FillStation/FillStation_LocalRoot/V2_base_reflection/V2_base_reflection",
            "/FillStation/FillStation_LocalRoot/Fill_right_reflection/Fill_right_reflection",
            "/FillStation/FillStation_LocalRoot/Fill_front_softbox/Fill_front_softbox",
            "/FillStation/FillStation_LocalRoot/Fill_overhead_panel/Fill_overhead_panel",
            "/FillStation/FillStation_LocalRoot/V2_guard_bounce/V2_guard_bounce",
            "/FillStation/env_light",
        }
    )
)


def _binding(tmp_path: Path) -> dict[str, object]:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "real_photometric_nuisance_range",
                "status": "measured",
                "photometric_status": "measured",
                "shadow_status": "measured",
                "pooled": {
                    "candidate_count": 888,
                    "usable_count": 884,
                    "shadow_observation_count": 884,
                    "percentile_bounds": {
                        "luma": {"p05": 148.503071, "p50": 151.881736, "p95": 171.578061},
                        "shadow_centroid_x": {"p05": 115.599612, "p50": 126.785024, "p95": 134.378198},
                        "shadow_centroid_y": {"p05": 127.736358, "p50": 146.26883, "p95": 152.090067},
                        "shadow_edge_angle_degrees": {"p05": 56.684932, "p50": 59.899991, "p95": 70.652955},
                        "shadow_area_fraction": {"p05": 0.100652, "p50": 0.136695, "p95": 0.401708},
                        "shadow_contrast": {"p05": 17.0, "p50": 18.8544, "p95": 20.7192},
                        "shadow_penumbra_10_90_px": {"p05": 104.188797, "p50": 123.021875, "p95": 131.824096},
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return bind_external_nuisance_receipt(
        receipt, hashlib.sha256(receipt.read_bytes()).hexdigest()
    )


def _inventory() -> dict[str, dict[str, object]]:
    rows = {
        "/FillStation/FillStation_LocalRoot/V2_base_reflection/V2_base_reflection": (
            350.140869140625,
            "/FillStation/FillStation_LocalRoot/V2_base_reflection",
            [-6.0000152587890625, -1.0, 10.0],
            [31.311159133911133, -0.00000330711668539152, -80.53768157958984],
        ),
        "/FillStation/FillStation_LocalRoot/Fill_right_reflection/Fill_right_reflection": (
            143.2394561767578,
            "/FillStation/FillStation_LocalRoot/Fill_right_reflection",
            [-0.6000213623046875, 7.0, 9.000003814697266],
            [54.56145477294922, -0.000004108448592887726, -175.1009063720703],
        ),
        "/FillStation/FillStation_LocalRoot/Fill_front_softbox/Fill_front_softbox": (
            206.90142822265625,
            "/FillStation/FillStation_LocalRoot/Fill_front_softbox",
            [6.9999847412109375, -7.0, 7.0],
            [73.14079284667969, -0.000005165207767277025, 44.999996185302734],
        ),
        "/FillStation/FillStation_LocalRoot/Fill_overhead_panel/Fill_overhead_panel": (
            557.0422973632812,
            "/FillStation/FillStation_LocalRoot/Fill_overhead_panel",
            [2.9999847412109375, 0.0, 15.000003814697266],
            [14.036245346069336, 0.0000035860348361893557, 90.0],
        ),
        "/FillStation/FillStation_LocalRoot/V2_guard_bounce/V2_guard_bounce": (
            70.0281753540039,
            "/FillStation/FillStation_LocalRoot/V2_guard_bounce",
            [-0.70001220703125, -4.0, 8.0],
            [46.437679290771484, -0.0000012382580507619423, 18.004161834716797],
        ),
    }
    inventory = {
        path: {
            "type_name": "RectLight",
            "intensity": intensity,
            "exposure": 0.0,
            "transform_path": transform_path,
            "translate": translate,
            "rotate_xyz_degrees": rotate,
        }
        for path, (intensity, transform_path, translate, rotate) in rows.items()
    }
    inventory["/FillStation/env_light"] = {
        "type_name": "DomeLight",
        "intensity": 0.18000000715255737,
        "exposure": 0.0,
        "transform_path": None,
        "translate": None,
        "rotate_xyz_degrees": None,
    }
    return inventory


def test_case_seed_plan_is_deterministic_bounded_and_model_excluded(tmp_path) -> None:
    build_nuisance_authoring_plan = getattr(
        runtime_contract, "build_nuisance_authoring_plan", None
    )
    assert callable(build_nuisance_authoring_plan), "nuisance authoring boundary is missing"
    binding = _binding(tmp_path)
    inventory = _inventory()

    plan = build_nuisance_authoring_plan(
        binding,
        case_id="BATCH_A:V2",
        seed=330102,
        light_inventory=inventory,
    )

    assert plan == build_nuisance_authoring_plan(
        binding,
        case_id="BATCH_A:V2",
        seed=330102,
        light_inventory=inventory,
    )
    assert plan != build_nuisance_authoring_plan(
        binding,
        case_id="BATCH_B:V2",
        seed=330102,
        light_inventory=inventory,
    )
    assert plan["mapping_status"] == "uncalibrated_synthetic_nuisance_mapping"
    assert plan["real_mapping_claimed"] is False
    assert plan["excluded_from_model_inputs"] is True
    assert plan["excluded_from_deploy_inputs"] is True
    assert plan["receipt_sha256"] == binding["receipt_sha256"]
    assert set(plan["light_authoring"]) == set(EXPECTED_LIGHT_PATHS)
    for metric, sample in plan["sampled_measured_metrics"].items():
        bounds = binding["percentile_bounds"][metric]
        assert bounds["p05"] <= sample <= bounds["p95"]
    for light_path, authored in plan["light_authoring"].items():
        assert authored["intensity"] > 0.0
        assert isinstance(authored["exposure"], float)
        if light_path == "/FillStation/env_light":
            assert authored["transform_path"] is None
            assert authored["translate"] is None
            assert authored["rotate_xyz_degrees"] is None
        else:
            assert len(authored["translate"]) == 3
            assert len(authored["rotate_xyz_degrees"]) == 3
    assert "case_id" not in plan.get("model_inputs", {})
    assert "light_authoring" not in plan.get("model_inputs", {})


def test_plan_fails_closed_when_expected_light_or_control_is_missing(tmp_path) -> None:
    build_nuisance_authoring_plan = getattr(
        runtime_contract, "build_nuisance_authoring_plan", None
    )
    assert callable(build_nuisance_authoring_plan), "nuisance authoring boundary is missing"
    binding = _binding(tmp_path)
    inventory = _inventory()
    inventory.pop("/FillStation/env_light")
    with pytest.raises(ValueError, match="light inventory"):
        build_nuisance_authoring_plan(
            binding,
            case_id="BATCH_A:V1",
            seed=330101,
            light_inventory=inventory,
        )

    inventory = _inventory()
    inventory[next(iter(inventory))]["exposure"] = None
    with pytest.raises(ValueError, match="exposure"):
        build_nuisance_authoring_plan(
            binding,
            case_id="BATCH_A:V1",
            seed=330101,
            light_inventory=inventory,
        )


class _FakeVec3:
    def __init__(self, *values: float) -> None:
        if len(values) != 3:
            raise TypeError("three components required")
        self.values = tuple(float(value) for value in values)

    def __iter__(self):
        return iter(self.values)


class _FakeAttr:
    def __init__(
        self,
        value: object,
        *,
        accepted_type: type | None = None,
        quantize_float: bool = False,
        offset: float = 0.0,
    ) -> None:
        self.value = value
        self.accepted_type = accepted_type
        self.quantize_float = quantize_float
        self.offset = offset
        self.set_calls: list[object] = []

    def IsValid(self) -> bool:
        return True

    def Get(self) -> object:
        return self.value

    def Set(self, value: object) -> bool:
        if self.accepted_type is not None and not isinstance(value, self.accepted_type):
            raise TypeError(f"expected {self.accepted_type.__name__}")
        self.set_calls.append(value)
        if self.quantize_float:
            self.value = struct.unpack("f", struct.pack("f", float(value)))[0] + self.offset
        else:
            self.value = value
        return True


class _FakePrim:
    def __init__(self, type_name: str, attrs: dict[str, _FakeAttr]) -> None:
        self.type_name = type_name
        self.attrs = attrs

    def IsValid(self) -> bool:
        return True

    def GetTypeName(self) -> str:
        return self.type_name

    def GetAttribute(self, name: str) -> _FakeAttr | None:
        return self.attrs.get(name)


class _FakeStage:
    def __init__(self, prims: dict[str, _FakePrim]) -> None:
        self.prims = prims
        self.session_layer = object()

    def GetPrimAtPath(self, path: str) -> _FakePrim | None:
        return self.prims.get(str(path))

    def GetSessionLayer(self) -> object:
        return self.session_layer


class _FakeUsd:
    class EditContext:
        def __init__(self, stage: _FakeStage, layer: object) -> None:
            self.stage = stage
            self.layer = layer

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None


def _fake_stage(
    *, typed: bool = True, quantized: bool = False, scalar_offset: float = 0.0
) -> _FakeStage:
    prims: dict[str, _FakePrim] = {}
    for path, row in _inventory().items():
        prims[path] = _FakePrim(
            str(row["type_name"]),
            {
                "inputs:intensity": _FakeAttr(
                    row["intensity"], quantize_float=quantized, offset=scalar_offset
                ),
                "inputs:exposure": _FakeAttr(
                    row["exposure"], quantize_float=quantized, offset=scalar_offset
                ),
            },
        )
        if row["transform_path"] is not None:
            prims[str(row["transform_path"])] = _FakePrim(
                "Xform",
                {
                    "xformOp:translate": _FakeAttr(
                        _FakeVec3(*row["translate"]) if typed else row["translate"],
                        accepted_type=_FakeVec3 if typed else None,
                    ),
                    "xformOp:rotateXYZ": _FakeAttr(
                        _FakeVec3(*row["rotate_xyz_degrees"])
                        if typed
                        else row["rotate_xyz_degrees"],
                        accepted_type=_FakeVec3 if typed else None,
                    ),
                },
            )
    return _FakeStage(prims)


def test_session_authoring_coerces_usd_vector_types_and_accepts_float_readback_quantization(
    tmp_path,
) -> None:
    apply = getattr(runtime_contract, "apply_nuisance_session_layer", None)
    assert callable(apply), "session nuisance authoring boundary is missing"
    stage = _fake_stage(quantized=True)
    result = apply(
        stage,
        _FakeUsd,
        _binding(tmp_path),
        case_id="BATCH_A:TYPED",
        seed=330103,
    )

    for path, spec in LIGHT_SPECS.items():
        if spec["transform_path"] is None:
            continue
        transform = stage.prims[spec["transform_path"]]
        assert all(
            isinstance(value, _FakeVec3)
            for attr_name in ("xformOp:translate", "xformOp:rotateXYZ")
            for value in transform.attrs[attr_name].set_calls
        )
    for path in AUTHORING_LIGHT_PATHS:
        authored = result["authored_light_properties"][path]
        readback = result["readback_light_properties"][path]
        assert math.isclose(
            authored["intensity"], readback["intensity"], rel_tol=0.0, abs_tol=1e-5
        )
        assert math.isclose(
            authored["exposure"], readback["exposure"], rel_tol=0.0, abs_tol=1e-5
        )


def test_session_authoring_rejects_raw_python_vector_attribute_type(tmp_path) -> None:
    apply = getattr(runtime_contract, "apply_nuisance_session_layer", None)
    assert callable(apply), "session nuisance authoring boundary is missing"
    with pytest.raises(ValueError, match="unsupported vector attribute type"):
        apply(
            _fake_stage(typed=False),
            _FakeUsd,
            _binding(tmp_path),
            case_id="BATCH_A:RAW",
            seed=330104,
        )


def test_session_authoring_rejects_material_float_readback_drift(tmp_path) -> None:
    apply = getattr(runtime_contract, "apply_nuisance_session_layer", None)
    assert callable(apply), "session nuisance authoring boundary is missing"
    with pytest.raises(ValueError, match="authored/readback mismatch"):
        apply(
            _fake_stage(quantized=True, scalar_offset=1.0e-2),
            _FakeUsd,
            _binding(tmp_path),
            case_id="BATCH_A:DRIFT",
            seed=330105,
        )


def test_session_authoring_is_deterministic_and_receipts_exact_readbacks(tmp_path) -> None:
    apply = getattr(runtime_contract, "apply_nuisance_session_layer", None)
    assert callable(apply), "session nuisance authoring boundary is missing"
    stage = _fake_stage()
    receipt = _binding(tmp_path)

    first = apply(
        stage,
        _FakeUsd,
        receipt,
        case_id="BATCH_A:V2",
        seed=330102,
    )
    second = apply(
        _fake_stage(),
        _FakeUsd,
        receipt,
        case_id="BATCH_A:V2",
        seed=330102,
    )

    assert first == second
    assert first["source_receipt_sha256"] == receipt["receipt_sha256"]
    assert first["case_id"] == "BATCH_A:V2"
    assert first["seed"] == 330102
    assert first["mapping_version"] == first["mapping"]["version"]
    assert first["provenance_only"] == {"case_id": "BATCH_A:V2", "seed": 330102}
    assert first["session_layer_only"] is True
    assert first["excluded_from_model_inputs"] is True
    assert first["model_inputs"] == {}
    assert first["authored_light_properties"] == first["readback_light_properties"]
    assert first["mapping"]["real_mapping_claimed"] is False
    limits = first["mapping"]["frozen_synthetic_limits"]
    source = _inventory()
    for path, row in first["authored_light_properties"].items():
        if row["transform_path"] is None:
            continue
        for index in range(2):
            assert abs(row["translate"][index] - source[path]["translate"][index]) <= limits["translation_xy_asset_units"]
        assert abs(row["translate"][2] - source[path]["translate"][2]) <= limits["translation_z_asset_units"]
        for index, key in enumerate(("rotation_x_degrees", "rotation_y_degrees", "rotation_z_degrees")):
            assert abs(row["rotate_xyz_degrees"][index] - source[path]["rotate_xyz_degrees"][index]) <= limits[key]


def test_session_authoring_fails_closed_before_any_write_when_readback_is_missing(tmp_path) -> None:
    apply = getattr(runtime_contract, "apply_nuisance_session_layer", None)
    assert callable(apply), "session nuisance authoring boundary is missing"
    stage = _fake_stage()
    receipt = _binding(tmp_path)
    light_path = EXPECTED_LIGHT_PATHS[0]
    exposure = stage.prims[light_path].attrs["inputs:exposure"]
    del stage.prims[light_path].attrs["inputs:exposure"]
    with pytest.raises(ValueError, match="exposure"):
        apply(stage, _FakeUsd, receipt, case_id="BATCH_A:V1", seed=330101)
    assert exposure.set_calls == []
