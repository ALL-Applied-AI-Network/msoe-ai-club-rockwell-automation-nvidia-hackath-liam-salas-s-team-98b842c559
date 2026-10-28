from __future__ import annotations

import copy
import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ROOT = "/FillStation/FillStation_LocalRoot"
SOURCE_HASHES = {
    "fill_station.usda": "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9",
    "geometry/fill_station.usdc": (
        "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db"
    ),
}
SOURCE_BYTES = {
    "fill_station.usda": 906,
    "geometry/fill_station.usdc": 26_464_568,
}
LOCAL_STAGE = Path(
    "/home/hackathon5/liamreporoot/"
    "msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/"
    "external/external/csi_digital_twin_v1/stations/fill_station/fill_station.usda"
)
LOCAL_ACCEPTED_CASE = Path(
    "/home/hackathon5/liamreporoot/f2-varied-cohort-merged-9e6254d/"
    "cases/CASE_7d3ff87739cdd500/app-output"
)
REMOTE_STAGE = "/home/ubuntu/final-recipe8/twin/fill_station.usda"
REMOTE_GEOMETRY = "/home/ubuntu/final-recipe8/twin/geometry/fill_station.usdc"

OUTGOING = {
    f"{ROOT}/Fill_vial_active",
    f"{ROOT}/FluidmixF2CountDrivenLiquid",
    f"{ROOT}/Fill_barcode_label",
    f"{ROOT}/Fill_label_top_rule",
    f"{ROOT}/Fill_barcode_00",
    f"{ROOT}/Fill_barcode_01",
    f"{ROOT}/Fill_barcode_02",
    f"{ROOT}/Fill_barcode_03",
    f"{ROOT}/Fill_barcode_04",
    f"{ROOT}/Fill_barcode_05",
    f"{ROOT}/Fill_barcode_06",
    f"{ROOT}/Fill_barcode_07",
    f"{ROOT}/Fill_barcode_08",
    f"{ROOT}/Fill_barcode_09",
    f"{ROOT}/Fill_barcode_10",
    f"{ROOT}/Fill_barcode_11",
    f"{ROOT}/Fill_barcode_12",
    f"{ROOT}/Fill_barcode_13",
    f"{ROOT}/Fill_barcode_14",
    f"{ROOT}/Fill_barcode_15",
    f"{ROOT}/Fill_barcode_16",
    f"{ROOT}/Fill_barcode_17",
    f"{ROOT}/Fill_barcode_18",
    f"{ROOT}/Fill_barcode_19",
    f"{ROOT}/Fill_barcode_20",
    f"{ROOT}/Fill_barcode_21",
    f"{ROOT}/Fill_barcode_22",
    f"{ROOT}/Fill_barcode_23",
    f"{ROOT}/Fill_barcode_24",
    f"{ROOT}/Fill_barcode_25",
    f"{ROOT}/Fill_barcode_26",
    f"{ROOT}/V3_carrier_0_cart_00",
    f"{ROOT}/V3_carrier_0_cart_01",
    f"{ROOT}/V3_carrier_0_cart_02",
    f"{ROOT}/V3_carrier_0_cart_03",
    f"{ROOT}/V3_carrier_0_cart_04",
    f"{ROOT}/V3_carrier_0_cart_05",
    f"{ROOT}/V3_carrier_0_cart_06",
    f"{ROOT}/V3_carrier_0_cart_07",
    f"{ROOT}/V3_carrier_0_cart_08",
    f"{ROOT}/V3_carrier_0_cart_09",
    f"{ROOT}/V3_carrier_0_cart_10",
    f"{ROOT}/V3_carrier_0_cart_11",
    f"{ROOT}/V3_carrier_0_cart_12",
    f"{ROOT}/V3_carrier_0_cart_13",
    f"{ROOT}/V3_carrier_0_cart_14",
    f"{ROOT}/V3_carrier_0_cart_15",
    f"{ROOT}/V3_carrier_0_cart_16",
    f"{ROOT}/V3_carrier_0_base_19",
    f"{ROOT}/V3_carrier_0_post_20",
    f"{ROOT}/V3_carrier_0_clamp_21",
    f"{ROOT}/V3_carrier_0_setscrew_17",
    f"{ROOT}/V3_carrier_0_setscrew_18",
    f"{ROOT}/V3_estimated_fastener_0",
    f"{ROOT}/V3_estimated_fastener_0_001",
    f"{ROOT}/V3_estimated_fastener_0_002",
    f"{ROOT}/V3_estimated_fastener_0_003",
}

INCOMING = {
    f"{ROOT}/Fill_vial_neighbour_1",
    f"{ROOT}/V3_carrier_1_cart_00",
    f"{ROOT}/V3_carrier_1_cart_01",
    f"{ROOT}/V3_carrier_1_cart_02",
    f"{ROOT}/V3_carrier_1_cart_03",
    f"{ROOT}/V3_carrier_1_cart_04",
    f"{ROOT}/V3_carrier_1_cart_05",
    f"{ROOT}/V3_carrier_1_cart_06",
    f"{ROOT}/V3_carrier_1_cart_07",
    f"{ROOT}/V3_carrier_1_cart_08",
    f"{ROOT}/V3_carrier_1_cart_09",
    f"{ROOT}/V3_carrier_1_cart_10",
    f"{ROOT}/V3_carrier_1_cart_11",
    f"{ROOT}/V3_carrier_1_cart_12",
    f"{ROOT}/V3_carrier_1_cart_13",
    f"{ROOT}/V3_carrier_1_cart_14",
    f"{ROOT}/V3_carrier_1_cart_15",
    f"{ROOT}/V3_carrier_1_cart_16",
    f"{ROOT}/V3_carrier_1_base_19",
    f"{ROOT}/V3_carrier_1_post_20",
    f"{ROOT}/V3_carrier_1_clamp_21",
    f"{ROOT}/V3_carrier_1_setscrew_17",
    f"{ROOT}/V3_carrier_1_setscrew_18",
    f"{ROOT}/V3_estimated_fastener_1",
    f"{ROOT}/V3_estimated_fastener_1_001",
    f"{ROOT}/V3_estimated_fastener_1_002",
    f"{ROOT}/V3_estimated_fastener_1_003",
}

QUEUED = {
    f"{ROOT}/Fill_vial_neighbour_2",
    f"{ROOT}/V3_carrier_2_cart_00",
    f"{ROOT}/V3_carrier_2_cart_01",
    f"{ROOT}/V3_carrier_2_cart_02",
    f"{ROOT}/V3_carrier_2_cart_03",
    f"{ROOT}/V3_carrier_2_cart_04",
    f"{ROOT}/V3_carrier_2_cart_05",
    f"{ROOT}/V3_carrier_2_cart_06",
    f"{ROOT}/V3_carrier_2_cart_07",
    f"{ROOT}/V3_carrier_2_cart_08",
    f"{ROOT}/V3_carrier_2_cart_09",
    f"{ROOT}/V3_carrier_2_cart_10",
    f"{ROOT}/V3_carrier_2_cart_11",
    f"{ROOT}/V3_carrier_2_cart_12",
    f"{ROOT}/V3_carrier_2_cart_13",
    f"{ROOT}/V3_carrier_2_cart_14",
    f"{ROOT}/V3_carrier_2_cart_15",
    f"{ROOT}/V3_carrier_2_cart_16",
    f"{ROOT}/V3_carrier_2_base_19",
    f"{ROOT}/V3_carrier_2_post_20",
    f"{ROOT}/V3_carrier_2_clamp_21",
    f"{ROOT}/V3_carrier_2_setscrew_17",
    f"{ROOT}/V3_carrier_2_setscrew_18",
    f"{ROOT}/V3_estimated_fastener_2",
    f"{ROOT}/V3_estimated_fastener_2_001",
    f"{ROOT}/V3_estimated_fastener_2_002",
    f"{ROOT}/V3_estimated_fastener_2_003",
}

ALL_MOVERS = OUTGOING | INCOMING | QUEUED


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_transport_groups():
    try:
        module = importlib.import_module("fluidmix.rigid_vial_transport")
        return module.resolve_transport_groups
    except (ModuleNotFoundError, AttributeError) as exc:
        pytest.fail(f"resolve_transport_groups contract is missing: {exc}")


def _load_rigid_transport_inputs():
    try:
        module = importlib.import_module("fluidmix.rigid_vial_transport")
        return module.load_rigid_transport_inputs
    except (ModuleNotFoundError, AttributeError) as exc:
        pytest.fail(f"load_rigid_transport_inputs contract is missing: {exc}")


def _motion_contracts():
    module = importlib.import_module("fluidmix.rigid_vial_transport")
    try:
        return module.RigidTransportSpec, module.build_rigid_transport_plan
    except AttributeError as exc:
        pytest.fail(f"rigid transport motion contract is missing: {exc}")


def _validate_authored_transport_diff():
    module = importlib.import_module("fluidmix.rigid_vial_transport")
    try:
        return module.validate_authored_transport_diff
    except AttributeError as exc:
        pytest.fail(f"authored transport diff validator is missing: {exc}")


def _validate_rigid_transport_receipt():
    module = importlib.import_module("fluidmix.rigid_vial_transport")
    try:
        return module.validate_rigid_transport_receipt
    except AttributeError as exc:
        pytest.fail(f"rigid transport receipt validator is missing: {exc}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_transport_input_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage_path: Path | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    reviewed_stage = tmp_path / "reviewed-source/fill_station.usda"
    reviewed_geometry = reviewed_stage.parent / "geometry/fill_station.usdc"
    reviewed_geometry.parent.mkdir(parents=True)
    reviewed_stage.write_bytes(b"#usda 1.0\n# portable rigid transport fixture\n")
    reviewed_geometry.write_bytes(b"portable-usdc-rigid-transport-fixture")
    source_hashes = {
        "fill_station.usda": _sha256(reviewed_stage),
        "geometry/fill_station.usdc": _sha256(reviewed_geometry),
    }
    source_bytes = {
        "fill_station.usda": reviewed_stage.stat().st_size,
        "geometry/fill_station.usdc": reviewed_geometry.stat().st_size,
    }
    module = importlib.import_module("fluidmix.rigid_vial_transport")
    monkeypatch.setattr(module, "SOURCE_LAYER_HASHES", source_hashes)
    monkeypatch.setattr(module, "SOURCE_LAYER_BYTES", source_bytes)
    if stage_path is None:
        stage_path = reviewed_stage
    case_output = tmp_path / "case"
    snapshot_dir = case_output / "particle-snapshots" / "step-02702-final"
    snapshot_dir.mkdir(parents=True)
    arrays = {
        "simulation_points": np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        "positions": np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))),
        "velocities": np.zeros((2, 3)),
        "orientations": np.asarray(((1.0, 0.0, 0.0, 0.0),) * 2),
        "scales": np.ones((2, 3)),
        "ids": np.asarray((0, 1), dtype=np.int64),
        "proto_indices": np.asarray((0, 1), dtype=np.int64),
        "scheduled_sources": np.asarray(("t200", "t100")),
        "birth_compositions": np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        "current_compositions": np.asarray(((0.75, 0.0, 0.25), (0.25, 0.0, 0.75))),
        "birth_steps": np.asarray((10, 20), dtype=np.int64),
        "birth_event_ids": np.asarray(
            (
                "pour-01-t200-segment-01-current_tank-t200-event-0001",
                "pour-02-t100-segment-01-current_tank-t100-event-0001",
            )
        ),
        "pour_event_ids": np.asarray(("pour-01-t200", "pour-02-t100")),
    }
    array_records = {}
    for name, values in arrays.items():
        path = snapshot_dir / f"{name}.npy"
        with path.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        array_records[name] = {
            "path": f"particle-snapshots/step-02702-final/{name}.npy",
            "sha256": _sha256(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "finite": bool(values.dtype.kind not in "fc" or np.all(np.isfinite(values))),
        }
    snapshot_manifest = {
        "event_id": "step-02702-final",
        "arrays_saved_before_assertions": True,
        "arrays": array_records,
    }
    snapshot_path = snapshot_dir / "snapshot.json"
    _write_json(snapshot_path, snapshot_manifest)
    snapshot_receipt = {
        **snapshot_manifest,
        "snapshot_path": "particle-snapshots/step-02702-final/snapshot.json",
        "snapshot_sha256": _sha256(snapshot_path),
    }

    video_path = case_output / "video.mp4"
    video_path.write_bytes(b"synthetic-video-fixture")
    receipt = {
        "schema_version": 20,
        "experiment": "physx_three_pour_f2_paint_ball_emitter",
        "status": "pass",
        "scene_contract": {
            "source_layers_unchanged": True,
            "source_layer_records": [
                {
                    "identifier": REMOTE_STAGE,
                    "resolved_path": REMOTE_STAGE,
                    "bytes": source_bytes["fill_station.usda"],
                    "sha256": source_hashes["fill_station.usda"],
                },
                {
                    "identifier": REMOTE_GEOMETRY,
                    "resolved_path": REMOTE_GEOMETRY,
                    "bytes": source_bytes["geometry/fill_station.usdc"],
                    "sha256": source_hashes["geometry/fill_station.usdc"],
                },
            ],
        },
        "render": {
            "video_path": "video.mp4",
            "video_sha256": _sha256(video_path),
            "video_decodable": True,
            "video_decoded_frame_count": 42,
        },
        "timing": {"primary_final_capture_step": 2702},
        "captures": [
            {
                "physics_step": 2702,
                "phase": "final",
                "particle_snapshot": snapshot_receipt,
            }
        ],
    }
    _write_json(case_output / "receipt.json", receipt)
    return case_output, stage_path, receipt


def _replace_snapshot_array(
    case_output: Path,
    receipt: dict[str, object],
    name: str,
    values: np.ndarray,
) -> None:
    snapshot = receipt["captures"][0]["particle_snapshot"]
    record = snapshot["arrays"][name]
    array_path = case_output / record["path"]
    with array_path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    record.update(
        {
            "sha256": _sha256(array_path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "finite": bool(
                values.dtype.kind not in "fc" or np.all(np.isfinite(values))
            ),
        }
    )
    snapshot_path = case_output / snapshot["snapshot_path"]
    _write_json(
        snapshot_path,
        {
            "event_id": snapshot["event_id"],
            "arrays_saved_before_assertions": True,
            "arrays": snapshot["arrays"],
        },
    )
    snapshot["snapshot_sha256"] = _sha256(snapshot_path)
    _write_json(case_output / "receipt.json", receipt)


def test_exact_reviewed_movers_resolve_to_three_sorted_rigid_groups() -> None:
    resolve_transport_groups = _resolve_transport_groups()

    groups = resolve_transport_groups(
        ALL_MOVERS | {f"{ROOT}/FillCam"}, SOURCE_HASHES
    )

    assert groups == {
        "outgoing": tuple(sorted(OUTGOING)),
        "incoming": tuple(sorted(INCOMING)),
        "queued": tuple(sorted(QUEUED)),
    }


@pytest.mark.parametrize(
    "prim_paths",
    [
        ALL_MOVERS - {f"{ROOT}/V3_carrier_0_cart_00"},
        ALL_MOVERS | {f"{ROOT}/V3_carrier_0_cart_17"},
        (ALL_MOVERS - {f"{ROOT}/Fill_vial_neighbour_1"})
        | {f"{ROOT}/Fill_vial_neighbor_1"},
        ALL_MOVERS | {f"{ROOT}/V3_carrier_1_cart_00/Mesh"},
    ],
    ids=("missing", "extra", "renamed", "non-transform-child"),
)
def test_managed_mover_inventory_drift_fails_closed(prim_paths: set[str]) -> None:
    resolve_transport_groups = _resolve_transport_groups()

    with pytest.raises(ValueError, match="prim inventory"):
        resolve_transport_groups(prim_paths, SOURCE_HASHES)


def test_wrong_source_hash_fails_before_group_resolution() -> None:
    resolve_transport_groups = _resolve_transport_groups()
    wrong = dict(SOURCE_HASHES)
    wrong["geometry/fill_station.usdc"] = "0" * 64

    with pytest.raises(ValueError, match="source layer"):
        resolve_transport_groups(ALL_MOVERS, wrong)


def test_portable_loader_fixture_uses_only_temporary_source_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch=monkeypatch
    )

    assert stage_path.is_relative_to(tmp_path)
    assert stage_path.stat().st_size < 1024
    assert (stage_path.parent / "geometry/fill_station.usdc").stat().st_size < 1024
    loaded = _load_rigid_transport_inputs()(case_output, stage_path)
    assert loaded["receipt"] == receipt


@pytest.mark.skipif(
    not LOCAL_STAGE.is_file()
    or not (LOCAL_STAGE.parent / "geometry/fill_station.usdc").is_file()
    or not (LOCAL_ACCEPTED_CASE / "receipt.json").is_file(),
    reason="optional reviewed local source/case integration inputs are unavailable",
)
def test_optional_reviewed_local_source_and_case_loader_integration() -> None:
    loaded = _load_rigid_transport_inputs()(LOCAL_ACCEPTED_CASE, LOCAL_STAGE)

    assert loaded["stage_sha256"] == SOURCE_HASHES["fill_station.usda"]
    assert loaded["source_layer_hashes"] == SOURCE_HASHES
    assert loaded["final_physics_step"] == 2702
    assert len(loaded["arrays"]["ids"]) == 3548


def test_loader_hash_checks_video_final_snapshot_and_every_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    load_rigid_transport_inputs = _load_rigid_transport_inputs()

    loaded = load_rigid_transport_inputs(case_output, stage_path)

    assert loaded["receipt"] == receipt
    assert loaded["final_physics_step"] == 2702
    assert loaded["video_path"] == case_output / "video.mp4"
    assert loaded["snapshot_path"] == (
        case_output / "particle-snapshots/step-02702-final/snapshot.json"
    )
    assert set(loaded["arrays"]) == {
        "simulation_points",
        "positions",
        "velocities",
        "orientations",
        "scales",
        "ids",
        "proto_indices",
        "scheduled_sources",
        "birth_compositions",
        "current_compositions",
        "birth_steps",
        "birth_event_ids",
        "pour_event_ids",
    }
    assert loaded["arrays"]["ids"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="read-only"):
        loaded["arrays"]["positions"][0, 0] = 99.0


def test_loader_rejects_a_hash_mismatched_snapshot_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_output, stage_path, _receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    load_rigid_transport_inputs = _load_rigid_transport_inputs()
    positions = case_output / "particle-snapshots/step-02702-final/positions.npy"
    positions.write_bytes(positions.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="array hash"):
        load_rigid_transport_inputs(case_output, stage_path)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("ids", np.asarray((0.0, 1.0), dtype=np.float64)),
        ("scheduled_sources", np.asarray((1, 2), dtype=np.int64)),
        ("positions", np.ones((2, 2), dtype=np.float64)),
        ("proto_indices", np.asarray((0, -1), dtype=np.int64)),
        ("birth_steps", np.asarray((10, -1), dtype=np.int64)),
        ("scales", np.asarray(((1.0, 1.0, 1.0), (1.0, 0.0, 1.0)))),
        ("scheduled_sources", np.asarray(("t200", "t999"))),
        ("birth_event_ids", np.asarray(("birth-0", ""))),
        ("pour_event_ids", np.asarray(("pour-0", "bad"))),
        (
            "birth_compositions",
            np.asarray(((-0.1, 0.1, 1.0), (0.0, 0.0, 1.0))),
        ),
        (
            "current_compositions",
            np.asarray(((-0.1, 0.1, 1.0), (0.25, 0.0, 0.75))),
        ),
    ],
    ids=(
        "ids-dtype",
        "string-dtype",
        "vector-shape",
        "negative-prototype",
        "negative-birth-step",
        "zero-scale",
        "unknown-source",
        "empty-birth-event",
        "malformed-pour-event",
        "negative-birth-composition",
        "negative-current-composition",
    ),
)
def test_loader_rejects_self_consistent_but_invalid_snapshot_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    values: np.ndarray,
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    _replace_snapshot_array(case_output, receipt, name, values)

    with pytest.raises(ValueError, match="snapshot array|composition"):
        _load_rigid_transport_inputs()(case_output, stage_path)


def test_loader_rejects_wrong_local_usda_or_usdc_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_rigid_transport_inputs = _load_rigid_transport_inputs()
    case_output, valid_stage, _receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    valid_geometry = valid_stage.parent / "geometry/fill_station.usdc"
    wrong_usda = tmp_path / "wrong-usda" / "fill_station.usda"
    (wrong_usda.parent / "geometry").mkdir(parents=True)
    wrong_usda.write_bytes(b"not-the-reviewed-usda")
    (wrong_usda.parent / "geometry/fill_station.usdc").write_bytes(
        valid_geometry.read_bytes()
    )
    with pytest.raises(ValueError, match="local source layer"):
        load_rigid_transport_inputs(case_output, wrong_usda)

    wrong_usdc = tmp_path / "wrong-usdc" / "fill_station.usda"
    (wrong_usdc.parent / "geometry").mkdir(parents=True)
    wrong_usdc.write_bytes(valid_stage.read_bytes())
    (wrong_usdc.parent / "geometry/fill_station.usdc").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="local source layer"):
        load_rigid_transport_inputs(case_output, wrong_usdc)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", "0" * 64),
        ("bytes", 905),
    ],
)
def test_loader_rejects_wrong_receipt_source_hash_or_byte_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    receipt["scene_contract"]["source_layer_records"][0][field] = value
    _write_json(case_output / "receipt.json", receipt)

    with pytest.raises(ValueError, match="source layer hash binding"):
        _load_rigid_transport_inputs()(case_output, stage_path)


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate"))
def test_loader_rejects_wrong_receipt_source_record_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    records = receipt["scene_contract"]["source_layer_records"]
    if mutation == "missing":
        records.pop()
    elif mutation == "extra":
        records.append(
            {
                "identifier": "/remote/extra.usdc",
                "resolved_path": "/remote/extra.usdc",
                "bytes": 1,
                "sha256": "e" * 64,
            }
        )
    else:
        records[1] = copy.deepcopy(records[0])
    _write_json(case_output / "receipt.json", receipt)

    with pytest.raises(ValueError, match="source layer"):
        _load_rigid_transport_inputs()(case_output, stage_path)


def test_loader_rejects_float_source_byte_count_that_compares_equal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    record = receipt["scene_contract"]["source_layer_records"][0]
    record["bytes"] = float(record["bytes"])
    _write_json(case_output / "receipt.json", receipt)

    with pytest.raises(ValueError, match="hash binding"):
        _load_rigid_transport_inputs()(case_output, stage_path)


def test_loader_requires_identifier_and_resolved_path_to_name_the_same_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    receipt["scene_contract"]["source_layer_records"][0]["identifier"] = (
        "/remote/geometry/fill_station.usdc"
    )
    _write_json(case_output / "receipt.json", receipt)

    with pytest.raises(ValueError, match="membership"):
        _load_rigid_transport_inputs()(case_output, stage_path)


def test_loader_rejects_non_integer_snapshot_manifest_shape_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    snapshot = receipt["captures"][0]["particle_snapshot"]
    snapshot["arrays"]["ids"]["shape"] = [2.0]
    snapshot_path = case_output / snapshot["snapshot_path"]
    _write_json(
        snapshot_path,
        {
            "event_id": snapshot["event_id"],
            "arrays_saved_before_assertions": True,
            "arrays": snapshot["arrays"],
        },
    )
    snapshot["snapshot_sha256"] = _sha256(snapshot_path)
    _write_json(case_output / "receipt.json", receipt)

    with pytest.raises(ValueError, match="manifest"):
        _load_rigid_transport_inputs()(case_output, stage_path)


def test_loader_rejects_float_final_capture_step_that_compares_equal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    receipt["captures"][0]["physics_step"] = 2702.0
    _write_json(case_output / "receipt.json", receipt)

    with pytest.raises(ValueError, match="final capture"):
        _load_rigid_transport_inputs()(case_output, stage_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 15),
        ("schema_version", 20.0),
        ("experiment", "physx_three_pour_f2"),
    ],
)
def test_loader_rejects_wrong_accepted_receipt_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path, monkeypatch
    )
    receipt[field] = value
    _write_json(case_output / "receipt.json", receipt)

    with pytest.raises(ValueError, match="identity"):
        _load_rigid_transport_inputs()(case_output, stage_path)


def test_loader_receipt_symlink_must_resolve_inside_case_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_rigid_transport_inputs = _load_rigid_transport_inputs()
    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path / "escape", monkeypatch
    )
    receipt_path = case_output / "receipt.json"
    outside = tmp_path / "outside-receipt.json"
    outside.write_bytes(receipt_path.read_bytes())
    receipt_path.unlink()
    receipt_path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the accepted case"):
        load_rigid_transport_inputs(case_output, stage_path)

    case_output, stage_path, receipt = _write_transport_input_fixture(
        tmp_path / "contained", monkeypatch
    )
    receipt_path = case_output / "receipt.json"
    contained = case_output / "manifests/accepted-receipt.json"
    contained.parent.mkdir()
    contained.write_bytes(receipt_path.read_bytes())
    receipt_path.unlink()
    receipt_path.symlink_to(contained)

    loaded = load_rigid_transport_inputs(case_output, stage_path)
    assert loaded["receipt"] == receipt
    assert loaded["receipt_path"] == contained.resolve()


def test_motion_plan_uses_one_smoothstep_delta_per_exact_group() -> None:
    RigidTransportSpec, build_rigid_transport_plan = _motion_contracts()
    groups = {
        "outgoing": tuple(sorted(OUTGOING)),
        "incoming": tuple(sorted(INCOMING)),
        "queued": tuple(sorted(QUEUED)),
    }
    anchor_y = {
        f"{ROOT}/Fill_vial_active": -1.9811999797821,
        f"{ROOT}/Fill_vial_neighbour_1": -2.0660413117645885,
        f"{ROOT}/Fill_vial_neighbour_2": -2.25248264251923,
    }

    plan = build_rigid_transport_plan(groups, anchor_y, RigidTransportSpec())

    assert len(plan) == 21
    assert plan[0]["frame_index"] == 0
    assert [row["source_time_code"] for row in plan] == [1] * 21
    assert plan[0]["time_s"] == 0.0
    assert plan[0]["smoothstep"] == 0.0
    assert set(plan[0]["path_delta_y_m"].values()) == {0.0}
    assert plan[10]["smoothstep"] == pytest.approx(0.5, abs=1.0e-15)
    assert plan[-1]["group_delta_y_m"] == {
        "outgoing": 0.19,
        "incoming": 0.0848413319824885,
        "queued": 0.0,
    }
    assert plan[-1]["group_delta_xyz_m"] == {
        "outgoing": [0.0, 0.19, 0.0],
        "incoming": [0.0, 0.0848413319824885, 0.0],
        "queued": [0.0, 0.0, 0.0],
    }
    assert plan[-1]["path_delta_xyz_m"][f"{ROOT}/Fill_vial_active"] == [
        0.0,
        0.19,
        0.0,
    ]
    for frame in plan:
        assert {
            frame["path_delta_y_m"][path] for path in groups["outgoing"]
        } == {frame["group_delta_y_m"]["outgoing"]}
        assert {
            frame["path_delta_y_m"][path] for path in groups["incoming"]
        } == {frame["group_delta_y_m"]["incoming"]}
        assert {
            frame["path_delta_y_m"][path] for path in groups["queued"]
        } == {0.0}
    assert all(
        before["group_delta_y_m"]["outgoing"]
        <= after["group_delta_y_m"]["outgoing"]
        for before, after in zip(plan, plan[1:])
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_count": 20},
        {"frame_count": True},
        {"frame_count": 21.0},
        {"fps": 10},
        {"fps": 0.0},
        {"fps": float("nan")},
        {"outgoing_final_delta_y_m": 1},
        {"outgoing_final_delta_y_m": 0.189},
        {"incoming_final_delta_y_m": 0.08},
    ],
)
def test_transport_spec_rejects_values_outside_the_reviewed_contract(
    kwargs: dict[str, object],
) -> None:
    RigidTransportSpec, _build_rigid_transport_plan = _motion_contracts()

    with pytest.raises(ValueError):
        RigidTransportSpec(**kwargs)


@pytest.mark.parametrize(
    "anchor_y",
    [
        {
            f"{ROOT}/Fill_vial_active": -2.1,
            f"{ROOT}/Fill_vial_neighbour_1": -2.0,
            f"{ROOT}/Fill_vial_neighbour_2": -2.2,
        },
        {
            f"{ROOT}/Fill_vial_active": -1.9811999797821,
            f"{ROOT}/Fill_vial_neighbour_1": float("nan"),
            f"{ROOT}/Fill_vial_neighbour_2": -2.25248264251923,
        },
        {
            f"{ROOT}/Fill_vial_active": -1.9811999797821,
            f"{ROOT}/Fill_vial_neighbour_1": -2.07,
            f"{ROOT}/Fill_vial_neighbour_2": -2.25248264251923,
        },
    ],
    ids=("crossed", "nan", "incoming-misses-active"),
)
def test_motion_plan_rejects_invalid_or_crossed_world_y_anchors(
    anchor_y: dict[str, float],
) -> None:
    RigidTransportSpec, build_rigid_transport_plan = _motion_contracts()
    groups = {
        "outgoing": tuple(sorted(OUTGOING)),
        "incoming": tuple(sorted(INCOMING)),
        "queued": tuple(sorted(QUEUED)),
    }

    with pytest.raises(ValueError, match="world-Y"):
        build_rigid_transport_plan(groups, anchor_y, RigidTransportSpec())


def _valid_plan_and_diff(outgoing_final_delta_y_m: float = 0.19):
    RigidTransportSpec, build_rigid_transport_plan = _motion_contracts()
    groups = {
        "outgoing": tuple(sorted(OUTGOING)),
        "incoming": tuple(sorted(INCOMING)),
        "queued": tuple(sorted(QUEUED)),
    }
    plan = build_rigid_transport_plan(
        groups,
        {
            f"{ROOT}/Fill_vial_active": -1.9811999797821,
            f"{ROOT}/Fill_vial_neighbour_1": -2.0660413117645885,
            f"{ROOT}/Fill_vial_neighbour_2": -2.25248264251923,
        },
        RigidTransportSpec(outgoing_final_delta_y_m=outgoing_final_delta_y_m),
    )
    count_body = f"{ROOT}/FluidmixF2CountDrivenLiquid"
    count_mesh = f"{count_body}/Fill_liquid"
    reconstruction_properties = {
        f"{count_body}.xformOp:transform",
        f"{count_mesh}.doubleSided",
        f"{count_mesh}.extent",
        f"{count_mesh}.faceVaryingLinearInterpolation",
        f"{count_mesh}.faceVertexCounts",
        f"{count_mesh}.faceVertexIndices",
        f"{count_mesh}.interpolateBoundary",
        f"{count_mesh}.material:binding",
        f"{count_mesh}.orientation",
        f"{count_mesh}.points",
        f"{count_mesh}.primvars:displayColor",
        f"{count_mesh}.semantics:labels:class",
        f"{count_mesh}.subdivisionScheme",
        f"{count_mesh}.triangleSubdivisionRule",
        f"{count_mesh}.visibility",
        f"{count_mesh}.xformOp:transform",
        f"{count_mesh}.xformOpOrder",
    }
    transport_properties = {
        property_path
        for path in ALL_MOVERS
        for property_path in (
            f"{path}.xformOp:translate:fluidmixTransport",
            f"{path}.xformOpOrder",
        )
    }
    source_movers = ALL_MOVERS - {count_body}
    existing_op_orders = {
        path: ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
        for path in source_movers
    }
    existing_op_orders[count_body] = ["xformOp:transform"]
    authored_op_orders = {
        path: ["xformOp:translate:fluidmixTransport", *order]
        for path, order in existing_op_orders.items()
    }
    authored_local_xyz = {
        path: [0.0, delta_y / -0.02539999969303608, 0.0]
        for path, delta_y in plan[20]["path_delta_y_m"].items()
    }
    measured_world_xyz = {
        path: [0.0, delta_y, 0.0]
        for path, delta_y in plan[20]["path_delta_y_m"].items()
    }
    diff = {
        "edit_target": "anonymous_session_layer",
        "source_time_code": 1,
        "source_layer_property_paths": [],
        "root_layer_property_paths": [],
        "session_prim_paths": sorted(
            {"/FillStation", ROOT, *ALL_MOVERS, count_mesh}
        ),
        "count_body_reconstruction_prim_paths": [count_body, count_mesh],
        "session_property_paths": sorted(
            transport_properties | reconstruction_properties
        ),
        "count_body_reconstruction_property_paths": sorted(reconstruction_properties),
        "frame_index": 20,
        "authored_local_translate_xyz_by_path": authored_local_xyz,
        "measured_world_delta_xyz_by_path": measured_world_xyz,
        "existing_xform_op_order_by_path": existing_op_orders,
        "authored_xform_op_order_by_path": authored_op_orders,
    }
    return groups, plan, diff


def test_authored_diff_accepts_only_exact_session_transport_and_count_body() -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    groups, plan, diff = _valid_plan_and_diff()

    active_vial = f"{ROOT}/Fill_vial_active"
    rotated_carrier = f"{ROOT}/V3_carrier_0_cart_00"
    count_body = f"{ROOT}/FluidmixF2CountDrivenLiquid"
    for path in (active_vial, rotated_carrier, count_body):
        assert diff["authored_local_translate_xyz_by_path"][path] == [
            0.0,
            -7.480315051030978,
            0.0,
        ]
        assert diff["measured_world_delta_xyz_by_path"][path] == [0.0, 0.19, 0.0]
        assert diff["authored_xform_op_order_by_path"][path][0] == (
            "xformOp:translate:fluidmixTransport"
        )

    validate_authored_transport_diff(diff, groups, plan)

    unauthorized = dict(diff)
    unauthorized["session_property_paths"] = [
        *diff["session_property_paths"],
        f"{ROOT}/FillCam.xformOp:translate",
    ]
    with pytest.raises(ValueError, match="unauthorized session"):
        validate_authored_transport_diff(unauthorized, groups, plan)


def test_parent_local_basis_contract_is_bound_to_time_one_and_source_hashes() -> None:
    module = importlib.import_module("fluidmix.rigid_vial_transport")

    assert dict(module.ROOT_LINEAR_BASIS_SOURCE_HASHES) == SOURCE_HASHES
    assert module.ROOT_LINEAR_BASIS_TIME_CODE == 1
    assert module.ROOT_LINEAR_BASIS_WORLD_M_PER_LOCAL_UNIT == (
        (-0.02539999969303608, 0.0, 0.0),
        (0.0, -0.02539999969303608, 0.0),
        (0.0, 0.0, 0.02539999969303608),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unknown", True, "top-level schema"),
        ("source_layer_property_paths", ["/FillStation.sourceEdit"], "source layer"),
        ("root_layer_property_paths", ["/FillStation.rootEdit"], "root layer"),
        (
            "session_prim_paths",
            [*sorted({"/FillStation", ROOT, *ALL_MOVERS, f"{ROOT}/FluidmixF2CountDrivenLiquid/Fill_liquid"}), f"{ROOT}/Extra"],
            "session prim",
        ),
        (
            "count_body_reconstruction_prim_paths",
            [
                f"{ROOT}/FluidmixF2CountDrivenLiquid",
                f"{ROOT}/FluidmixF2CountDrivenLiquid/Fill_liquid",
                f"{ROOT}/FluidmixF2CountDrivenLiquid/Extra",
            ],
            "count-body reconstruction prim",
        ),
    ],
    ids=("unknown-field", "source-edit", "root-edit", "extra-prim", "extra-body-prim"),
)
def test_authored_diff_rejects_schema_layer_or_prim_inventory_bypass(
    field: str, value: object, message: str
) -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    groups, plan, diff = _valid_plan_and_diff()
    diff[field] = value

    with pytest.raises(ValueError, match=message):
        validate_authored_transport_diff(diff, groups, plan)


@pytest.mark.parametrize(
    ("axis", "value"),
    ((0, 0.01), (1, -7.4), (2, -0.01)),
)
def test_authored_diff_rejects_wrong_parent_local_transport_component(
    axis: int, value: float
) -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    groups, plan, diff = _valid_plan_and_diff()
    path = f"{ROOT}/Fill_vial_active"
    diff["authored_local_translate_xyz_by_path"][path][axis] = value

    with pytest.raises(ValueError, match="parent-local transport XYZ"):
        validate_authored_transport_diff(diff, groups, plan)


@pytest.mark.parametrize(
    ("path", "axis"),
    (
        (f"{ROOT}/Fill_vial_active", 0),
        (f"{ROOT}/V3_carrier_0_cart_00", 1),
        (f"{ROOT}/FluidmixF2CountDrivenLiquid", 2),
    ),
)
def test_authored_diff_rejects_measured_world_delta_beyond_tolerance(
    path: str, axis: int
) -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    groups, plan, diff = _valid_plan_and_diff()
    diff["measured_world_delta_xyz_by_path"][path][axis] += 2.0e-12

    with pytest.raises(ValueError, match="measured world transport XYZ"):
        validate_authored_transport_diff(diff, groups, plan)


@pytest.mark.parametrize("mutation", ("remove", "replace", "most-local"))
def test_authored_diff_preserves_existing_ops_after_least_local_transport(
    mutation: str,
) -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    groups, plan, diff = _valid_plan_and_diff()
    path = f"{ROOT}/Fill_vial_active"
    if mutation == "remove":
        diff["authored_xform_op_order_by_path"][path].pop(1)
    elif mutation == "replace":
        diff["authored_xform_op_order_by_path"][path][1] = "xformOp:rotateZ"
    else:
        for mover, existing in diff["existing_xform_op_order_by_path"].items():
            diff["authored_xform_op_order_by_path"][mover] = [
                *existing,
                "xformOp:translate:fluidmixTransport",
            ]

    with pytest.raises(ValueError, match="op order"):
        validate_authored_transport_diff(diff, groups, plan)


def test_authored_diff_rejects_a_non_monotone_motion_plan_row() -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    groups, plan, diff = _valid_plan_and_diff()
    non_monotone = copy.deepcopy(plan)
    non_monotone[10]["group_delta_y_m"]["outgoing"] = -0.01
    for path in groups["outgoing"]:
        non_monotone[10]["path_delta_y_m"][path] = -0.01

    with pytest.raises(ValueError, match="strict transport plan"):
        validate_authored_transport_diff(diff, groups, non_monotone)


def test_authored_diff_rejects_wrong_plan_time_even_when_motion_is_unchanged() -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    groups, plan, diff = _valid_plan_and_diff()
    wrong_time = copy.deepcopy(plan)
    wrong_time[10]["time_s"] = 1.01

    with pytest.raises(ValueError, match="strict transport plan"):
        validate_authored_transport_diff(diff, groups, wrong_time)


@pytest.mark.parametrize(
    "mutation", ("all-zero", "linear", "wrong-endpoint", "nan", "reordered")
)
def test_authored_diff_rejects_plan_that_is_not_exact_frozen_smoothstep(
    mutation: str,
) -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    groups, plan, diff = _valid_plan_and_diff()
    changed = list(copy.deepcopy(plan))
    if mutation == "all-zero":
        for row in changed:
            row["group_delta_y_m"] = {
                "outgoing": 0.0,
                "incoming": 0.0,
                "queued": 0.0,
            }
            row["path_delta_y_m"] = {path: 0.0 for path in ALL_MOVERS}
            row["group_delta_xyz_m"] = {
                name: [0.0, 0.0, 0.0] for name in ("outgoing", "incoming", "queued")
            }
            row["path_delta_xyz_m"] = {
                path: [0.0, 0.0, 0.0] for path in ALL_MOVERS
            }
    elif mutation == "linear":
        changed[5]["smoothstep"] = 0.25
    elif mutation == "wrong-endpoint":
        changed[-1]["group_delta_y_m"]["outgoing"] = 0.18
    elif mutation == "nan":
        changed[5]["group_delta_y_m"]["incoming"] = float("nan")
    else:
        changed[5], changed[6] = changed[6], changed[5]

    with pytest.raises(ValueError, match="strict transport plan"):
        validate_authored_transport_diff(diff, groups, changed)


def _valid_transport_receipt(outgoing_final_delta_y_m: float = 0.19):
    groups, plan, _diff = _valid_plan_and_diff(outgoing_final_delta_y_m)
    expected_inputs = {
        "stage_sha256": SOURCE_HASHES["fill_station.usda"],
        "source_layer_hashes": SOURCE_HASHES,
        "receipt_sha256": "a" * 64,
        "video_sha256": "b" * 64,
        "snapshot_sha256": "c" * 64,
        "final_physics_step": 2702,
        "authored_diff_sha256": "d" * 64,
    }
    frames = [
        {
            "frame_index": row["frame_index"],
            "source_time_code": row["source_time_code"],
            "time_s": row["time_s"],
            "smoothstep": row["smoothstep"],
            "rgb_sha256": hashlib.sha256(
                f"transport-frame-{row['frame_index']:04d}".encode()
            ).hexdigest(),
            "group_delta_y_m": dict(row["group_delta_y_m"]),
            "path_delta_y_m": dict(row["path_delta_y_m"]),
            "group_delta_xyz_m": copy.deepcopy(row["group_delta_xyz_m"]),
            "path_delta_xyz_m": copy.deepcopy(row["path_delta_xyz_m"]),
        }
        for row in plan
    ]
    receipt = {
        "schema_version": 1,
        "artifact_type": "rigid_vial_transport_receipt",
        "status": "pass",
        "part": "A",
        "excluded_from_model_inputs": True,
        "spec": {
            "frame_count": 21,
            "fps": 10.0,
            "outgoing_final_delta_y_m": outgoing_final_delta_y_m,
            "incoming_final_delta_y_m": 0.0848413319824885,
        },
        "bindings": {
            "stage_sha256": SOURCE_HASHES["fill_station.usda"],
            "source_layer_hashes": SOURCE_HASHES,
            "accepted_receipt_sha256": "a" * 64,
            "accepted_video_sha256": "b" * 64,
            "final_snapshot_sha256": "c" * 64,
            "final_physics_step": 2702,
        },
        "transport_groups": {name: list(paths) for name, paths in groups.items()},
        "frames": frames,
        "authored_diff_sha256": "d" * 64,
        "counters": {
            "physics_steps_before": 2702,
            "physics_steps_after": 2702,
            "physics_steps_during_transport": 0,
            "direct_step_calls_before": 0,
            "direct_step_calls_after": 0,
            "direct_step_calls_during_transport": 0,
            "physics_callbacks_before": 2702,
            "physics_callbacks_after": 2702,
            "physics_callbacks_during_transport": 0,
            "simulation_time_s_before": 22.516666666666666,
            "simulation_time_s_after": 22.516666666666666,
            "timeline_time_s_before": 22.516666666666666,
            "timeline_time_s_after": 22.516666666666666,
            "timeline_playing_before": False,
            "timeline_playing_after": False,
            "source_time_code_before": 1,
            "source_time_code_after": 1,
        },
        "automatic_acceptance_gates": {
            "source_and_receipt_hashes_unchanged": True,
            "snapshot_binding": True,
            "session_diff_allowlisted": True,
            "zero_clock_and_physics_advancement": True,
            "outgoing_absent_final": True,
            "incoming_centered_final": True,
            "fixed_background_sentinel": True,
            "start_splice": True,
            "end_splice": True,
            "unique_frame_hashes": True,
        },
    }
    return groups, plan, expected_inputs, receipt


def test_complete_transport_receipt_binds_frames_groups_inputs_and_zero_steps() -> None:
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, expected_inputs, receipt = _valid_transport_receipt()

    validate_rigid_transport_receipt(receipt, expected_inputs, groups, plan)


def test_both_validators_carry_the_originating_nondefault_frozen_spec() -> None:
    RigidTransportSpec, _build_rigid_transport_plan = _motion_contracts()
    validate_authored_transport_diff = _validate_authored_transport_diff()
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    spec = RigidTransportSpec(outgoing_final_delta_y_m=0.20)
    groups, plan, diff = _valid_plan_and_diff(0.20)
    receipt_groups, receipt_plan, expected_inputs, receipt = _valid_transport_receipt(
        0.20
    )

    assert receipt["spec"] == {
        "frame_count": 21,
        "fps": 10.0,
        "outgoing_final_delta_y_m": 0.20,
        "incoming_final_delta_y_m": 0.0848413319824885,
    }
    validate_authored_transport_diff(diff, groups, plan, spec=spec)
    validate_rigid_transport_receipt(
        receipt,
        expected_inputs,
        receipt_groups,
        receipt_plan,
        spec=spec,
    )

    with pytest.raises(ValueError, match="strict transport plan"):
        validate_authored_transport_diff(diff, groups, plan)
    with pytest.raises(ValueError, match="strict transport plan"):
        validate_rigid_transport_receipt(
            receipt,
            expected_inputs,
            receipt_groups,
            receipt_plan,
        )


def test_transport_receipt_rejects_wrong_input_hash_or_group_binding() -> None:
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, expected_inputs, receipt = _valid_transport_receipt()
    wrong_hash = copy.deepcopy(receipt)
    wrong_hash["bindings"]["accepted_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance binding"):
        validate_rigid_transport_receipt(wrong_hash, expected_inputs, groups, plan)

    wrong_group = copy.deepcopy(receipt)
    wrong_group["transport_groups"]["queued"].pop()
    with pytest.raises(ValueError, match="group binding"):
        validate_rigid_transport_receipt(wrong_group, expected_inputs, groups, plan)


def test_transport_receipt_binds_independently_expected_authored_diff_hash() -> None:
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, expected_inputs, receipt = _valid_transport_receipt()
    receipt["authored_diff_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="authored diff binding"):
        validate_rigid_transport_receipt(receipt, expected_inputs, groups, plan)


def test_transport_receipt_revalidates_plan_instead_of_trusting_matching_frames() -> None:
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, expected_inputs, receipt = _valid_transport_receipt()
    invalid_plan = list(copy.deepcopy(plan))
    invalid_plan[10]["time_s"] = 1.01
    receipt["frames"][10]["time_s"] = 1.01

    with pytest.raises(ValueError, match="strict transport plan"):
        validate_rigid_transport_receipt(receipt, expected_inputs, groups, invalid_plan)


@pytest.mark.parametrize("bad_time", (False, 0, 180, 216))
def test_plan_diff_receipt_and_counters_all_freeze_source_time_code_one(
    bad_time: object,
) -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, diff = _valid_plan_and_diff()

    bad_diff = copy.deepcopy(diff)
    bad_diff["source_time_code"] = bad_time
    with pytest.raises(ValueError, match="source time code 1"):
        validate_authored_transport_diff(bad_diff, groups, plan)

    bad_plan = list(copy.deepcopy(plan))
    bad_plan[5]["source_time_code"] = bad_time
    with pytest.raises(ValueError, match="source time code 1"):
        validate_authored_transport_diff(diff, groups, bad_plan)

    groups, plan, expected_inputs, receipt = _valid_transport_receipt()
    bad_frame = copy.deepcopy(receipt)
    bad_frame["frames"][5]["source_time_code"] = bad_time
    with pytest.raises(ValueError, match="frame manifest"):
        validate_rigid_transport_receipt(bad_frame, expected_inputs, groups, plan)

    bad_counter = copy.deepcopy(receipt)
    bad_counter["counters"]["source_time_code_after"] = bad_time
    with pytest.raises(ValueError, match="source time code 1"):
        validate_rigid_transport_receipt(bad_counter, expected_inputs, groups, plan)


def test_authored_diff_rejects_equal_numeric_values_with_wrong_scalar_types() -> None:
    validate_authored_transport_diff = _validate_authored_transport_diff()
    groups, plan, diff = _valid_plan_and_diff()

    wrong_time_type = copy.deepcopy(diff)
    wrong_time_type["source_time_code"] = 1.0
    with pytest.raises(ValueError, match="source time code 1"):
        validate_authored_transport_diff(wrong_time_type, groups, plan)

    wrong_xyz_type = copy.deepcopy(diff)
    wrong_xyz_type["authored_local_translate_xyz_by_path"][next(iter(OUTGOING))][0] = 0
    with pytest.raises(ValueError, match="parent-local transport XYZ"):
        validate_authored_transport_diff(wrong_xyz_type, groups, plan)


def test_transport_receipt_rejects_duplicate_hashes_or_nonfinite_transforms() -> None:
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, expected_inputs, receipt = _valid_transport_receipt()
    duplicate = copy.deepcopy(receipt)
    duplicate["frames"][10]["rgb_sha256"] = duplicate["frames"][9]["rgb_sha256"]
    with pytest.raises(ValueError, match="frame manifest"):
        validate_rigid_transport_receipt(duplicate, expected_inputs, groups, plan)

    nonfinite = copy.deepcopy(receipt)
    nonfinite["frames"][10]["group_delta_y_m"]["incoming"] = float("nan")
    with pytest.raises(ValueError, match="frame manifest"):
        validate_rigid_transport_receipt(nonfinite, expected_inputs, groups, plan)


def test_transport_receipt_rejects_clock_or_physics_advancement() -> None:
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, expected_inputs, receipt = _valid_transport_receipt()
    advanced = copy.deepcopy(receipt)
    advanced["counters"]["physics_callbacks_after"] += 1

    with pytest.raises(ValueError, match="counter"):
        validate_rigid_transport_receipt(advanced, expected_inputs, groups, plan)


def test_transport_receipt_requires_every_automatic_gate_and_part_a_boundary() -> None:
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, expected_inputs, receipt = _valid_transport_receipt()
    missing_gate = copy.deepcopy(receipt)
    missing_gate["automatic_acceptance_gates"].pop("end_splice")
    with pytest.raises(ValueError, match="acceptance gates"):
        validate_rigid_transport_receipt(missing_gate, expected_inputs, groups, plan)

    model_input = copy.deepcopy(receipt)
    model_input["excluded_from_model_inputs"] = False
    with pytest.raises(ValueError, match="Part A"):
        validate_rigid_transport_receipt(model_input, expected_inputs, groups, plan)


def test_transport_receipt_rejects_malformed_unknown_top_level_fields() -> None:
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, expected_inputs, receipt = _valid_transport_receipt()
    malformed = copy.deepcopy(receipt)
    malformed["model_inputs"] = {"particle_positions": "forbidden"}

    with pytest.raises(ValueError, match="malformed"):
        validate_rigid_transport_receipt(malformed, expected_inputs, groups, plan)

    malformed_counter = copy.deepcopy(receipt)
    malformed_counter["counters"]["render_updates_during_transport"] = 1
    with pytest.raises(ValueError, match="counter contract"):
        validate_rigid_transport_receipt(
            malformed_counter, expected_inputs, groups, plan
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "bool-schema-version",
        "float-frame-count",
        "integer-fps",
        "bool-frame-index",
        "bool-frame-time",
        "bool-smoothstep",
        "bool-zero-counter",
        "negative-unchanged-counter",
        "integer-clock",
        "float-binding-step",
        "non-positive-final-step",
    ),
)
def test_transport_receipt_rejects_nested_scalar_type_and_range_bypasses(
    mutation: str,
) -> None:
    validate_rigid_transport_receipt = _validate_rigid_transport_receipt()
    groups, plan, expected_inputs, receipt = _valid_transport_receipt()
    if mutation == "bool-schema-version":
        receipt["schema_version"] = True
    elif mutation == "float-frame-count":
        receipt["spec"]["frame_count"] = 21.0
    elif mutation == "integer-fps":
        receipt["spec"]["fps"] = 10
    elif mutation == "bool-frame-index":
        receipt["frames"][0]["frame_index"] = False
    elif mutation == "bool-frame-time":
        receipt["frames"][0]["time_s"] = False
    elif mutation == "bool-smoothstep":
        receipt["frames"][0]["smoothstep"] = False
    elif mutation == "bool-zero-counter":
        receipt["counters"]["direct_step_calls_during_transport"] = False
    elif mutation == "negative-unchanged-counter":
        receipt["counters"]["direct_step_calls_before"] = -1
        receipt["counters"]["direct_step_calls_after"] = -1
    elif mutation == "integer-clock":
        receipt["counters"]["simulation_time_s_before"] = 22
        receipt["counters"]["simulation_time_s_after"] = 22
    elif mutation == "float-binding-step":
        receipt["bindings"]["final_physics_step"] = 2702.0
    else:
        expected_inputs["final_physics_step"] = 0
        receipt["bindings"]["final_physics_step"] = 0
        receipt["counters"]["physics_steps_before"] = 0
        receipt["counters"]["physics_steps_after"] = 0

    with pytest.raises(ValueError):
        validate_rigid_transport_receipt(receipt, expected_inputs, groups, plan)


def test_renderer_and_checker_import_without_isaac_side_effects() -> None:
    before = set(sys.modules)

    renderer = _load_script("render_rigid_vial_transport")
    checker = _load_script("check_rigid_vial_transport")

    assert callable(renderer.run_transport)
    assert callable(checker.check_transport_output)
    newly_loaded = set(sys.modules) - before
    assert not any(
        name == "omni"
        or name.startswith("omni.")
        or name == "pxr"
        or name.startswith("pxr.")
        or name == "isaacsim"
        or name.startswith("isaacsim.")
        for name in newly_loaded
    )


def test_renderer_cli_requires_paths_and_exactly_one_capture_mode() -> None:
    renderer = _load_script("render_rigid_vial_transport")
    required = ["--stage", "stage.usda", "--case-output", "case", "--output-dir", "out"]

    with pytest.raises(SystemExit):
        renderer.parse_args(required)
    with pytest.raises(SystemExit):
        renderer.parse_args(required + ["--pilot-frames", "0,10,20", "--all-frames"])
    pilot = renderer.parse_args(required + ["--pilot-frames", "0,10,20"])
    full = renderer.parse_args(required + ["--all-frames"])

    assert pilot.frame_indices == (0, 10, 20)
    assert full.frame_indices == tuple(range(21))
    with pytest.raises(SystemExit):
        renderer.parse_args(required + ["--pilot-frames", "0,20"])


def _frame_diff(frame_index: int) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    groups, plan, final_diff = _valid_plan_and_diff()
    diff = copy.deepcopy(final_diff)
    diff["frame_index"] = frame_index
    row = plan[frame_index]
    diff["authored_local_translate_xyz_by_path"] = {
        path: [0.0, delta / -0.02539999969303608, 0.0]
        for path, delta in row["path_delta_y_m"].items()
    }
    diff["measured_world_delta_xyz_by_path"] = copy.deepcopy(
        row["path_delta_xyz_m"]
    )
    return diff, plan


class _FakeTransportRuntime:
    def __init__(self, *, readback_mutation: tuple[str, object] | None = None) -> None:
        self.groups, self.plan, _diff = _valid_plan_and_diff()
        self.expected = {
            "isaac_sim_version": "6.2.0.0",
            "renderer": "RaytracedLighting",
            "resolution": [640, 480],
            "camera_path": f"{ROOT}/FillCam/FillCam",
            "root_linear_basis_world_m_per_local_unit": [
                [-0.02539999969303608, 0.0, 0.0],
                [0.0, -0.02539999969303608, 0.0],
                [0.0, 0.0, 0.02539999969303608],
            ],
            "light_state_sha256": "1" * 64,
            "material_state_sha256": "2" * 64,
            "tone_exposure": {
                "tone_mapping_operator": 4,
                "auto_exposure_enabled": False,
                "camera_exposure_compensation": 0.0,
                "camera_auto_exposure_enabled": False,
            },
        }
        self.readback = copy.deepcopy(self.expected)
        if readback_mutation is not None:
            key, value = readback_mutation
            self.readback[key] = value
        self.counter = {
            "physics_steps": 2702,
            "direct_step_calls": 0,
            "physics_callbacks": 2702,
            "simulation_time_s": 22.516666666666666,
            "timeline_time_s": 22.516666666666666,
            "timeline_playing": False,
            "source_time_code": 1,
        }
        self.captured: list[int] = []

    def preflight(self, _loaded: dict[str, object]) -> dict[str, object]:
        return {
            "readback": copy.deepcopy(self.readback),
            "prim_paths": sorted(ALL_MOVERS | {f"{ROOT}/FillCam"}),
            "groups": self.groups,
            "plan": self.plan,
            "counter_state": copy.deepcopy(self.counter),
        }

    def author_frame(self, frame_index: int) -> dict[str, object]:
        diff, _plan = _frame_diff(frame_index)
        return diff

    def capture_frame(self, frame_index: int) -> dict[str, object]:
        self.captured.append(frame_index)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[..., 0] = frame_index
        outgoing = np.zeros((480, 640), dtype=np.uint8)
        incoming = np.zeros((480, 640), dtype=np.uint8)
        if frame_index < 20:
            outgoing[200:220, 300 + frame_index : 310 + frame_index] = 1
        if frame_index > 0:
            incoming[200:220, 280 + frame_index : 290 + frame_index] = 1
        instance_ids = outgoing.astype(np.uint32)
        instance_ids[incoming.astype(bool)] = 2
        return {
            "rgb": rgb,
            "outgoing_mask": outgoing,
            "incoming_mask": incoming,
            "instance_ids": instance_ids,
            "instance_id_to_path": {
                "1": f"{ROOT}/Fill_vial_active",
                "2": f"{ROOT}/Fill_vial_neighbour_1",
            },
            "counter_state_before": copy.deepcopy(self.counter),
            "counter_state_after": copy.deepcopy(self.counter),
        }

    def counter_state(self) -> dict[str, object]:
        return copy.deepcopy(self.counter)


def _loaded_runtime_fixture(tmp_path: Path, runtime: _FakeTransportRuntime) -> dict[str, object]:
    stage = tmp_path / "fill_station.usda"
    stage.write_bytes(b"stage-bytes")
    case = tmp_path / "case"
    case.mkdir()
    accepted = case / "receipt.json"
    accepted.write_bytes(b'{"accepted":true}\n')
    return {
        "case_output": case,
        "stage_path": stage,
        "stage_sha256": _sha256(stage),
        "source_layer_hashes": SOURCE_HASHES,
        "receipt": {},
        "receipt_path": accepted,
        "receipt_sha256": _sha256(accepted),
        "video_sha256": "b" * 64,
        "snapshot_sha256": "c" * 64,
        "final_physics_step": 2702,
        "expected_runtime_readback": runtime.expected,
    }


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("isaac_sim_version", "6.0.1"),
        ("renderer", "PathTracing"),
        ("resolution", [320, 240]),
        ("camera_path", "/OtherCamera"),
        ("light_state_sha256", "3" * 64),
        ("material_state_sha256", "4" * 64),
        ("tone_exposure", {"tone_mapping_operator": 0}),
        ("root_linear_basis_world_m_per_local_unit", [[1.0, 0.0, 0.0]] * 3),
    ),
)
def test_renderer_rejects_runtime_preflight_before_creating_output(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    renderer = _load_script("render_rigid_vial_transport")
    runtime = _FakeTransportRuntime(readback_mutation=(field, bad_value))
    loaded = _loaded_runtime_fixture(tmp_path, runtime)
    output = tmp_path / "transport"

    with pytest.raises(ValueError, match="preflight"):
        renderer.run_transport(
            stage=loaded["stage_path"],
            case_output=loaded["case_output"],
            output_dir=output,
            frame_indices=(0, 10, 20),
            runtime=runtime,
            load_inputs=lambda _case, _stage: loaded,
            check_output=lambda **_kwargs: {"status": "pilot_pending"},
        )

    assert not output.exists()
    assert not output.with_name(output.name + ".partial").exists()
    assert loaded["receipt_path"].read_bytes() == b'{"accepted":true}\n'


def _fake_checker(**kwargs):
    indices = tuple(kwargs["expected_frame_indices"])
    gates = {
        "outgoing_absent_final": True,
        "incoming_centered_final": True,
        "fixed_background_sentinel": True,
        "start_splice": True,
        "end_splice": True if indices == tuple(range(21)) else "pending",
        "unique_frame_hashes": True if indices == tuple(range(21)) else "pending",
    }
    return {
        "status": "pass" if indices == tuple(range(21)) else "pilot_pending",
        "automatic_acceptance_gates": gates,
    }


def test_pilot_publication_is_atomic_pending_and_preserves_accepted_bytes(
    tmp_path: Path,
) -> None:
    renderer = _load_script("render_rigid_vial_transport")
    runtime = _FakeTransportRuntime()
    loaded = _loaded_runtime_fixture(tmp_path, runtime)
    accepted_before = loaded["receipt_path"].read_bytes()
    output = tmp_path / "transport"

    receipt = renderer.run_transport(
        stage=loaded["stage_path"],
        case_output=loaded["case_output"],
        output_dir=output,
        frame_indices=(0, 10, 20),
        runtime=runtime,
        load_inputs=lambda _case, _stage: loaded,
        check_output=_fake_checker,
    )

    assert receipt["status"] == "pilot_pending"
    assert receipt["automatic_acceptance_gates"]["end_splice"] == "pending"
    assert receipt["automatic_acceptance_gates"]["unique_frame_hashes"] == "pending"
    assert sorted(path.name for path in output.glob("frame-*.png")) == [
        "frame-0000.png",
        "frame-0010.png",
        "frame-0020.png",
    ]
    assert (output / "contact-sheet.png").is_file()
    assert not (output / "transport.mp4").exists()
    assert not output.with_name(output.name + ".partial").exists()
    assert loaded["receipt_path"].read_bytes() == accepted_before


@pytest.mark.parametrize("failure", ("counter", "unauthorized-diff", "capture"))
def test_failed_capture_removes_only_owned_partial_and_never_publishes(
    tmp_path: Path, failure: str
) -> None:
    renderer = _load_script("render_rigid_vial_transport")
    runtime = _FakeTransportRuntime()
    loaded = _loaded_runtime_fixture(tmp_path, runtime)
    accepted_before = loaded["receipt_path"].read_bytes()
    if failure == "counter":
        original = runtime.capture_frame

        def changed_counter(frame_index: int):
            value = original(frame_index)
            value["counter_state_after"]["physics_callbacks"] += 1
            return value

        runtime.capture_frame = changed_counter
    elif failure == "unauthorized-diff":
        original_author = runtime.author_frame

        def unauthorized(frame_index: int):
            value = original_author(frame_index)
            value["session_property_paths"].append(f"{ROOT}/FillCam.exposure")
            return value

        runtime.author_frame = unauthorized
    else:
        runtime.capture_frame = lambda _index: (_ for _ in ()).throw(
            RuntimeError("injected capture failure")
        )
    output = tmp_path / "transport"

    with pytest.raises((RuntimeError, ValueError)):
        renderer.run_transport(
            stage=loaded["stage_path"],
            case_output=loaded["case_output"],
            output_dir=output,
            frame_indices=(0, 10, 20),
            runtime=runtime,
            load_inputs=lambda _case, _stage: loaded,
            check_output=_fake_checker,
        )

    assert not output.exists()
    assert not output.with_name(output.name + ".partial").exists()
    assert loaded["receipt_path"].read_bytes() == accepted_before


@pytest.mark.parametrize("existing_name", ("transport", "transport.partial"))
def test_renderer_never_clobbers_existing_final_or_partial(
    tmp_path: Path, existing_name: str
) -> None:
    renderer = _load_script("render_rigid_vial_transport")
    runtime = _FakeTransportRuntime()
    loaded = _loaded_runtime_fixture(tmp_path, runtime)
    existing = tmp_path / existing_name
    existing.mkdir()
    marker = existing / "owner.txt"
    marker.write_text("not this invocation\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        renderer.run_transport(
            stage=loaded["stage_path"],
            case_output=loaded["case_output"],
            output_dir=tmp_path / "transport",
            frame_indices=(0, 10, 20),
            runtime=runtime,
            load_inputs=lambda _case, _stage: loaded,
            check_output=_fake_checker,
        )

    assert marker.read_text(encoding="utf-8") == "not this invocation\n"
    assert runtime.captured == []


def test_atomic_publication_does_not_replace_a_racing_final_directory(
    tmp_path: Path,
) -> None:
    renderer = _load_script("render_rigid_vial_transport")
    runtime = _FakeTransportRuntime()
    loaded = _loaded_runtime_fixture(tmp_path, runtime)
    output = tmp_path / "transport"

    def racing_checker(**kwargs):
        result = _fake_checker(**kwargs)
        if kwargs["evaluate_only"] is False:
            output.mkdir()
            (output / "owner.txt").write_text("racing owner\n", encoding="utf-8")
        return result

    with pytest.raises(OSError):
        renderer.run_transport(
            stage=loaded["stage_path"],
            case_output=loaded["case_output"],
            output_dir=output,
            frame_indices=(0, 10, 20),
            runtime=runtime,
            load_inputs=lambda _case, _stage: loaded,
            check_output=racing_checker,
        )

    assert (output / "owner.txt").read_text(encoding="utf-8") == "racing owner\n"
    assert not output.with_name(output.name + ".partial").exists()


def _write_metric_artifacts(root: Path, indices: tuple[int, ...]) -> None:
    import cv2

    root.mkdir()
    for index in indices:
        rgb = np.full((480, 640, 3), 10, dtype=np.uint8)
        outgoing = np.zeros((480, 640), dtype=np.uint8)
        incoming = np.zeros((480, 640), dtype=np.uint8)
        if index < 20:
            left = 300 + min(index, 10)
            outgoing[200:220, left : left + 10] = 1
            rgb[outgoing.astype(bool), 0] = 20 + index
        if index > 0:
            left = 500 - 9 * index
            if index == 20:
                left = 300
            incoming[200:220, left : left + 10] = 1
            rgb[incoming.astype(bool), 1] = 40 + index
        assert cv2.imwrite(
            str(root / f"frame-{index:04d}.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        np.save(root / f"outgoing-mask-{index:04d}.npy", outgoing, allow_pickle=False)
        np.save(root / f"incoming-mask-{index:04d}.npy", incoming, allow_pickle=False)
        instance_ids = outgoing.astype(np.uint32)
        instance_ids[incoming.astype(bool)] = 2
        np.save(root / f"instance-{index:04d}.npy", instance_ids, allow_pickle=False)
        _write_json(
            root / f"instance-mapping-{index:04d}.json",
            {
                "1": f"{ROOT}/Fill_vial_active",
                "2": f"{ROOT}/Fill_vial_neighbour_1",
            },
        )


def test_checker_recomputes_full_mask_center_sentinel_splice_and_unique_gates(
    tmp_path: Path,
) -> None:
    checker = _load_script("check_rigid_vial_transport")
    output = tmp_path / "full"
    _write_metric_artifacts(output, tuple(range(21)))
    accepted = (
        np.full((480, 640, 3), 9, dtype=np.uint8),
        np.full((480, 640, 3), 10, dtype=np.uint8),
    )

    result = checker.compute_automatic_gates(
        output, frame_indices=tuple(range(21)), accepted_video_frames=accepted
    )

    assert all(value is True for value in result["automatic_acceptance_gates"].values())
    assert result["metrics"]["outgoing_final_pixel_count"] == 0
    assert result["metrics"]["incoming_center_error_px"] == 0.0
    assert result["metrics"]["fixed_background_sentinel_max_delta"] == 0
    assert len(set(result["metrics"]["frame_hashes"])) == 21


def test_checker_rejects_ambiguous_or_malformed_masks_and_duplicate_frames(
    tmp_path: Path,
) -> None:
    checker = _load_script("check_rigid_vial_transport")
    accepted = (
        np.full((480, 640, 3), 9, dtype=np.uint8),
        np.full((480, 640, 3), 10, dtype=np.uint8),
    )
    ambiguous = tmp_path / "ambiguous"
    _write_metric_artifacts(ambiguous, (0, 10, 20))
    outgoing = np.load(ambiguous / "outgoing-mask-0010.npy", allow_pickle=False)
    np.save(ambiguous / "incoming-mask-0010.npy", outgoing, allow_pickle=False)
    with pytest.raises(ValueError, match="ambiguous"):
        checker.compute_automatic_gates(
            ambiguous, frame_indices=(0, 10, 20), accepted_video_frames=accepted
        )

    malformed = tmp_path / "malformed"
    _write_metric_artifacts(malformed, (0, 10, 20))
    np.save(
        malformed / "incoming-mask-0020.npy",
        np.full((480, 640), np.nan),
        allow_pickle=False,
    )
    with pytest.raises(ValueError, match="uint8"):
        checker.compute_automatic_gates(
            malformed, frame_indices=(0, 10, 20), accepted_video_frames=accepted
        )

    tampered = tmp_path / "tampered"
    _write_metric_artifacts(tampered, (0, 10, 20))
    stored = np.load(tampered / "incoming-mask-0020.npy", allow_pickle=False)
    stored[100, 100] = 1
    np.save(tampered / "incoming-mask-0020.npy", stored, allow_pickle=False)
    with pytest.raises(ValueError, match="independently resolved"):
        checker.compute_automatic_gates(
            tampered, frame_indices=(0, 10, 20), accepted_video_frames=accepted
        )


def test_checker_marks_unavailable_pilot_gates_pending(tmp_path: Path) -> None:
    checker = _load_script("check_rigid_vial_transport")
    output = tmp_path / "pilot"
    _write_metric_artifacts(output, (0, 10, 20))
    accepted = (
        np.full((480, 640, 3), 9, dtype=np.uint8),
        np.full((480, 640, 3), 10, dtype=np.uint8),
    )

    result = checker.compute_automatic_gates(
        output, frame_indices=(0, 10, 20), accepted_video_frames=accepted
    )

    assert result["automatic_acceptance_gates"]["end_splice"] == "pending"
    assert result["automatic_acceptance_gates"]["unique_frame_hashes"] == "pending"


def test_full_publication_requires_all_gates_then_writes_21_frame_mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = _load_script("render_rigid_vial_transport")
    contract = importlib.import_module("fluidmix.rigid_vial_transport")
    runtime = _FakeTransportRuntime()
    loaded = _loaded_runtime_fixture(tmp_path, runtime)
    source_hashes = dict(SOURCE_HASHES)
    source_hashes["fill_station.usda"] = loaded["stage_sha256"]
    loaded["source_layer_hashes"] = source_hashes
    monkeypatch.setattr(contract, "SOURCE_LAYER_HASHES", source_hashes)
    output = tmp_path / "transport"

    receipt = renderer.run_transport(
        stage=loaded["stage_path"],
        case_output=loaded["case_output"],
        output_dir=output,
        frame_indices=tuple(range(21)),
        runtime=runtime,
        load_inputs=lambda _case, _stage: loaded,
        check_output=_fake_checker,
    )

    assert receipt["status"] == "pass"
    assert len(receipt["frames"]) == 21
    assert len({row["rgb_sha256"] for row in receipt["frames"]}) == 21
    assert (output / "transport.mp4").stat().st_size > 0


def test_checker_independently_rejects_counter_and_session_diff_claims(
    tmp_path: Path,
) -> None:
    checker = _load_script("check_rigid_vial_transport")
    groups, plan, diff = _valid_plan_and_diff()
    _groups, _plan, _expected_inputs, receipt = _valid_transport_receipt()
    diff_path = tmp_path / "session-diff.json"
    _write_json(diff_path, {"frames": [diff]})
    receipt["authored_diff_sha256"] = _sha256(diff_path)
    spec = importlib.import_module("fluidmix.rigid_vial_transport").RigidTransportSpec()

    checker._validate_counters(receipt, 2702)
    checker._validate_session_diff(tmp_path, receipt, (20,), spec, plan)

    advanced = copy.deepcopy(receipt)
    advanced["counters"]["timeline_time_s_after"] += 0.1
    with pytest.raises(ValueError, match="clock"):
        checker._validate_counters(advanced, 2702)

    unauthorized = copy.deepcopy(diff)
    unauthorized["session_property_paths"].append(f"{ROOT}/FillCam.exposure")
    _write_json(tmp_path / "bad-session-diff.json", {"frames": [unauthorized]})
    (tmp_path / "session-diff.json").write_bytes(
        (tmp_path / "bad-session-diff.json").read_bytes()
    )
    receipt["authored_diff_sha256"] = _sha256(diff_path)
    with pytest.raises(ValueError, match="unauthorized"):
        checker._validate_session_diff(tmp_path, receipt, (20,), spec, plan)
