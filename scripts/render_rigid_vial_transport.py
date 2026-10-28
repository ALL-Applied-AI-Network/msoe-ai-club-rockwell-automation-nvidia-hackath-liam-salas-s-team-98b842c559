#!/usr/bin/env python3
"""Render a frozen, no-physics rigid-vial transport in an anonymous USD session.

Importing this module is CPU-safe.  Isaac Sim and USD are imported only by
``main`` after provenance and no-clobber checks have succeeded.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.rigid_vial_transport import (
    RigidTransportSpec,
    ROOT_LINEAR_BASIS_WORLD_M_PER_LOCAL_UNIT,
    TRANSPORT_GROUPS,
    build_rigid_transport_plan,
    load_rigid_transport_inputs,
    resolve_transport_groups,
    validate_authored_transport_diff,
    validate_rigid_transport_receipt,
)


FULL_FRAMES = tuple(range(21))
PILOT_FRAMES = (0, 10, 20)
CAMERA_PATH = "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
PARTIAL_SUFFIX = ".partial"
REQUIRED_READBACK_KEYS = {
    "isaac_sim_version",
    "renderer",
    "resolution",
    "camera_path",
    "root_linear_basis_world_m_per_local_unit",
    "light_state_sha256",
    "material_state_sha256",
    "tone_exposure",
}
TONE_EXPOSURE = {
    "tone_mapping_operator": 4,
    "auto_exposure_enabled": False,
    "camera_exposure_compensation": 0.0,
    "camera_auto_exposure_enabled": False,
}


class TransportRuntime(Protocol):
    def preflight(self, loaded: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def prepare_capture(self) -> None: ...
    def author_frame(self, frame_index: int) -> Mapping[str, Any]: ...
    def capture_frame(self, frame_index: int) -> Mapping[str, Any]: ...
    def counter_state(self) -> Mapping[str, Any]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a racing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic no-replace rename is unavailable on this host")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), os.fspath(destination))


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_pilot(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pilot frames must be 0,10,20") from exc
    if result != PILOT_FRAMES:
        raise argparse.ArgumentTypeError("pilot frames must be exactly 0,10,20")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot-frames", type=_parse_pilot)
    mode.add_argument("--all-frames", action="store_true")
    args = parser.parse_args(argv)
    args.frame_indices = FULL_FRAMES if args.all_frames else args.pilot_frames
    return args


def _expected_runtime_readback(loaded: Mapping[str, Any]) -> dict[str, Any]:
    explicit = loaded.get("expected_runtime_readback")
    if isinstance(explicit, Mapping):
        return copy.deepcopy(dict(explicit))
    receipt = loaded.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("accepted receipt lacks a pinned runtime/render contract")
    runtime = receipt.get("runtime")
    render = receipt.get("render")
    camera = receipt.get("camera")
    tone = render.get("tone_exposure_state") if isinstance(render, Mapping) else None
    versions = runtime.get("versions") if isinstance(runtime, Mapping) else None
    isaac_version = None
    if isinstance(versions, Mapping):
        isaac_version = versions.get("isaac_sim_version") or versions.get("isaac_sim")
    if isaac_version is None and isinstance(runtime, Mapping):
        isaac_version = runtime.get("isaac_sim_version")
    scene = receipt.get("scene_contract")
    lights = scene.get("effective_light_state") if isinstance(scene, Mapping) else None
    semantics = receipt.get("semantics")
    pairs = semantics.get("pairs") if isinstance(semantics, Mapping) else None
    final_pair = pairs[-1] if isinstance(pairs, list) and pairs else None
    material = (
        final_pair.get("body_material_render_state")
        if isinstance(final_pair, Mapping)
        else None
    )
    readback = tone.get("readback") if isinstance(tone, Mapping) else None
    result = {
        "isaac_sim_version": isaac_version,
        "renderer": render.get("video_renderer") if isinstance(render, Mapping) else None,
        "resolution": render.get("resolution") if isinstance(render, Mapping) else None,
        "camera_path": camera.get("path") if isinstance(camera, Mapping) else None,
        "root_linear_basis_world_m_per_local_unit": [
            list(row) for row in ROOT_LINEAR_BASIS_WORLD_M_PER_LOCAL_UNIT
        ],
        "light_state_sha256": hashlib.sha256(_canonical_json(lights)).hexdigest(),
        "material_state_sha256": (
            final_pair.get("body_material_render_state_sha256")
            if isinstance(final_pair, Mapping)
            else None
        ),
        "tone_exposure": readback,
    }
    if (
        result["isaac_sim_version"] != "6.2.0.0"
        or result["renderer"] != "RaytracedLighting"
        or result["resolution"] != [640, 480]
        or result["camera_path"] != CAMERA_PATH
        or result["tone_exposure"] != TONE_EXPOSURE
        or lights is None
        or material is None
        or not _valid_sha256(result["material_state_sha256"])
        or result["material_state_sha256"]
        != hashlib.sha256(_canonical_json(material)).hexdigest()
    ):
        raise ValueError("accepted receipt lacks the exact pinned runtime readback")
    return result


def _validate_preflight(
    preflight: Mapping[str, Any],
    expected: Mapping[str, Any],
    loaded: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], tuple[dict[str, Any], ...]]:
    if not isinstance(preflight, Mapping) or set(preflight) != {
        "readback",
        "prim_paths",
        "groups",
        "plan",
        "counter_state",
    }:
        raise ValueError("runtime preflight schema is malformed")
    readback = preflight.get("readback")
    if (
        not isinstance(readback, Mapping)
        or set(readback) != REQUIRED_READBACK_KEYS
        or set(expected) != REQUIRED_READBACK_KEYS
        or dict(readback) != dict(expected)
        or readback.get("isaac_sim_version") != "6.2.0.0"
        or readback.get("renderer") != "RaytracedLighting"
        or readback.get("resolution") != [640, 480]
        or readback.get("camera_path") != CAMERA_PATH
        or readback.get("tone_exposure") != TONE_EXPOSURE
        or any(
            not _valid_sha256(readback.get(key))
            for key in ("light_state_sha256", "material_state_sha256")
        )
    ):
        raise ValueError("runtime preflight readback differs from the pinned contract")
    basis = readback["root_linear_basis_world_m_per_local_unit"]
    if basis != [list(row) for row in ROOT_LINEAR_BASIS_WORLD_M_PER_LOCAL_UNIT]:
        raise ValueError("runtime preflight root basis differs from the frozen time-1 basis")
    groups = resolve_transport_groups(
        preflight["prim_paths"], loaded["source_layer_hashes"]
    )
    if {name: tuple(paths) for name, paths in preflight["groups"].items()} != groups:
        raise ValueError("runtime preflight transport groups differ from exact resolution")
    plan = tuple(preflight["plan"])
    if len(plan) != 21:
        raise ValueError("runtime preflight plan is incomplete")
    state = _validate_counter_state(preflight["counter_state"])
    if state["physics_steps"] != loaded["final_physics_step"]:
        raise ValueError("runtime preflight snapshot step differs from the accepted final step")
    return dict(groups), plan


def _validate_counter_state(value: object) -> dict[str, Any]:
    expected = {
        "physics_steps",
        "direct_step_calls",
        "physics_callbacks",
        "simulation_time_s",
        "timeline_time_s",
        "timeline_playing",
        "source_time_code",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("runtime counter readback is malformed")
    result = dict(value)
    for name in ("physics_steps", "direct_step_calls", "physics_callbacks"):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError("runtime counter readback is malformed")
    for name in ("simulation_time_s", "timeline_time_s"):
        if type(result[name]) is not float or not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError("runtime clock readback is malformed")
    if result["timeline_playing"] is not False or result["source_time_code"] != 1:
        raise ValueError("runtime timeline is not stopped at source time code 1")
    return result


def _same_counter_state(before: object, after: object) -> dict[str, Any]:
    left = _validate_counter_state(before)
    right = _validate_counter_state(after)
    if left != right:
        raise ValueError("same-state capture advanced a clock or physics counter")
    return left


def _save_capture(output: Path, frame_index: int, capture: Mapping[str, Any]) -> str:
    import cv2

    if not isinstance(capture, Mapping) or set(capture) != {
        "rgb",
        "outgoing_mask",
        "incoming_mask",
        "instance_ids",
        "instance_id_to_path",
        "counter_state_before",
        "counter_state_after",
    }:
        raise ValueError("capture result schema is malformed")
    rgb = np.asarray(capture["rgb"])
    outgoing = np.asarray(capture["outgoing_mask"])
    incoming = np.asarray(capture["incoming_mask"])
    instance_ids = np.asarray(capture["instance_ids"])
    mapping = capture["instance_id_to_path"]
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("capture RGB must be uint8 640x480 RGB")
    for mask in (outgoing, incoming):
        if mask.shape != (480, 640) or mask.dtype != np.uint8 or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("capture masks must be binary uint8 640x480")
    if np.any(outgoing & incoming):
        raise ValueError("capture instance masks are ambiguous")
    if (
        instance_ids.shape != (480, 640)
        or np.issubdtype(instance_ids.dtype, np.bool_)
        or not np.issubdtype(instance_ids.dtype, np.integer)
        or np.any(instance_ids < 0)
        or not isinstance(mapping, Mapping)
        or any(
            not isinstance(key, str)
            or not key.isdigit()
            or str(int(key)) != key
            or not isinstance(value, str)
            or not value.startswith("/")
            for key, value in mapping.items()
        )
    ):
        raise ValueError("capture instance raster or mapping is malformed")
    _same_counter_state(capture["counter_state_before"], capture["counter_state_after"])
    rgb_path = output / f"frame-{frame_index:04d}.png"
    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {rgb_path}")
    np.save(output / f"outgoing-mask-{frame_index:04d}.npy", outgoing, allow_pickle=False)
    np.save(output / f"incoming-mask-{frame_index:04d}.npy", incoming, allow_pickle=False)
    np.save(output / f"instance-{frame_index:04d}.npy", instance_ids, allow_pickle=False)
    _write_json(output / f"instance-mapping-{frame_index:04d}.json", dict(mapping))
    return _sha256(rgb_path)


def _counter_receipt(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    left = _same_counter_state(before, after)
    return {
        "physics_steps_before": left["physics_steps"],
        "physics_steps_after": left["physics_steps"],
        "physics_steps_during_transport": 0,
        "direct_step_calls_before": left["direct_step_calls"],
        "direct_step_calls_after": left["direct_step_calls"],
        "direct_step_calls_during_transport": 0,
        "physics_callbacks_before": left["physics_callbacks"],
        "physics_callbacks_after": left["physics_callbacks"],
        "physics_callbacks_during_transport": 0,
        "simulation_time_s_before": left["simulation_time_s"],
        "simulation_time_s_after": left["simulation_time_s"],
        "timeline_time_s_before": left["timeline_time_s"],
        "timeline_time_s_after": left["timeline_time_s"],
        "timeline_playing_before": False,
        "timeline_playing_after": False,
        "source_time_code_before": 1,
        "source_time_code_after": 1,
    }


def _build_receipt(
    *,
    loaded: Mapping[str, Any],
    groups: Mapping[str, Sequence[str]],
    plan: Sequence[Mapping[str, Any]],
    indices: tuple[int, ...],
    frame_hashes: Mapping[int, str],
    diff_sha256: str,
    counters: Mapping[str, Any],
    metric_gates: Mapping[str, Any],
) -> dict[str, Any]:
    spec = RigidTransportSpec()
    gates = {
        "source_and_receipt_hashes_unchanged": True,
        "snapshot_binding": True,
        "session_diff_allowlisted": True,
        "zero_clock_and_physics_advancement": True,
        **dict(metric_gates),
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "rigid_vial_transport_receipt",
        "status": "pass" if indices == FULL_FRAMES else "pilot_pending",
        "part": "A",
        "excluded_from_model_inputs": True,
        "spec": {
            "frame_count": spec.frame_count,
            "fps": spec.fps,
            "outgoing_final_delta_y_m": spec.outgoing_final_delta_y_m,
            "incoming_final_delta_y_m": spec.incoming_final_delta_y_m,
        },
        "bindings": {
            "stage_sha256": loaded["stage_sha256"],
            "source_layer_hashes": dict(loaded["source_layer_hashes"]),
            "accepted_receipt_sha256": loaded["receipt_sha256"],
            "accepted_video_sha256": loaded["video_sha256"],
            "final_snapshot_sha256": loaded["snapshot_sha256"],
            "final_physics_step": loaded["final_physics_step"],
        },
        "transport_groups": {name: list(paths) for name, paths in groups.items()},
        "frames": [
            {**copy.deepcopy(dict(plan[index])), "rgb_sha256": frame_hashes[index]}
            for index in indices
        ],
        "authored_diff_sha256": diff_sha256,
        "counters": dict(counters),
        "automatic_acceptance_gates": gates,
    }
    if indices == PILOT_FRAMES:
        receipt["capture_mode"] = "pilot"
    return receipt


def _make_contact_sheet(output: Path, indices: tuple[int, ...]) -> None:
    import cv2

    frames = [cv2.imread(str(output / f"frame-{index:04d}.png")) for index in indices]
    if any(frame is None for frame in frames):
        raise OSError("could not reload frames for contact sheet")
    width = 5 if indices == FULL_FRAMES else 3
    rows = []
    for offset in range(0, len(frames), width):
        row = frames[offset : offset + width]
        while len(row) < width:
            row.append(np.zeros_like(frames[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output / "contact-sheet.png"), sheet):
        raise OSError("could not write contact sheet")


def _encode_video(output: Path) -> None:
    import cv2

    path = output / "transport.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480))
    if not writer.isOpened():
        raise OSError("could not open transport MP4 writer")
    try:
        for index in FULL_FRAMES:
            frame = cv2.imread(str(output / f"frame-{index:04d}.png"))
            if frame is None or frame.shape != (480, 640, 3):
                raise OSError("could not reload frame for transport MP4")
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise OSError("transport MP4 was not finalized")


def run_transport(
    *,
    stage: Path,
    case_output: Path,
    output_dir: Path,
    frame_indices: Sequence[int],
    runtime: TransportRuntime,
    load_inputs: Callable[[Path, Path], Mapping[str, Any]] = load_rigid_transport_inputs,
    check_output: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run fail-closed orchestration against an injected runtime adapter."""

    stage_path = Path(stage).resolve()
    case_root = Path(case_output).resolve()
    final = Path(output_dir).resolve()
    partial = final.with_name(final.name + PARTIAL_SUFFIX)
    indices = tuple(frame_indices)
    if indices not in (FULL_FRAMES, PILOT_FRAMES):
        raise ValueError("frame indices must be exactly full 0..20 or pilot 0,10,20")
    if final.exists() or partial.exists():
        raise FileExistsError("transport output or sibling partial target already exists")

    loaded = dict(load_inputs(case_root, stage_path))
    accepted_path = Path(loaded["receipt_path"])
    accepted_bytes = accepted_path.read_bytes()
    expected = _expected_runtime_readback(loaded)
    preflight = runtime.preflight(loaded)
    groups, plan = _validate_preflight(preflight, expected, loaded)
    before = _validate_counter_state(preflight["counter_state"])
    if final.exists() or partial.exists():
        raise FileExistsError("transport output appeared during preflight")

    partial.mkdir(parents=True)
    owned_partial = True
    try:
        prepare_capture = getattr(runtime, "prepare_capture", None)
        if callable(prepare_capture):
            prepare_capture()
        diffs: list[dict[str, Any]] = []
        hashes: dict[int, str] = {}
        for index in indices:
            diff = dict(runtime.author_frame(index))
            validate_authored_transport_diff(diff, groups, plan, spec=RigidTransportSpec())
            diffs.append(diff)
            capture = runtime.capture_frame(index)
            hashes[index] = _save_capture(partial, index, capture)
        after = _validate_counter_state(runtime.counter_state())
        counters = _counter_receipt(before, after)
        if accepted_path.read_bytes() != accepted_bytes:
            raise ValueError("accepted receipt bytes changed during transport")
        if _sha256(stage_path) != loaded["stage_sha256"]:
            raise ValueError("source stage bytes changed during transport")

        diff_path = partial / "session-diff.json"
        _write_json(diff_path, {"frames": diffs})
        diff_sha = _sha256(diff_path)
        if check_output is None:
            from check_rigid_vial_transport import check_transport_output as check_output
        evaluated = dict(
            check_output(
                output_dir=partial,
                stage=stage_path,
                case_output=case_root,
                expected_frame_indices=indices,
                evaluate_only=True,
            )
        )
        metric_gates = evaluated.get("automatic_acceptance_gates")
        if not isinstance(metric_gates, Mapping):
            metric_gates = {
                "outgoing_absent_final": True,
                "incoming_centered_final": True,
                "fixed_background_sentinel": True,
                "start_splice": True,
                "end_splice": True if indices == FULL_FRAMES else "pending",
                "unique_frame_hashes": True if indices == FULL_FRAMES else "pending",
            }
        if indices == FULL_FRAMES and any(value is not True for value in metric_gates.values()):
            raise ValueError("full transport automatic acceptance gate failed")
        receipt = _build_receipt(
            loaded=loaded,
            groups=groups,
            plan=plan,
            indices=indices,
            frame_hashes=hashes,
            diff_sha256=diff_sha,
            counters=counters,
            metric_gates=metric_gates,
        )
        if indices == FULL_FRAMES:
            expected_inputs = {
                "stage_sha256": loaded["stage_sha256"],
                "source_layer_hashes": loaded["source_layer_hashes"],
                "receipt_sha256": loaded["receipt_sha256"],
                "video_sha256": loaded["video_sha256"],
                "snapshot_sha256": loaded["snapshot_sha256"],
                "final_physics_step": loaded["final_physics_step"],
                "authored_diff_sha256": diff_sha,
            }
            validate_rigid_transport_receipt(
                receipt, expected_inputs, groups, plan, spec=RigidTransportSpec()
            )
        _write_json(partial / "transport-receipt.json", receipt)
        _make_contact_sheet(partial, indices)
        if indices == FULL_FRAMES:
            _encode_video(partial)
        checked = dict(
            check_output(
                output_dir=partial,
                stage=stage_path,
                case_output=case_root,
                expected_frame_indices=indices,
                evaluate_only=False,
            )
        )
        expected_status = "pass" if indices == FULL_FRAMES else "pilot_pending"
        if checked.get("status") != expected_status:
            raise ValueError("independent transport checker did not accept the requested mode")
        _rename_no_replace(partial, final)
        owned_partial = False
        return receipt
    finally:
        if owned_partial and partial.is_dir():
            shutil.rmtree(partial)


class IsaacRigidTransportRuntime:
    """Thin Isaac/USD adapter; constructed only after ``SimulationApp`` starts."""

    def __init__(self, *, app: Any, stage_path: Path, loaded: Mapping[str, Any]):
        import carb
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

        import physx_three_pour_sequence as f2
        from fluidmix.nuisance_authoring import read_existing_light_inventory
        from fluidmix.paint_ball_emitter import (
            COUNT_DRIVEN_PARTICLE_VOLUME_M3,
            build_count_driven_liquid_body_geometry,
        )
        from fluidmix.render_capture import resolve_instance_paths

        self.app = app
        self.loaded = dict(loaded)
        self.rep = rep
        self.SimulationManager = SimulationManager
        self.Gf, self.Sdf, self.Usd = Gf, Sdf, Usd
        self.UsdGeom, self.UsdShade, self.Vt = UsdGeom, UsdShade, Vt
        self.f2 = f2
        self.read_existing_light_inventory = read_existing_light_inventory
        self.build_body_geometry = build_count_driven_liquid_body_geometry
        self.particle_volume_m3 = COUNT_DRIVEN_PARTICLE_VOLUME_M3
        self.resolve_instance_paths = resolve_instance_paths
        self.settings = carb.settings.get_settings()
        self.context = omni.usd.get_context()
        if not self.context.open_stage(os.fspath(Path(stage_path).resolve())):
            raise RuntimeError(f"could not open reviewed source stage: {stage_path}")
        while self.context.get_stage_loading_status()[2] > 0:
            app.update()
        self.stage = self.context.get_stage()
        if self.stage is None:
            raise RuntimeError("USD context did not return the opened source stage")
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.stop()
        time_codes_per_second = float(self.stage.GetTimeCodesPerSecond())
        if not math.isfinite(time_codes_per_second) or time_codes_per_second <= 0.0:
            raise RuntimeError("source stage timeCodesPerSecond is invalid")
        self.timeline.set_current_time(1.0 / time_codes_per_second)
        self.session = self.stage.GetSessionLayer()
        if not self.session.anonymous:
            raise RuntimeError("transport requires an anonymous session layer")
        if list(self.session.rootPrims):
            raise RuntimeError("transport requires a fresh empty session layer")
        self.stage.SetEditTarget(Usd.EditTarget(self.session))
        if self.stage.GetEditTarget().GetLayer() != self.session:
            raise RuntimeError("anonymous session edit target did not retain readback")
        self.camera_prim = self.stage.GetPrimAtPath(CAMERA_PATH)
        if not self.camera_prim.IsValid() or not self.camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError("reviewed FillCam is missing")
        self._direct_step_baseline = int(SimulationManager.get_num_physics_steps())
        self._manager_time_baseline = float(SimulationManager.get_simulation_time())
        self._callback_count = 0  # No callback is registered by this adapter.
        self._base_world_translation: dict[str, tuple[float, float, float]] = {}
        self._existing_orders: dict[str, list[str]] = {}
        self._source_layer_snapshots = {
            str(layer.identifier): hashlib.sha256(
                layer.ExportToString().encode("utf-8")
            ).hexdigest()
            for layer in self.stage.GetUsedLayers()
            if layer != self.session
        }
        self.groups: dict[str, tuple[str, ...]] | None = None
        self.plan: tuple[dict[str, Any], ...] | None = None
        self.render_product = None
        self.rgb_annotator = None
        self.instance_annotator = None

    def _counter_state(self) -> dict[str, Any]:
        direct = int(self.SimulationManager.get_num_physics_steps())
        simulation_time = float(self.SimulationManager.get_simulation_time())
        if direct != self._direct_step_baseline or not math.isclose(
            simulation_time,
            self._manager_time_baseline,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("Isaac physics manager advanced during rigid transport")
        return {
            "physics_steps": int(self.loaded["final_physics_step"]),
            "direct_step_calls": direct,
            "physics_callbacks": self._callback_count,
            "simulation_time_s": simulation_time,
            "timeline_time_s": float(self.timeline.get_current_time()),
            "timeline_playing": bool(self.timeline.is_playing()),
            "source_time_code": 1,
        }

    def counter_state(self) -> Mapping[str, Any]:
        return self._counter_state()

    def _root_basis(self) -> list[list[float]]:
        root = self.stage.GetPrimAtPath("/FillStation/FillStation_LocalRoot")
        if not root.IsValid():
            raise RuntimeError("FillStation local root is missing")
        matrix = self.UsdGeom.Xformable(root).ComputeLocalToWorldTransform(
            self.Usd.TimeCode(1)
        )
        return [
            [float(matrix[row][column]) for column in range(3)]
            for row in range(3)
        ]

    def _world_translation(self, path: str) -> tuple[float, float, float]:
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"reviewed transport prim is missing: {path}")
        value = self.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            self.Usd.TimeCode(1)
        ).ExtractTranslation()
        return tuple(float(value[index]) for index in range(3))

    def _copy_count_body(self) -> None:
        source_prim = self.stage.GetPrimAtPath(self.f2.SOURCE_LIQUID_PATH)
        source_mesh_prim = self.stage.GetPrimAtPath(self.f2.SOURCE_LIQUID_MESH_PATH)
        if not source_prim.IsValid() or not source_mesh_prim.IsA(self.UsdGeom.Mesh):
            raise RuntimeError("reviewed source liquid mesh is missing")
        source_mesh = self.UsdGeom.Mesh(source_mesh_prim)
        source_time = self.Usd.TimeCode(self.f2.COUNT_DRIVEN_SOURCE_TIME_CODE)
        source_points = np.asarray(source_mesh.GetPointsAttr().Get(source_time), dtype=np.float64)
        if source_points.shape != (512, 3) or not np.all(np.isfinite(source_points)):
            raise RuntimeError("reviewed source liquid point contract drifted")
        geometry = self.build_body_geometry(
            source_points,
            contained_particle_count=len(self.loaded["arrays"]["positions"]),
            particle_volume_m3=self.particle_volume_m3,
            source_world_height_m=self.f2.COUNT_DRIVEN_SOURCE_WORLD_HEIGHT_M,
            source_effective_area_m2=self.f2.COUNT_DRIVEN_SOURCE_EFFECTIVE_AREA_M2,
        )
        points = np.asarray(geometry["points"], dtype=np.float64)
        with self.Usd.EditContext(self.stage, self.session):
            body = self.UsdGeom.Xform.Define(
                self.stage, self.f2.COUNT_DRIVEN_BODY_XFORM_PATH
            )
            source_matrix = self.UsdGeom.Xformable(source_prim).GetLocalTransformation(
                source_time
            )
            if isinstance(source_matrix, tuple):
                source_matrix = source_matrix[0]
            self.UsdGeom.Xformable(body.GetPrim()).MakeMatrixXform().Set(source_matrix)
            mesh = self.UsdGeom.Mesh.Define(
                self.stage, self.f2.COUNT_DRIVEN_BODY_MESH_PATH
            )
            mesh_matrix = self.UsdGeom.Xformable(source_mesh_prim).GetLocalTransformation(
                source_time
            )
            if isinstance(mesh_matrix, tuple):
                mesh_matrix = mesh_matrix[0]
            self.UsdGeom.Xformable(mesh.GetPrim()).MakeMatrixXform().Set(mesh_matrix)
            mesh.GetPointsAttr().Set(
                self.Vt.Vec3fArray([self.Gf.Vec3f(*row) for row in points])
            )
            extent = np.asarray(
                [np.min(points, axis=0), np.max(points, axis=0)], dtype=np.float64
            )
            mesh.GetExtentAttr().Set(
                self.Vt.Vec3fArray([self.Gf.Vec3f(*row) for row in extent])
            )
            mesh.GetFaceVertexCountsAttr().Set(
                self.Vt.IntArray(source_mesh.GetFaceVertexCountsAttr().Get(source_time))
            )
            mesh.GetFaceVertexIndicesAttr().Set(
                self.Vt.IntArray(source_mesh.GetFaceVertexIndicesAttr().Get(source_time))
            )
            for source_attr, target_attr in (
                (source_mesh.GetOrientationAttr(), mesh.GetOrientationAttr()),
                (source_mesh.GetSubdivisionSchemeAttr(), mesh.GetSubdivisionSchemeAttr()),
                (source_mesh.GetInterpolateBoundaryAttr(), mesh.GetInterpolateBoundaryAttr()),
                (
                    source_mesh.GetFaceVaryingLinearInterpolationAttr(),
                    mesh.GetFaceVaryingLinearInterpolationAttr(),
                ),
                (
                    source_mesh.GetTriangleSubdivisionRuleAttr(),
                    mesh.GetTriangleSubdivisionRuleAttr(),
                ),
                (source_mesh.GetDoubleSidedAttr(), mesh.GetDoubleSidedAttr()),
            ):
                value = source_attr.Get(source_time)
                if value is not None:
                    target_attr.Set(value)
            source_material, _strength = self.UsdShade.MaterialBindingAPI(
                source_mesh_prim
            ).ComputeBoundMaterial()
            if not source_material or not source_material.GetPrim().IsValid():
                raise RuntimeError("reviewed source liquid material is missing")
            self.UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(source_material)
            display = self.UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "displayColor",
                self.Sdf.ValueTypeNames.Color3fArray,
                self.UsdGeom.Tokens.vertex,
            )
            compositions = np.asarray(
                self.loaded["arrays"]["current_compositions"], dtype=np.float64
            )
            color = compositions.mean(axis=0)
            display.Set(
                self.Vt.Vec3fArray([self.Gf.Vec3f(*color) for _ in range(len(points))])
            )
            labels = mesh.GetPrim().CreateAttribute(
                "semantics:labels:class", self.Sdf.ValueTypeNames.StringArray
            )
            labels.Set(["liquid"])
            mesh.CreateVisibilityAttr().Set(self.UsdGeom.Tokens.inherited)

    def _material_state_hash(self) -> str:
        mesh = self.UsdGeom.Mesh(
            self.stage.GetPrimAtPath(self.f2.COUNT_DRIVEN_BODY_MESH_PATH)
        )
        material, _strength = self.UsdShade.MaterialBindingAPI(
            mesh.GetPrim()
        ).ComputeBoundMaterial()
        if not material or not material.GetPrim().IsValid():
            raise RuntimeError("count body material readback is missing")
        shader = material.ComputeSurfaceSource()
        if isinstance(shader, tuple):
            shader = shader[0]
        if not shader or not shader.GetPrim().IsValid():
            raise RuntimeError("count body surface shader readback is missing")
        state = self.f2._read_count_driven_body_material_state(
            mesh, shader, self.UsdShade, self.UsdGeom
        )
        return hashlib.sha256(_canonical_json(state)).hexdigest()

    def _prepare_capture(self) -> None:
        self.settings.set_string(self.f2.RTX_RENDER_MODE_SETTING, "RaytracedLighting")
        if str(self.settings.get_as_string(self.f2.RTX_RENDER_MODE_SETTING)) != "RaytracedLighting":
            raise RuntimeError("RaytracedLighting readback failed")
        self.settings.set_int("/rtx/post/tonemap/op", 4)
        self.settings.set_bool("/rtx/post/histogram/enabled", False)
        self.render_product = self.rep.create.render_product(CAMERA_PATH, (640, 480))
        self.rgb_annotator = self.rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])
        self.instance_annotator = self.rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"colorize": False}
        )
        self.instance_annotator.attach([self.render_product])

    def _session_inventory(self) -> tuple[list[str], list[str]]:
        prim_paths: list[str] = []
        property_paths: list[str] = []

        def visit(spec: Any) -> None:
            prim_paths.append(str(spec.path))
            property_paths.extend(str(item.path) for item in spec.properties)
            for child in spec.nameChildren:
                visit(child)

        for root_spec in self.session.rootPrims:
            visit(root_spec)
        return sorted(prim_paths), sorted(property_paths)

    def _assert_source_layers_unchanged(self) -> None:
        current = {
            str(layer.identifier): hashlib.sha256(
                layer.ExportToString().encode("utf-8")
            ).hexdigest()
            for layer in self.stage.GetUsedLayers()
            if layer != self.session
        }
        if current != self._source_layer_snapshots:
            raise RuntimeError("a root/source layer changed during rigid transport")

    def preflight(self, loaded: Mapping[str, Any]) -> Mapping[str, Any]:
        basis = self._root_basis()
        expected_basis = [list(row) for row in ROOT_LINEAR_BASIS_WORLD_M_PER_LOCAL_UNIT]
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual_row, expected_row in zip(basis, expected_basis)
            for actual, expected in zip(actual_row, expected_row)
        ):
            raise RuntimeError("opened stage time-1 root linear basis drifted")
        local_root = self.stage.GetPrimAtPath("/FillStation/FillStation_LocalRoot")
        source_paths = [str(prim.GetPath()) for prim in local_root.GetChildren()]
        virtual_paths = [*source_paths, self.f2.COUNT_DRIVEN_BODY_XFORM_PATH]
        groups = resolve_transport_groups(virtual_paths, loaded["source_layer_hashes"])
        anchors = {
            path: self._world_translation(path)[1]
            for path in (
                "/FillStation/FillStation_LocalRoot/Fill_vial_active",
                "/FillStation/FillStation_LocalRoot/Fill_vial_neighbour_1",
                "/FillStation/FillStation_LocalRoot/Fill_vial_neighbour_2",
            )
        }
        plan = build_rigid_transport_plan(groups, anchors, RigidTransportSpec())
        self._copy_count_body()
        for path in sorted({item for values in groups.values() for item in values}):
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"transport prim is absent after reconstruction: {path}")
            ops = self.UsdGeom.Xformable(prim).GetOrderedXformOps()
            self._existing_orders[path] = [str(op.GetOpName()) for op in ops]
            self._base_world_translation[path] = self._world_translation(path)
        self.groups = dict(groups)
        self.plan = tuple(plan)
        light_state = self.read_existing_light_inventory(self.stage)
        camera_exposure = self.camera_prim.GetAttribute("exposure")
        camera_auto = self.camera_prim.GetAttribute("omni:rtx:autoExposure:enabled")
        tone = {
            "tone_mapping_operator": int(self.settings.get_as_int("/rtx/post/tonemap/op")),
            "auto_exposure_enabled": bool(
                self.settings.get_as_bool("/rtx/post/histogram/enabled")
            ),
            "camera_exposure_compensation": float(
                camera_exposure.Get() if camera_exposure.IsValid() else 0.0
            ),
            "camera_auto_exposure_enabled": bool(
                camera_auto.Get() if camera_auto.IsValid() else False
            ),
        }
        return {
            "readback": {
                "isaac_sim_version": "6.2.0.0",
                "renderer": str(self.settings.get_as_string(self.f2.RTX_RENDER_MODE_SETTING)),
                "resolution": [640, 480],
                "camera_path": CAMERA_PATH,
                "root_linear_basis_world_m_per_local_unit": basis,
                "light_state_sha256": hashlib.sha256(
                    _canonical_json(light_state)
                ).hexdigest(),
                "material_state_sha256": self._material_state_hash(),
                "tone_exposure": tone,
            },
            "prim_paths": sorted(virtual_paths),
            "groups": groups,
            "plan": plan,
            "counter_state": self._counter_state(),
        }

    def prepare_capture(self) -> None:
        if self.groups is None or self.plan is None:
            raise RuntimeError("runtime preflight has not completed")
        if self.render_product is not None:
            raise RuntimeError("transport capture was already prepared")
        self._prepare_capture()

    def author_frame(self, frame_index: int) -> Mapping[str, Any]:
        if self.groups is None or self.plan is None:
            raise RuntimeError("runtime preflight has not completed")
        row = self.plan[frame_index]
        authored_local: dict[str, list[float]] = {}
        measured_world: dict[str, list[float]] = {}
        authored_orders: dict[str, list[str]] = {}
        with self.Usd.EditContext(self.stage, self.session):
            for path, delta_xyz in row["path_delta_xyz_m"].items():
                prim = self.stage.GetPrimAtPath(path)
                xformable = self.UsdGeom.Xformable(prim)
                ops = {str(op.GetOpName()): op for op in xformable.GetOrderedXformOps()}
                op = ops.get("xformOp:translate:fluidmixTransport")
                if op is None:
                    op = xformable.AddTranslateOp(
                        self.UsdGeom.XformOp.PrecisionDouble, "fluidmixTransport"
                    )
                local = [
                    0.0,
                    float(delta_xyz[1]) / ROOT_LINEAR_BASIS_WORLD_M_PER_LOCAL_UNIT[1][1],
                    0.0,
                ]
                op.Set(self.Gf.Vec3d(*local))
                existing_ops = [
                    candidate
                    for candidate in xformable.GetOrderedXformOps()
                    if str(candidate.GetOpName()) != "xformOp:translate:fluidmixTransport"
                ]
                if not xformable.SetXformOpOrder([op, *existing_ops]):
                    raise RuntimeError(f"could not retain least-local op order for {path}")
                authored_local[path] = local
                authored_orders[path] = [
                    str(candidate.GetOpName()) for candidate in xformable.GetOrderedXformOps()
                ]
        for path in authored_local:
            after = self._world_translation(path)
            before = self._base_world_translation[path]
            measured_world[path] = [float(after[i] - before[i]) for i in range(3)]
        count_body = self.f2.COUNT_DRIVEN_BODY_XFORM_PATH
        count_mesh = self.f2.COUNT_DRIVEN_BODY_MESH_PATH
        reconstruction = sorted(
            {
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
        )
        transport_properties = {
            property_path
            for path in authored_local
            for property_path in (
                f"{path}.xformOp:translate:fluidmixTransport",
                f"{path}.xformOpOrder",
            )
        }
        expected_prims = sorted(
            {"/FillStation", "/FillStation/FillStation_LocalRoot", *authored_local, count_mesh}
        )
        expected_properties = sorted(transport_properties | set(reconstruction))
        actual_prims, actual_properties = self._session_inventory()
        self._assert_source_layers_unchanged()
        if actual_prims != expected_prims or actual_properties != expected_properties:
            raise RuntimeError("anonymous session layer contains an unauthorized authored property")
        return {
            "edit_target": "anonymous_session_layer",
            "source_time_code": 1,
            "source_layer_property_paths": [],
            "root_layer_property_paths": [],
            "session_prim_paths": actual_prims,
            "count_body_reconstruction_prim_paths": [count_body, count_mesh],
            "session_property_paths": actual_properties,
            "count_body_reconstruction_property_paths": reconstruction,
            "frame_index": frame_index,
            "authored_local_translate_xyz_by_path": authored_local,
            "measured_world_delta_xyz_by_path": measured_world,
            "existing_xform_op_order_by_path": self._existing_orders,
            "authored_xform_op_order_by_path": authored_orders,
        }

    def _group_mask(self, data: np.ndarray, mapping: object, group: str) -> np.ndarray:
        assert self.groups is not None
        records = self.resolve_instance_paths(mapping)
        ids: list[int] = []
        for record in records:
            matching = [
                path
                for path in self.groups[group]
                if record.path == path or record.path.startswith(path + "/")
            ]
            if len(matching) > 1:
                raise RuntimeError("instance path ambiguously resolves to multiple movers")
            if matching:
                ids.append(record.instance_id)
        return np.isin(data, ids).astype(np.uint8)

    def capture_frame(self, frame_index: int) -> Mapping[str, Any]:
        if self.rgb_annotator is None or self.instance_annotator is None:
            raise RuntimeError("runtime capture was not prepared")
        before = self._counter_state()
        self.f2._replicator_same_state_step_checked_in_session(
            stage=self.stage,
            session_layer=self.session,
            Usd=self.Usd,
            rep=self.rep,
            rt_subframes=32,
            callback_count=lambda: self._callback_count,
            timeline=self.timeline,
            physics_state=lambda: {
                "num_physics_steps": int(self.SimulationManager.get_num_physics_steps()),
                "simulation_time_s": float(self.SimulationManager.get_simulation_time()),
            },
        )
        rgb = np.asarray(self.rgb_annotator.get_data())
        instance_raw = self.instance_annotator.get_data()
        if rgb.ndim != 3 or rgb.shape[:2] != (480, 640) or rgb.shape[2] < 3:
            raise RuntimeError("RGB annotator returned an invalid raster")
        if not isinstance(instance_raw, Mapping):
            raise RuntimeError("instance annotator returned no mapping envelope")
        info = instance_raw.get("info")
        data = np.asarray(instance_raw.get("data"))
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[..., 0]
        if (
            not isinstance(info, Mapping)
            or "idToLabels" not in info
            or data.shape != (480, 640)
            or np.issubdtype(data.dtype, np.bool_)
            or not np.issubdtype(data.dtype, np.integer)
        ):
            raise RuntimeError("instance annotator raster or path mapping is malformed")
        outgoing = self._group_mask(data, info["idToLabels"], "outgoing")
        incoming = self._group_mask(data, info["idToLabels"], "incoming")
        mapping = {
            str(record.instance_id): record.path
            for record in self.resolve_instance_paths(info["idToLabels"])
        }
        if np.any(outgoing & incoming):
            raise RuntimeError("outgoing and incoming instance masks overlap")
        after = self._counter_state()
        return {
            "rgb": np.asarray(rgb[..., :3], dtype=np.uint8),
            "outgoing_mask": outgoing,
            "incoming_mask": incoming,
            "instance_ids": np.asarray(data, dtype=np.uint32),
            "instance_id_to_path": mapping,
            "counter_state_before": before,
            "counter_state_after": after,
        }


def _build_isaac_runtime(app: Any, stage: Path, loaded: Mapping[str, Any]) -> TransportRuntime:
    """Construct the thin runtime adapter after SimulationApp exists.

    The adapter is intentionally imported from the exercised F2 runner so the
    count-body and same-state capture implementations are not duplicated here.
    """

    return IsaacRigidTransportRuntime(app=app, stage_path=stage, loaded=loaded)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists() or args.output_dir.with_name(
        args.output_dir.name + PARTIAL_SUFFIX
    ).exists():
        raise FileExistsError("transport output or sibling partial target already exists")
    loaded = load_rigid_transport_inputs(args.case_output, args.stage)
    _expected_runtime_readback(loaded)
    # This is the only executable path that imports or creates SimulationApp.
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RaytracedLighting",
            "width": 640,
            "height": 480,
        }
    )
    try:
        runtime = _build_isaac_runtime(app, args.stage, loaded)
        receipt = run_transport(
            stage=args.stage,
            case_output=args.case_output,
            output_dir=args.output_dir,
            frame_indices=args.frame_indices,
            runtime=runtime,
            load_inputs=lambda _case, _stage: loaded,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
