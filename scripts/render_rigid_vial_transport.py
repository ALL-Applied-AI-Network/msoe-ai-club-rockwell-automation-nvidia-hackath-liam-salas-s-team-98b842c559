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
import tempfile
import uuid
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.rigid_vial_transport import (
    COUNT_BODY_RECONSTRUCTION_PROPERTY_PATHS,
    COUNT_BODY_RECONSTRUCTION_PRIM_PATHS,
    GLOBAL_MATERIAL_PATH,
    MATERIAL_NAMESPACE_PATH,
    MATERIAL_RECONSTRUCTION_APPLIED_SCHEMAS,
    MATERIAL_RECONSTRUCTION_PRIM_TYPES,
    SPATIAL_MATERIAL_PATH,
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
    def prepare_render_product(self) -> Mapping[str, Any]: ...
    def prepare_capture(self) -> None: ...
    def author_frame(self, frame_index: int) -> Mapping[str, Any]: ...
    def capture_frame(self, frame_index: int) -> Mapping[str, Any]: ...
    def counter_state(self) -> Mapping[str, Any]: ...
    def attest_state(self) -> Mapping[str, Any]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError("transport output manifest contains a non-file artifact")
        result[path.name] = _sha256(path)
    return result


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


def _owned_directory_identity(path: Path) -> tuple[int, int]:
    stat_result = path.lstat()
    if not path.is_dir() or path.is_symlink():
        raise OSError("transport partial is not an owned directory")
    return stat_result.st_dev, stat_result.st_ino


def _assert_owned_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = _owned_directory_identity(path)
    except (FileNotFoundError, OSError) as exc:
        raise OSError("transport partial ownership was lost") from exc
    if current != identity:
        raise OSError("transport partial ownership was lost")


def _private_quarantine(parent: Path) -> Path:
    path = Path(tempfile.mkdtemp(prefix=".rigid-transport-quarantine-", dir=parent))
    path.chmod(0o700)
    return path


def _publish_owned_directory(
    source: Path, destination: Path, identity: tuple[int, int]
) -> None:
    quarantine_root = _private_quarantine(source.parent)
    candidate = quarantine_root / "owned"
    try:
        os.rename(source, candidate)
        if _owned_directory_identity(candidate) != identity:
            raise OSError("transport partial ownership was lost before publication")
        _rename_no_replace(candidate, destination)
    finally:
        try:
            quarantine_root.rmdir()
        except OSError:
            # A foreign replacement is deliberately preserved in the private quarantine.
            pass


def _cleanup_owned_directory(source: Path, identity: tuple[int, int]) -> None:
    if not source.exists() and not source.is_symlink():
        return
    quarantine_root = _private_quarantine(source.parent)
    candidate = quarantine_root / "candidate"
    try:
        os.rename(source, candidate)
        if _owned_directory_identity(candidate) == identity:
            shutil.rmtree(candidate)
    finally:
        try:
            quarantine_root.rmdir()
        except OSError:
            # Never delete a replacement inode merely to tidy the quarantine.
            pass


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
    receipt = loaded.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("accepted receipt lacks a pinned runtime/render contract")
    versions = receipt.get("versions")
    if (
        not isinstance(versions, Mapping)
        or set(versions) != {"isaac", "kit", "physx"}
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise ValueError("accepted receipt versions schema is malformed")
    render = receipt.get("render")
    camera = receipt.get("camera")
    tone = render.get("tone_exposure_state") if isinstance(render, Mapping) else None
    isaac_version = versions["isaac"]
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


def _read_live_isaac_version(
    app: Any,
    version_info: Callable[[Any], Mapping[str, Any]],
    *,
    expected: str | None = None,
) -> str:
    versions = version_info(app)
    if not isinstance(versions, Mapping) or set(versions) != {"isaac", "kit", "physx"}:
        raise ValueError("qualified live version helper returned a malformed schema")
    value = versions.get("isaac")
    if not isinstance(value, str) or not value:
        raise ValueError("qualified live Isaac version is missing")
    if expected is not None and value != expected:
        raise ValueError(f"live Isaac version {value!r} differs from accepted {expected!r}")
    return value


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
        "frame_zero_diff",
        "render_product_state",
        "writer_constructed",
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
    if preflight["writer_constructed"] is not False or preflight["render_product_state"] != {
        "constructed": False,
        "camera_path": CAMERA_PATH,
        "resolution": [640, 480],
    }:
        raise ValueError("runtime preflight render-product state is malformed")
    validate_authored_transport_diff(
        preflight["frame_zero_diff"], groups, plan, spec=RigidTransportSpec()
    )
    return dict(groups), plan


def _validate_render_product_state(value: object) -> dict[str, Any]:
    required = {
        "product_path",
        "product_type",
        "runtime_prim_paths",
        "runtime_property_paths",
        "camera_target",
        "resolution",
        "session_layer_only",
        "baseline_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("live render product schema is malformed")
    result = copy.deepcopy(dict(value))
    path = result["product_path"]
    if (
        not isinstance(path, str)
        or not path.startswith("/Render/FluidmixRigidTransport_")
        or result["product_type"] != "RenderProduct"
        or result["runtime_prim_paths"] != ["/Render", path]
        or not isinstance(result["runtime_property_paths"], list)
        or result["runtime_property_paths"] != sorted(set(result["runtime_property_paths"]))
        or not result["runtime_property_paths"]
        or any(not item.startswith(path + ".") for item in result["runtime_property_paths"])
        or result["camera_target"] != CAMERA_PATH
        or result["resolution"] != [640, 480]
        or result["session_layer_only"] is not True
        or not _valid_sha256(result["baseline_sha256"])
    ):
        raise ValueError("live render product readback differs from the pinned contract")
    return result


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


def _validate_capture_guard(frame_index: int, value: object) -> dict[str, Any]:
    required = {
        "callbacks_before",
        "callbacks_after",
        "timeline_time_before_s",
        "timeline_time_after_s",
        "timeline_playing_before",
        "timeline_playing_after",
        "physics_steps_before",
        "physics_steps_after",
        "simulation_time_before_s",
        "simulation_time_after_s",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("same-state capture guard is malformed")
    result = dict(value)
    for before_name, after_name in (
        ("callbacks_before", "callbacks_after"),
        ("physics_steps_before", "physics_steps_after"),
    ):
        if (
            type(result[before_name]) is not int
            or result[before_name] < 0
            or result[after_name] != result[before_name]
        ):
            raise ValueError("same-state capture guard advanced physics")
    for before_name, after_name in (
        ("timeline_time_before_s", "timeline_time_after_s"),
        ("simulation_time_before_s", "simulation_time_after_s"),
    ):
        if (
            type(result[before_name]) is not float
            or not math.isfinite(result[before_name])
            or result[before_name] < 0.0
            or result[after_name] != result[before_name]
        ):
            raise ValueError("same-state capture guard advanced a clock")
    if (
        result["timeline_playing_before"] is not False
        or result["timeline_playing_after"] is not False
    ):
        raise ValueError("same-state capture guard changed playback")
    return {"frame_index": frame_index, **result}


def _validate_counter_guard_binding(
    before: Mapping[str, Any], after: Mapping[str, Any], guard: Mapping[str, Any]
) -> None:
    pairs = (
        ("physics_callbacks", "callbacks"),
        ("direct_step_calls", "physics_steps"),
        ("simulation_time_s", "simulation_time"),
        ("timeline_time_s", "timeline_time"),
        ("timeline_playing", "timeline_playing"),
    )
    for counter_name, guard_name in pairs:
        if (
            before[counter_name] != guard[f"{guard_name}_before" + ("_s" if guard_name in {"simulation_time", "timeline_time"} else "")]
            or after[counter_name] != guard[f"{guard_name}_after" + ("_s" if guard_name in {"simulation_time", "timeline_time"} else "")]
        ):
            raise ValueError("capture counter state differs from same-state guard evidence")
    if before["physics_steps"] != guard["physics_steps_before"] or after["physics_steps"] != guard["physics_steps_after"]:
        raise ValueError("capture live physics steps differ from same-state guard evidence")


def _validate_evidence_chain(
    diffs: Sequence[Mapping[str, Any]],
    guards: Sequence[Mapping[str, Any]],
    attestations: Sequence[Mapping[str, Any]],
    counters: Mapping[str, Any],
) -> None:
    if not (len(diffs) == len(guards) == len(attestations)) or not diffs:
        raise ValueError("transport evidence chain length is malformed")
    for position, (diff, guard, attestation) in enumerate(zip(diffs, guards, attestations)):
        index = diff.get("frame_index")
        state = attestation.get("state") if isinstance(attestation, Mapping) else None
        if (
            guard.get("frame_index") != index
            or attestation.get("frame_index") != index
            or not isinstance(state, Mapping)
            or state.get("scene_prim_paths") != diff.get("session_prim_paths")
            or state.get("scene_property_paths") != diff.get("session_property_paths")
        ):
            raise ValueError("transport evidence chain index or scene inventory is disconnected")
        if position:
            prior = guards[position - 1]
            for stem in (
                "callbacks", "physics_steps", "simulation_time", "timeline_time", "timeline_playing"
            ):
                suffix = "_s" if stem in {"simulation_time", "timeline_time"} else ""
                if prior[f"{stem}_after{suffix}"] != guard[f"{stem}_before{suffix}"]:
                    raise ValueError("transport evidence chain has an adjacent discontinuity")
            prior_state = attestations[position - 1]["state"]
            for name in ("non_session_layer_sha256", "source_file_sha256", "runtime_render_state"):
                if prior_state[name] != state[name]:
                    raise ValueError("transport evidence chain provenance changed between frames")
    first, last = guards[0], guards[-1]
    endpoint_pairs = (
        ("physics_callbacks", "callbacks"),
        ("physics_steps", "physics_steps"),
        ("direct_step_calls", "physics_steps"),
        ("simulation_time_s", "simulation_time"),
        ("timeline_time_s", "timeline_time"),
        ("timeline_playing", "timeline_playing"),
    )
    for counter_stem, guard_stem in endpoint_pairs:
        suffix = "_s" if guard_stem in {"simulation_time", "timeline_time"} else ""
        if (
            counters[f"{counter_stem}_before"] != first[f"{guard_stem}_before{suffix}"]
            or counters[f"{counter_stem}_after"] != last[f"{guard_stem}_after{suffix}"]
        ):
            raise ValueError("transport evidence chain endpoints differ from aggregates")


def _validate_attestation(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "session_prim_paths",
        "session_property_paths",
        "scene_prim_paths",
        "scene_property_paths",
        "runtime_render_state",
        "session_sha256",
        "non_session_layer_sha256",
        "source_file_sha256",
    }:
        raise ValueError("runtime state attestation is malformed")
    result = copy.deepcopy(dict(value))
    if (
        not isinstance(result["session_prim_paths"], list)
        or not isinstance(result["session_property_paths"], list)
        or not isinstance(result["scene_prim_paths"], list)
        or not isinstance(result["scene_property_paths"], list)
        or result["session_prim_paths"] != sorted(set(result["session_prim_paths"]))
        or result["session_property_paths"] != sorted(set(result["session_property_paths"]))
        or result["scene_prim_paths"] != sorted(set(result["scene_prim_paths"]))
        or result["scene_property_paths"] != sorted(set(result["scene_property_paths"]))
        or not _valid_sha256(result["session_sha256"])
    ):
        raise ValueError("runtime state attestation is malformed")
    for name in ("non_session_layer_sha256", "source_file_sha256"):
        mapping = result[name]
        if (
            not isinstance(mapping, Mapping)
            or not mapping
            or any(not isinstance(key, str) or not _valid_sha256(item) for key, item in mapping.items())
        ):
            raise ValueError("runtime state attestation is malformed")
    render_state = _validate_render_product_state(result["runtime_render_state"])
    if (
        sorted([*result["scene_prim_paths"], *render_state["runtime_prim_paths"]])
        != result["session_prim_paths"]
        or sorted([*result["scene_property_paths"], *render_state["runtime_property_paths"]])
        != result["session_property_paths"]
        or set(result["scene_prim_paths"]) & set(render_state["runtime_prim_paths"])
        or set(result["scene_property_paths"]) & set(render_state["runtime_property_paths"])
    ):
        raise ValueError("runtime state attestation partitions overlap or are incomplete")
    result["runtime_render_state"] = render_state
    return result


def _validate_runtime_rgb(value: object) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[:2] != (480, 640)
        or value.shape[2] not in (3, 4)
    ):
        raise ValueError("RGB annotator must return uint8 640x480 RGB or RGBA")
    return value if value.shape[2] == 3 else value[..., :3]


def _run_qualified_count_body_chain(
    *,
    helpers: Any,
    arrays: Mapping[str, np.ndarray],
    accepted_source_evidence: Mapping[str, Any],
    accepted_capture_geometry: Mapping[str, Any],
    accepted_material_state: Mapping[str, Any],
    final_physics_step: int,
    source_transmissions: np.ndarray,
    author_kwargs: Mapping[str, Any],
    update_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if type(final_physics_step) is not int or final_physics_step != 2702:
        raise ValueError("accepted count-body final physics step must be exactly 2702")
    runtime, source_evidence = helpers._author_count_driven_twin_liquid_body(
        **dict(author_kwargs)
    )
    if any(
        accepted_source_evidence.get(key) != value
        for key, value in source_evidence.items()
    ):
        raise ValueError("accepted count-body source geometry evidence drifted")
    positions = arrays["positions"]
    compositions = arrays["current_compositions"]
    contained = helpers._count_driven_contained_mask(positions)
    if (
        not isinstance(contained, np.ndarray)
        or contained.dtype != np.bool_
        or contained.shape != (len(positions),)
        or not np.any(contained)
    ):
        raise ValueError("accepted count-body containment mask is malformed or empty")
    live_geometry = helpers._update_count_driven_twin_liquid_body(
        runtime,
        contained_positions=positions[contained],
        contained_compositions=compositions[contained],
        source_transmissions=source_transmissions,
        **dict(update_kwargs),
    )
    selected = helpers._select_count_driven_body_capture_geometry(
        live_geometry,
        physics_step=final_physics_step,
        phase=str(accepted_capture_geometry.get("phase")),
    )
    if dict(selected) != dict(accepted_capture_geometry):
        raise ValueError("accepted count-body capture geometry drifted")
    material_state = helpers._read_count_driven_body_material_state(
        runtime["mesh"],
        author_kwargs["body_shader"],
        author_kwargs["UsdShade"],
        author_kwargs["UsdGeom"],
    )
    if dict(material_state) != dict(accepted_material_state):
        raise ValueError("accepted count-body material state drifted")
    return {
        "runtime": runtime,
        "source_evidence": source_evidence,
        "capture_geometry": selected,
        "material_state": material_state,
        "contained_particle_count": int(np.count_nonzero(contained)),
    }


def _run_qualified_count_body_reconstruction(
    *,
    helpers: Any,
    spatial_args: Sequence[Any],
    global_args: Sequence[Any],
    global_semantics_args: Sequence[Any],
    chain_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    global_shader = helpers._author_omnisurface(*tuple(global_args))
    helpers._author_direct_liquid_semantics(*tuple(global_semantics_args))
    shader = helpers._author_spatial_omnisurface(*tuple(spatial_args))
    kwargs = dict(chain_kwargs)
    kwargs["author_kwargs"] = {**kwargs["author_kwargs"], "body_shader": shader}
    kwargs["update_kwargs"] = {**kwargs["update_kwargs"], "global_shader": global_shader}
    result = _run_qualified_count_body_chain(helpers=helpers, **kwargs)
    result["global_shader"] = global_shader
    result["spatial_shader"] = shader
    return result


def _validate_material_network_state(
    state: object,
    *,
    accepted_material_state: Mapping[str, Any],
    accepted_capture_geometry: Mapping[str, Any],
    spatial_source_asset: str,
    spatial_subidentifier: str,
) -> dict[str, Any]:
    spatial_inputs = accepted_material_state.get("shader_inputs")
    global_color = accepted_capture_geometry.get("body_material_color")
    if not isinstance(spatial_inputs, Mapping) or set(spatial_inputs) != {"fallback_color"}:
        raise ValueError("accepted spatial material inputs are malformed")
    if not isinstance(global_color, list) or len(global_color) != 3:
        raise ValueError("accepted global material color is malformed")
    expected = {
        "spatial": {
            "implementation_source": "sourceAsset",
            "source_asset": spatial_source_asset,
            "subidentifier": spatial_subidentifier,
            "inputs": dict(spatial_inputs),
            "material_outputs": {
                "surface": f"{accepted_material_state['shader_path']}.outputs:out",
                "volume": f"{accepted_material_state['shader_path']}.outputs:out",
                "displacement": f"{accepted_material_state['shader_path']}.outputs:out",
            },
            "shader_output_type": "token",
            "shader_inputs_sha256": accepted_material_state.get("shader_inputs_sha256"),
            "display_color_sha256": accepted_material_state.get("display_color_sha256"),
        },
        "global": {
            "implementation_source": "sourceAsset",
            "source_asset": "OmniSurface.mdl",
            "subidentifier": "OmniSurface",
            "inputs": {
                "diffuse_reflection_weight": 0.0,
                "enable_opacity": False,
                "enable_specular_transmission": True,
                "geometry_opacity": 1.0,
                "geometry_opacity_threshold": 0.0,
                "specular_reflection_ior": 1.333,
                "specular_reflection_roughness": 0.08,
                "specular_transmission_color": global_color,
                "specular_transmission_scattering_depth": 0.01,
                "specular_transmission_weight": 1.0,
                "thin_walled": False,
            },
            "material_outputs": {
                "surface": f"{GLOBAL_MATERIAL_PATH}/OmniSurface.outputs:out",
                "volume": f"{GLOBAL_MATERIAL_PATH}/OmniSurface.outputs:out",
                "displacement": f"{GLOBAL_MATERIAL_PATH}/OmniSurface.outputs:out",
            },
            "shader_output_type": "token",
            "body_material_color_sha256": accepted_capture_geometry.get(
                "body_material_color_sha256"
            ),
        },
    }
    if not isinstance(state, Mapping) or dict(state) != expected:
        raise ValueError("authored material network differs from the accepted exact state")
    return copy.deepcopy(expected)


def _save_capture(
    output: Path, frame_index: int, capture: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    import cv2

    if not isinstance(capture, Mapping) or set(capture) != {
        "rgb",
        "outgoing_mask",
        "incoming_mask",
        "instance_ids",
        "instance_id_to_path",
        "counter_state_before",
        "counter_state_after",
        "same_state_guard",
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
    capture_before = _validate_counter_state(capture["counter_state_before"])
    capture_after = _validate_counter_state(capture["counter_state_after"])
    _same_counter_state(capture_before, capture_after)
    guard = _validate_capture_guard(frame_index, capture["same_state_guard"])
    _validate_counter_guard_binding(capture_before, capture_after, guard)
    rgb_path = output / f"frame-{frame_index:04d}.png"
    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {rgb_path}")
    np.save(output / f"outgoing-mask-{frame_index:04d}.npy", outgoing, allow_pickle=False)
    np.save(output / f"incoming-mask-{frame_index:04d}.npy", incoming, allow_pickle=False)
    np.save(output / f"instance-{frame_index:04d}.npy", instance_ids, allow_pickle=False)
    _write_json(output / f"instance-mapping-{frame_index:04d}.json", dict(mapping))
    return _sha256(rgb_path), guard


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
    evidence_gates: Mapping[str, bool],
) -> dict[str, Any]:
    spec = RigidTransportSpec()
    gates = {
        **dict(evidence_gates),
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
            "accepted_final_physics_step": loaded["final_physics_step"],
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


METRIC_GATE_KEYS = {
    "outgoing_absent_final",
    "incoming_centered_final",
    "fixed_background_sentinel",
    "start_splice",
    "end_splice",
    "unique_frame_hashes",
}
METRIC_KEYS = {
    "outgoing_final_pixel_count",
    "incoming_final_center_x_px",
    "active_start_center_x_px",
    "incoming_center_error_px",
    "fixed_background_sentinel_max_delta",
    "accepted_in_fill_splice_mad_p99",
    "start_splice_mad",
    "end_splice_background_mad",
    "frame_hashes",
}


def _gates_from_metrics(value: object, indices: tuple[int, ...]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != METRIC_KEYS:
        raise ValueError("independent checker metrics schema is malformed")
    metrics = dict(value)
    for name in ("outgoing_final_pixel_count", "fixed_background_sentinel_max_delta"):
        if type(metrics[name]) is not int or metrics[name] < 0:
            raise ValueError("independent checker metrics contain an invalid integer")
    for name in METRIC_KEYS - {
        "outgoing_final_pixel_count", "fixed_background_sentinel_max_delta", "frame_hashes"
    }:
        if type(metrics[name]) is not float or not math.isfinite(metrics[name]) or metrics[name] < 0.0:
            raise ValueError("independent checker metrics contain an invalid finite value")
    hashes = metrics["frame_hashes"]
    if (
        not isinstance(hashes, list)
        or len(hashes) != len(indices)
        or any(not _valid_sha256(item) for item in hashes)
    ):
        raise ValueError("independent checker metrics frame hashes are malformed")
    gates: dict[str, Any] = {
        "outgoing_absent_final": metrics["outgoing_final_pixel_count"] == 0,
        "incoming_centered_final": metrics["incoming_center_error_px"] <= 3.0,
        "fixed_background_sentinel": metrics["fixed_background_sentinel_max_delta"] <= 2,
        "start_splice": metrics["start_splice_mad"] <= metrics["accepted_in_fill_splice_mad_p99"],
    }
    if indices == FULL_FRAMES:
        gates.update(
            {
                "end_splice": metrics["end_splice_background_mad"]
                <= metrics["accepted_in_fill_splice_mad_p99"],
                "unique_frame_hashes": len(set(hashes)) == len(indices),
            }
        )
    else:
        gates.update({"end_splice": "pending", "unique_frame_hashes": "pending"})
    return gates


def _validated_checker_result(
    value: object, indices: tuple[int, ...]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "automatic_acceptance_gates",
        "metrics",
    }:
        raise ValueError("independent checker result schema is malformed")
    gates = value["automatic_acceptance_gates"]
    if not isinstance(gates, Mapping) or set(gates) != METRIC_GATE_KEYS:
        raise ValueError("independent checker gate schema is malformed")
    if any(
        item is not True and not (indices == PILOT_FRAMES and item == "pending" and type(item) is str)
        for item in gates.values()
    ):
        raise ValueError("independent checker returned a failed or mistyped gate")
    if indices == PILOT_FRAMES and (
        gates["end_splice"] != "pending"
        or gates["unique_frame_hashes"] != "pending"
        or any(gates[name] is not True for name in METRIC_GATE_KEYS - {"end_splice", "unique_frame_hashes"})
    ):
        raise ValueError("pilot checker gates do not match the exact pending contract")
    derived_gates = _gates_from_metrics(value["metrics"], indices)
    if dict(gates) != derived_gates:
        raise ValueError("independent checker gates differ from exact metrics")
    return {"automatic_acceptance_gates": dict(gates), "metrics": dict(value["metrics"])}


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
    predictable_partial = final.with_name(final.name + PARTIAL_SUFFIX)
    indices = tuple(frame_indices)
    if indices not in (FULL_FRAMES, PILOT_FRAMES):
        raise ValueError("frame indices must be exactly full 0..20 or pilot 0,10,20")
    if final.exists() or predictable_partial.exists():
        raise FileExistsError("transport output or sibling partial target already exists")

    loaded = dict(load_inputs(case_root, stage_path))
    accepted_path = Path(loaded["receipt_path"])
    accepted_bytes = accepted_path.read_bytes()
    expected = _expected_runtime_readback(loaded)
    preflight = runtime.preflight(loaded)
    groups, plan = _validate_preflight(preflight, expected, loaded)
    before = _validate_counter_state(preflight["counter_state"])
    render_product_state = _validate_render_product_state(
        runtime.prepare_render_product()
    )
    if final.exists() or predictable_partial.exists():
        raise FileExistsError("transport output appeared during preflight")

    partial = Path(tempfile.mkdtemp(prefix=final.name + PARTIAL_SUFFIX + "-", dir=final.parent))
    partial_fd = os.open(partial, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    partial_stat = os.fstat(partial_fd)
    partial_identity = (partial_stat.st_dev, partial_stat.st_ino)
    owned_view = Path(f"/proc/self/fd/{partial_fd}")
    owned_partial = True
    try:
        _assert_owned_directory(partial, partial_identity)
        prepare_capture = getattr(runtime, "prepare_capture", None)
        if callable(prepare_capture):
            prepare_capture()
        diffs: list[dict[str, Any]] = []
        guards: list[dict[str, Any]] = []
        attestations: list[dict[str, Any]] = []
        hashes: dict[int, str] = {}
        for index in indices:
            diff = dict(runtime.author_frame(index))
            validate_authored_transport_diff(diff, groups, plan, spec=RigidTransportSpec())
            diffs.append(diff)
            before_attestation = _validate_attestation(runtime.attest_state())
            if before_attestation["runtime_render_state"] != render_product_state:
                raise ValueError(
                    "persisted render product differs from the pre-output live readback"
                )
            capture = runtime.capture_frame(index)
            after_attestation = _validate_attestation(runtime.attest_state())
            if after_attestation != before_attestation:
                raise ValueError("runtime state changed during same-state capture")
            _assert_owned_directory(partial, partial_identity)
            hashes[index], guard = _save_capture(owned_view, index, capture)
            guards.append(guard)
            attestations.append({"frame_index": index, "state": after_attestation})
        after = _validate_counter_state(runtime.counter_state())
        counters = _counter_receipt(before, after)
        _validate_evidence_chain(diffs, guards, attestations, counters)
        if accepted_path.read_bytes() != accepted_bytes:
            raise ValueError("accepted receipt bytes changed during transport")
        if _sha256(stage_path) != loaded["stage_sha256"]:
            raise ValueError("source stage bytes changed during transport")

        final_attestation = _validate_attestation(runtime.attest_state())
        if final_attestation != attestations[-1]["state"]:
            raise ValueError("runtime state changed before checker invocation")
        diff_path = owned_view / "session-diff.json"
        _assert_owned_directory(partial, partial_identity)
        _write_json(
            diff_path,
            {
                "frames": diffs,
                "capture_guards": guards,
                "state_attestations": attestations,
            },
        )
        diff_sha = _sha256(diff_path)
        if check_output is None:
            from check_rigid_vial_transport import check_transport_output as check_output
        _assert_owned_directory(partial, partial_identity)
        evaluated = _validated_checker_result(
            check_output(
                output_dir=owned_view,
                stage=stage_path,
                case_output=case_root,
                expected_frame_indices=indices,
                evaluate_only=True,
            ),
            indices,
        )
        metric_gates = evaluated["automatic_acceptance_gates"]
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
            evidence_gates={
                "source_and_receipt_hashes_unchanged": accepted_path.read_bytes() == accepted_bytes
                and _sha256(stage_path) == loaded["stage_sha256"],
                "snapshot_binding": _sha256(Path(loaded["snapshot_path"]))
                == loaded["snapshot_sha256"],
                "session_diff_allowlisted": len(diffs) == len(indices),
                "zero_clock_and_physics_advancement": before == after
                and len(guards) == len(indices),
            },
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
        _assert_owned_directory(partial, partial_identity)
        _write_json(owned_view / "transport-receipt.json", receipt)
        _assert_owned_directory(partial, partial_identity)
        _make_contact_sheet(owned_view, indices)
        if indices == FULL_FRAMES:
            _assert_owned_directory(partial, partial_identity)
            _encode_video(owned_view)
        if _validate_attestation(runtime.attest_state()) != final_attestation:
            raise ValueError("runtime state changed before publication checker")
        pre_final_checker_manifest = _output_manifest(owned_view)
        _assert_owned_directory(partial, partial_identity)
        checked = _validated_checker_result(
            check_output(
                output_dir=owned_view,
                stage=stage_path,
                case_output=case_root,
                expected_frame_indices=indices,
                evaluate_only=False,
            ),
            indices,
        )
        if checked != evaluated:
            raise ValueError("independent checker changed its evaluated result")
        if _validate_attestation(runtime.attest_state()) != final_attestation:
            raise ValueError("runtime state changed during final checker")
        if _output_manifest(owned_view) != pre_final_checker_manifest:
            raise ValueError("transport output changed during final checker")
        if accepted_path.read_bytes() != accepted_bytes or _sha256(stage_path) != loaded["stage_sha256"]:
            raise ValueError("source bytes changed during final checker")
        _assert_owned_directory(partial, partial_identity)
        _publish_owned_directory(partial, final, partial_identity)
        owned_partial = False
        return receipt
    finally:
        try:
            if owned_partial:
                _cleanup_owned_directory(partial, partial_identity)
        finally:
            os.close(partial_fd)


class IsaacRigidTransportRuntime:
    """Thin Isaac/USD adapter; constructed only after ``SimulationApp`` starts."""

    def __init__(self, *, app: Any, stage_path: Path, loaded: Mapping[str, Any]):
        import carb
        import omni.replicator.core as rep
        import omni.physx
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdRender, UsdSemantics, UsdShade, Vt
        try:
            from pxr import Semantics as LegacySemantics
        except ImportError:
            LegacySemantics = None

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
        self.UsdRender = UsdRender
        self.UsdSemantics, self.LegacySemantics = UsdSemantics, LegacySemantics
        self.f2 = f2
        self.live_isaac_version = _read_live_isaac_version(
            app,
            f2._version_info,
            expected=str(loaded["receipt"]["versions"]["isaac"]),
        )
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
        self._stage_time_codes_per_second = time_codes_per_second
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
        self._callback_records: list[float] = []

        def on_physics_step(callback_dt: float) -> None:
            value = float(callback_dt)
            if not math.isfinite(value):
                raise RuntimeError("physics callback dt is non-finite")
            self._callback_records.append(value)

        self._physics_step_subscription = (
            omni.physx.get_physx_interface().subscribe_physics_step_events(
                on_physics_step
            )
        )
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
        self.render_product_state: dict[str, Any] | None = None

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
        timeline_time = float(self.timeline.get_current_time())
        timeline_tcps = float(self.timeline.get_time_codes_per_second())
        source_time_code = float(self.timeline.time_to_time_code(timeline_time))
        if (
            not math.isfinite(timeline_tcps)
            or timeline_tcps <= 0.0
            or not math.isclose(
                timeline_tcps,
                self._stage_time_codes_per_second,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isfinite(source_time_code)
            or source_time_code != 1.0
        ):
            raise RuntimeError("live timeline time-code readback drifted")
        return {
            "physics_steps": direct,
            "direct_step_calls": direct,
            "physics_callbacks": len(self._callback_records),
            "simulation_time_s": simulation_time,
            "timeline_time_s": timeline_time,
            "timeline_playing": bool(self.timeline.is_playing()),
            "source_time_code": int(source_time_code),
        }

    def counter_state(self) -> Mapping[str, Any]:
        return self._counter_state()

    def attest_state(self) -> Mapping[str, Any]:
        prims, properties = self._session_inventory()
        if self.render_product_state is None:
            raise RuntimeError("render product baseline is unavailable")
        live_render = self._read_live_render_product_state()
        if live_render != self.render_product_state:
            raise RuntimeError("live render product drifted from its pre-output baseline")
        runtime_prims = set(live_render["runtime_prim_paths"])
        runtime_properties = set(live_render["runtime_property_paths"])
        scene_prims = sorted(set(prims) - runtime_prims)
        scene_properties = sorted(set(properties) - runtime_properties)
        non_session = {
            str(layer.identifier): hashlib.sha256(
                layer.ExportToString().encode("utf-8")
            ).hexdigest()
            for layer in self.stage.GetUsedLayers()
            if layer != self.session
        }
        source_paths = tuple(Path(path) for path in self.loaded["source_layer_paths"])
        source_files = {os.fspath(path.resolve()): _sha256(path) for path in source_paths}
        source_files[os.fspath(Path(self.loaded["receipt_path"]).resolve())] = _sha256(
            Path(self.loaded["receipt_path"])
        )
        return {
            "session_prim_paths": prims,
            "session_property_paths": properties,
            "scene_prim_paths": scene_prims,
            "scene_property_paths": scene_properties,
            "runtime_render_state": live_render,
            "session_sha256": hashlib.sha256(
                self.session.ExportToString().encode("utf-8")
            ).hexdigest(),
            "non_session_layer_sha256": non_session,
            "source_file_sha256": source_files,
        }

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
        receipt = self.loaded["receipt"]
        count_body = receipt.get("count_driven_liquid_body")
        capture_rows = count_body.get("capture_geometry") if isinstance(count_body, Mapping) else None
        final_step = self.loaded["final_physics_step"]
        selected_rows = [
            row
            for row in capture_rows or []
            if isinstance(row, Mapping) and row.get("physics_step") == final_step
        ]
        semantics = receipt.get("semantics")
        pairs = semantics.get("pairs") if isinstance(semantics, Mapping) else None
        selected_pairs = [
            row
            for row in pairs or []
            if isinstance(row, Mapping) and row.get("physics_step") == final_step
        ]
        plan = receipt.get("plan")
        palette = plan.get("run_palette") if isinstance(plan, Mapping) else None
        transmissions = palette.get("transmission_1cm") if isinstance(palette, Mapping) else None
        treatment = plan.get("render_treatment") if isinstance(plan, Mapping) else None
        if (
            not isinstance(count_body, Mapping)
            or len(selected_rows) != 1
            or len(selected_pairs) != 1
            or not isinstance(transmissions, Mapping)
            or set(transmissions) != set(self.f2.COMPOSITION_COMPONENTS)
            or not isinstance(treatment, Mapping)
            or type(treatment.get("particle_volume_m3")) is not float
        ):
            raise ValueError("accepted count-body reconstruction contract is malformed")
        transmission_matrix = np.asarray(
            [transmissions[tank] for tank in self.f2.COMPOSITION_COMPONENTS],
            dtype=np.float64,
        )
        pair = selected_pairs[0]
        material_state = pair.get("body_material_render_state")
        if not isinstance(material_state, Mapping):
            raise ValueError("accepted count-body material state is missing")
        with self.Usd.EditContext(self.stage, self.session):
            self.UsdGeom.Scope.Define(self.stage, MATERIAL_NAMESPACE_PATH)
            material = self.UsdShade.Material.Define(self.stage, SPATIAL_MATERIAL_PATH)
            global_material = self.UsdShade.Material.Define(self.stage, GLOBAL_MATERIAL_PATH)
            result = _run_qualified_count_body_reconstruction(
                helpers=self.f2,
                spatial_args=(
                    self.stage,
                    material,
                    tuple(float(value) for value in transmission_matrix[0]),
                    self.Sdf,
                    self.Gf,
                    self.UsdShade,
                ),
                global_args=(
                    self.stage,
                    global_material,
                    tuple(float(value) for value in transmission_matrix[0]),
                    self.Sdf,
                    self.Gf,
                    self.UsdShade,
                ),
                global_semantics_args=(global_material.GetPrim(), self.UsdSemantics),
                chain_kwargs={
                    "arrays": self.loaded["arrays"],
                    "accepted_source_evidence": count_body,
                    "accepted_capture_geometry": selected_rows[0],
                    "accepted_material_state": material_state,
                    "final_physics_step": final_step,
                    "source_transmissions": transmission_matrix,
                    "author_kwargs": {
                        "stage": self.stage,
                        "body_material": material,
                        "height_banded": True,
                        "Usd": self.Usd,
                        "UsdGeom": self.UsdGeom,
                        "UsdShade": self.UsdShade,
                        "UsdSemantics": self.UsdSemantics,
                        "LegacySemantics": self.LegacySemantics,
                        "Sdf": self.Sdf,
                        "Gf": self.Gf,
                        "Vt": self.Vt,
                        "requested_source_time_code": self.f2.COUNT_DRIVEN_SOURCE_TIME_CODE,
                        "particle_volume_m3": treatment["particle_volume_m3"],
                    },
                    "update_kwargs": {
                        "Usd": self.Usd,
                        "UsdGeom": self.UsdGeom,
                        "Gf": self.Gf,
                        "Vt": self.Vt,
                    },
                },
            )
            def network(material: Any, shader: Any, input_names: Sequence[str]) -> dict[str, Any]:
                inputs = {
                    name: self.f2._serial_shader_input(shader.GetInput(name).Get(), name=name)
                    for name in input_names
                }
                outputs: dict[str, str] = {}
                for name, output in (
                    ("surface", material.GetSurfaceOutput("mdl")),
                    ("volume", material.GetVolumeOutput("mdl")),
                    ("displacement", material.GetDisplacementOutput("mdl")),
                ):
                    connected = output.GetConnectedSource()
                    if not isinstance(connected, tuple) or len(connected) < 2:
                        raise RuntimeError("material output is not connected to the accepted shader")
                    outputs[name] = f"{connected[0].GetPrim().GetPath()}.outputs:{connected[1]}"
                source_asset = shader.GetSourceAsset("mdl")
                return {
                    "implementation_source": str(shader.GetImplementationSource()),
                    "source_asset": str(source_asset.path),
                    "subidentifier": str(shader.GetSourceAssetSubIdentifier("mdl")),
                    "inputs": inputs,
                    "material_outputs": outputs,
                    "shader_output_type": str(shader.GetOutput("out").GetTypeName()),
                }
            spatial_network = network(
                material, result["spatial_shader"], ("fallback_color",)
            )
            spatial_network.update(
                {
                    "shader_inputs_sha256": result["material_state"]["shader_inputs_sha256"],
                    "display_color_sha256": result["material_state"]["display_color_sha256"],
                }
            )
            global_network = network(
                global_material,
                result["global_shader"],
                self.f2._OMNISURFACE_RENDER_INPUTS,
            )
            global_network["body_material_color_sha256"] = selected_rows[0][
                "body_material_color_sha256"
            ]
            self.material_network_state = _validate_material_network_state(
                {"spatial": spatial_network, "global": global_network},
                accepted_material_state=material_state,
                accepted_capture_geometry=selected_rows[0],
                spatial_source_asset=self.f2.SPATIAL_OMNISURFACE_MDL.name,
                spatial_subidentifier=self.f2.SPATIAL_OMNISURFACE_SUBIDENTIFIER,
            )
        self.count_body_runtime = result["runtime"]
        self.count_body_material_state = result["material_state"]

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

    def _create_render_product(self) -> dict[str, Any]:
        self.settings.set_string(self.f2.RTX_RENDER_MODE_SETTING, "RaytracedLighting")
        if str(self.settings.get_as_string(self.f2.RTX_RENDER_MODE_SETTING)) != "RaytracedLighting":
            raise RuntimeError("RaytracedLighting readback failed")
        self.settings.set_int("/rtx/post/tonemap/op", 4)
        self.settings.set_bool("/rtx/post/histogram/enabled", False)
        name = "FluidmixRigidTransport_" + uuid.uuid4().hex
        self.render_product = self.rep.create.render_product(
            CAMERA_PATH, (640, 480), name=name
        )
        self.render_product_state = self._read_live_render_product_state()
        return copy.deepcopy(self.render_product_state)

    def _read_live_render_product_state(self) -> dict[str, Any]:
        if self.render_product is None:
            raise RuntimeError("live render product is missing")
        product_path = str(self.render_product.path)
        product_prim = self.stage.GetPrimAtPath(product_path)
        if not product_prim.IsValid():
            raise RuntimeError("live render product prim is missing")
        product = self.UsdRender.Product(product_prim)
        camera_targets = [str(path) for path in product.GetCameraRel().GetTargets()]
        resolution = product.GetResolutionAttr().Get()
        prims, properties = self._session_inventory()
        runtime_prims = sorted(path for path in prims if path == "/Render" or path.startswith("/Render/"))
        runtime_properties = sorted(path for path in properties if path.startswith("/Render/"))
        if runtime_prims != ["/Render", product_path] or camera_targets != [CAMERA_PATH]:
            raise RuntimeError("live render product authored an unexpected runtime partition")
        if any(spec.layer != self.session for spec in product_prim.GetPrimStack()):
            raise RuntimeError("live render product has a non-session spec")
        state = {
            "product_path": product_path,
            "product_type": str(product_prim.GetTypeName()),
            "runtime_prim_paths": runtime_prims,
            "runtime_property_paths": runtime_properties,
            "camera_target": camera_targets[0],
            "resolution": [int(resolution[0]), int(resolution[1])],
            "session_layer_only": True,
        }
        state["baseline_sha256"] = hashlib.sha256(_canonical_json(state)).hexdigest()
        return _validate_render_product_state(state)

    def _prepare_capture(self) -> None:
        if self.render_product is None or self.render_product_state is None:
            raise RuntimeError("live render product was not prepared")
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
        self.settings.set_string(self.f2.RTX_RENDER_MODE_SETTING, "RaytracedLighting")
        self.settings.set_int("/rtx/post/tonemap/op", 4)
        self.settings.set_bool("/rtx/post/histogram/enabled", False)
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
        frame_zero_diff = dict(self.author_frame(0))
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
                "isaac_sim_version": self.live_isaac_version,
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
            "frame_zero_diff": frame_zero_diff,
            "render_product_state": {
                "constructed": self.render_product is not None,
                "camera_path": CAMERA_PATH,
                "resolution": [640, 480],
            },
            "writer_constructed": any(
                value is not None for value in (self.rgb_annotator, self.instance_annotator)
            ),
        }

    def prepare_render_product(self) -> Mapping[str, Any]:
        if self.groups is None or self.plan is None:
            raise RuntimeError("runtime preflight has not completed")
        if self.render_product is not None:
            raise RuntimeError("transport render product was already prepared")
        return self._create_render_product()

    def prepare_capture(self) -> None:
        if self.groups is None or self.plan is None:
            raise RuntimeError("runtime preflight has not completed")
        if self.render_product is None or self.render_product_state is None:
            raise RuntimeError("transport render product was not prepared")
        if self.rgb_annotator is not None or self.instance_annotator is not None:
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
        reconstruction = sorted(COUNT_BODY_RECONSTRUCTION_PROPERTY_PATHS)
        transport_properties = {
            property_path
            for path in authored_local
            for property_path in (
                f"{path}.xformOp:translate:fluidmixTransport",
                f"{path}.xformOpOrder",
            )
        }
        expected_prims = sorted(
            {
                "/FillStation",
                "/FillStation/FillStation_LocalRoot",
                *authored_local,
                *COUNT_BODY_RECONSTRUCTION_PRIM_PATHS,
            }
        )
        expected_properties = sorted(transport_properties | set(reconstruction))
        actual_prims, actual_properties = self._session_inventory()
        if self.render_product_state is None:
            scene_prims = actual_prims
            scene_properties = actual_properties
        else:
            live_render = self._read_live_render_product_state()
            if live_render != self.render_product_state:
                raise RuntimeError("live render product drifted during transport authoring")
            scene_prims = sorted(
                set(actual_prims) - set(live_render["runtime_prim_paths"])
            )
            scene_properties = sorted(
                set(actual_properties) - set(live_render["runtime_property_paths"])
            )
        self._assert_source_layers_unchanged()
        if scene_prims != expected_prims or scene_properties != expected_properties:
            raise RuntimeError("anonymous session layer contains an unauthorized authored property")
        prim_types = {
            path: str(self.stage.GetPrimAtPath(path).GetTypeName())
            for path in COUNT_BODY_RECONSTRUCTION_PRIM_PATHS
        }
        applied_schemas = {
            path: sorted(str(value) for value in self.stage.GetPrimAtPath(path).GetAppliedSchemas())
            for path in COUNT_BODY_RECONSTRUCTION_PRIM_PATHS
        }
        if (
            prim_types != MATERIAL_RECONSTRUCTION_PRIM_TYPES
            or applied_schemas != MATERIAL_RECONSTRUCTION_APPLIED_SCHEMAS
        ):
            raise RuntimeError("reconstruction prim type or applied API inventory drifted")
        return {
            "edit_target": "anonymous_session_layer",
            "source_time_code": 1,
            "source_layer_property_paths": [],
            "root_layer_property_paths": [],
            "session_prim_paths": scene_prims,
            "count_body_reconstruction_prim_paths": list(COUNT_BODY_RECONSTRUCTION_PRIM_PATHS),
            "session_property_paths": scene_properties,
            "count_body_reconstruction_property_paths": reconstruction,
            "reconstruction_prim_type_by_path": prim_types,
            "reconstruction_applied_schemas_by_path": applied_schemas,
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
        guard = self.f2._replicator_same_state_step_checked_in_session(
            stage=self.stage,
            session_layer=self.session,
            Usd=self.Usd,
            rep=self.rep,
            rt_subframes=32,
            callback_count=lambda: len(self._callback_records),
            timeline=self.timeline,
            physics_state=lambda: {
                "num_physics_steps": int(self.SimulationManager.get_num_physics_steps()),
                "simulation_time_s": float(self.SimulationManager.get_simulation_time()),
            },
        )
        rgb = _validate_runtime_rgb(self.rgb_annotator.get_data())
        instance_raw = self.instance_annotator.get_data()
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
            "rgb": rgb,
            "outgoing_mask": outgoing,
            "incoming_mask": incoming,
            "instance_ids": np.asarray(data, dtype=np.uint32),
            "instance_id_to_path": mapping,
            "counter_state_before": before,
            "counter_state_after": after,
            "same_state_guard": guard,
        }

    def close(self) -> None:
        if self.rgb_annotator is not None and self.render_product is not None:
            self.rgb_annotator.detach([self.render_product])
            self.rgb_annotator = None
        if self.instance_annotator is not None and self.render_product is not None:
            self.instance_annotator.detach([self.render_product])
            self.instance_annotator = None
        self._physics_step_subscription = None


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
    runtime: TransportRuntime | None = None
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
        close_runtime = getattr(runtime, "close", None)
        try:
            if callable(close_runtime):
                close_runtime()
        finally:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
