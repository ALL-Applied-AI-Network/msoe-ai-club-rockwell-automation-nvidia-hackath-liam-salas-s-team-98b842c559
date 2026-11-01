"""Strict contracts for the frozen, acceptance-only appearance pilot.

This module deliberately keeps deployable RGB inputs separate from oracle
labels.  The pilot validates renderer plumbing; it is not a training set and
does not attach physical meaning to its direct appearance controls.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess
from typing import Mapping

import numpy as np
from PIL import Image

from fluidmix.core.paths import REPO
from ..render.render_process import (
    CHILD_ENVIRONMENT_PROFILES,
    capture_requested_settings,
    hash_regular_file,
    load_strict_json_snapshot,
    verify_capture_summary,
)


CONFIG_PATH = REPO / "data/twin_appearance_pilot_v1.json"
CONFIG_ID = "twin-appearance-pilot-v1"
DATASET_VERSION = "twin-appearance-pilot-v1"
GRAPH_FILE = "experiment-graph.json"
MODEL_INPUT_FILE = "model_input.jsonl"
LABEL_PROVENANCE_FILE = "label_provenance.jsonl"
RECEIPT_FILE = "dataset-receipt.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ATTEMPT_NAME = re.compile(r"R[0-9]{3}.*")
_MATERIAL_PATH = "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF"
DEFAULT_ENVIRONMENT_PROFILE = "gb10-aarch64"
_IMPLEMENTATION_PATHS = (
    "src/twin/probes/pilot_dataset.py",
    "src/twin/probes/pilot_diagnostic.py",
    "src/twin/render/render_process.py",
    "cli/twin/render/render_twin_dataset.py",
    "cli/twin/render/twin_render_sequence.py",
    "cli/twin/probes/build_twin_pilot_dataset.py",
    "cli/twin/probes/check_render_dataset.py",
    "cli/twin/probes/evaluate_appearance_pilot.py",
)

_A = [0.006, 0.2, 0.3]
_B = [0.45, 0.12, 0.01]
_M010 = [0.0504, 0.192, 0.271]
_M030 = [0.1392, 0.176, 0.213]
_S0001_ML = {"t100": 0, "t200": 10, "t400": 20}
_S0436_ML = {"t100": 60, "t200": 10, "t400": 0}


def _row(
    sample_id: str,
    pair_id: str,
    sequence_id: str,
    event_id: str,
    *,
    current_recipe: str,
    previous_recipe: str,
    run_position: int,
    recipe_changed: bool,
    appearance_id: str,
    alpha: float | None,
    injected: bool,
) -> dict[str, object]:
    programmed = {"S0001": _S0001_ML, "S0436": _S0436_ML}
    return {
        "sample_id": sample_id,
        "pair_id": pair_id,
        "sequence_id": sequence_id,
        "event_id": event_id,
        "counterfactual_id": pair_id,
        "partition": "train",
        "current_sim_recipe_id": current_recipe,
        "previous_sim_recipe_id": previous_recipe,
        "current_programmed_ml": dict(programmed[current_recipe]),
        "previous_programmed_ml": dict(programmed[previous_recipe]),
        "run_position": run_position,
        "recipe_changed": recipe_changed,
        "nuisance_id": "N000",
        "appearance_id": appearance_id,
        "appearance_blend_alpha": alpha,
        "injected_appearance": injected,
        "recipe_appearance_mapping_validated": False,
        "layer_id": sample_id,
    }


def canonical_pilot_config() -> dict[str, object]:
    """Return the one frozen schema-1 pilot configuration."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "config_id": CONFIG_ID,
        "dataset_version": DATASET_VERSION,
        "purpose": "acceptance_only",
        "training_eligible": False,
        "recipe_appearance_mapping_validated": False,
        "programmed_dispense_order": ["t200", "t400", "t100"],
        "sources": {
            "recipe_table": "data/sim_recipes.csv",
            "transition_table": "data/sim_transitions.csv",
            "recipes": {
                "S0001": {
                    "partition": "train",
                    "exact_row": "S0001,train,0.0-0.3-0.7,0,10,20,30\n",
                    "row_sha256": "72f1b918272723c9ece3a62c4f22bdd2c135348d0ab6d52554a845891e6d30bf",
                    "programmed_ml": dict(_S0001_ML),
                },
                "S0436": {
                    "partition": "train",
                    "exact_row": "S0436,train,0.9-0.1-0.0,60,10,0,70\n",
                    "row_sha256": "f066c88f55c9f07d719b0eafde4db2ab26e2c0b98a17f80796b409d83b3e7a0b",
                    "programmed_ml": dict(_S0436_ML),
                },
            },
            "transition": {
                "partition": "train",
                "previous_sim_recipe_id": "S0436",
                "current_sim_recipe_id": "S0001",
                "exact_row": "train,S0436,S0001\n",
                "row_sha256": "712e8b1db445cf1c3ab74790ea4d70c03248688c89691ebcbc404b6a5ff97eac",
            },
        },
        "controls": {
            "nuisance_id": "N000",
            "augmentation": "none",
            "camera": "FillCam",
            "view_id": "vial_fill",
            "time_code": 686,
            "frame_spec": "686:687:1",
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "subframes": 64,
            "warmup": 60,
            "max_tries": 20,
            "require_liquid_pixels": True,
            "timeout_seconds": 1800,
            "seed": 310241,
            "process_attempts": 8,
            "replacement_process_attempts": 0,
        },
        "endpoints": {
            "A": {"linear_diffuse_rgb": list(_A), "opacity": 0.6},
            "B": {"linear_diffuse_rgb": list(_B), "opacity": 0.6},
        },
        "appearances": {
            "A": {
                "construction": "endpoint_a",
                "linear_diffuse_rgb": list(_A),
                "opacity": 0.6,
            },
            "B": {
                "construction": "endpoint_b",
                "linear_diffuse_rgb": list(_B),
                "opacity": 0.6,
            },
            "M010": {
                "construction": "linear_interpolation_a_to_b",
                "appearance_blend_alpha": 0.10,
                "linear_diffuse_rgb": list(_M010),
                "opacity": 0.6,
            },
            "M030": {
                "construction": "linear_interpolation_a_to_b",
                "appearance_blend_alpha": 0.30,
                "linear_diffuse_rgb": list(_M030),
                "opacity": 0.6,
            },
        },
        "rows": [
            _row("R000", "P000", "Q000", "E000", current_recipe="S0436", previous_recipe="S0436", run_position=3, recipe_changed=False, appearance_id="B", alpha=None, injected=False),
            _row("R001", "P000", "Q001", "E000", current_recipe="S0436", previous_recipe="S0436", run_position=3, recipe_changed=False, appearance_id="B", alpha=None, injected=False),
            _row("R002", "P001", "Q001", "E001", current_recipe="S0001", previous_recipe="S0436", run_position=1, recipe_changed=True, appearance_id="M030", alpha=0.30, injected=True),
            _row("R003", "P001", "Q000", "E001", current_recipe="S0001", previous_recipe="S0436", run_position=1, recipe_changed=True, appearance_id="A", alpha=0.0, injected=False),
            _row("R004", "P002", "Q000", "E002", current_recipe="S0001", previous_recipe="S0436", run_position=2, recipe_changed=False, appearance_id="A", alpha=0.0, injected=False),
            _row("R005", "P002", "Q001", "E002", current_recipe="S0001", previous_recipe="S0436", run_position=2, recipe_changed=False, appearance_id="M010", alpha=0.10, injected=True),
            _row("R006", "P003", "Q001", "E003", current_recipe="S0001", previous_recipe="S0436", run_position=3, recipe_changed=False, appearance_id="A", alpha=0.0, injected=False),
            _row("R007", "P003", "Q000", "E003", current_recipe="S0001", previous_recipe="S0436", run_position=3, recipe_changed=False, appearance_id="A", alpha=0.0, injected=False),
        ],
    }
    return deepcopy(payload)


def _exact_equal(left: object, right: object) -> bool:
    """JSON equality that never treats Boolean, integer, and float as interchangeable."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        assert isinstance(right, dict)
        return set(left) == set(right) and all(
            _exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        assert isinstance(right, list)
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def validate_pilot_config(payload: object) -> dict[str, object]:
    """Reject every schema drift from the reviewed pilot configuration."""

    expected = canonical_pilot_config()
    if not isinstance(payload, Mapping) or not _exact_equal(dict(payload), expected):
        raise ValueError("pilot config must exactly match the frozen schema-1 matrix")
    return deepcopy(expected)


def _verify_source_rows(payload: Mapping[str, object], repo_root: Path) -> None:
    sources = payload["sources"]
    assert isinstance(sources, Mapping)
    checks: list[tuple[str, Mapping[str, object]]] = []
    recipes = sources["recipes"]
    assert isinstance(recipes, Mapping)
    for recipe_id in ("S0001", "S0436"):
        record = recipes[recipe_id]
        assert isinstance(record, Mapping)
        checks.append((str(sources["recipe_table"]), record))
    transition = sources["transition"]
    assert isinstance(transition, Mapping)
    checks.append((str(sources["transition_table"]), transition))
    for relative, record in checks:
        candidate = repo_root / relative
        exact = str(record["exact_row"]).encode("utf-8")
        rows = candidate.read_bytes().splitlines(keepends=True)
        if rows.count(exact) != 1:
            raise ValueError(f"frozen source row must occur exactly once in {relative}")
        if hashlib.sha256(exact).hexdigest() != record["row_sha256"]:
            raise ValueError(f"frozen source row hash is inconsistent for {relative}")


def _verify_split_contract(payload: Mapping[str, object], repo_root: Path) -> None:
    spec, _ = load_strict_json_snapshot(
        repo_root / "data/sim_split_spec.json", max_bytes=1024 * 1024
    )
    if not isinstance(spec, Mapping):
        raise ValueError("split specification must be an object")
    expected_ranges = {
        "train": [0, 999999],
        "val": [1000000, 1999999],
        "test": [2000000, 2999999],
    }
    if (
        spec.get("version") != "v2"
        or not _exact_equal(spec.get("tanks"), ["t100", "t200", "t400"])
        or not _exact_equal(spec.get("render_seed_ranges"), expected_ranges)
        or not _exact_equal(
            spec.get("near_rule"),
            {"ml_per_tank": 3, "fraction_per_tank": 0.05},
        )
    ):
        raise ValueError("current split seed/range contract drifted")
    seed = payload["controls"]["seed"]  # type: ignore[index]
    if type(seed) is not int or not expected_ranges["train"][0] <= seed <= expected_ranges["train"][1]:
        raise ValueError("pilot seed is outside the train partition range")
    reserved = spec.get("real_recipes_reserved")
    if not isinstance(reserved, Mapping) or not reserved:
        raise ValueError("reserved real recipe contract is missing")
    recipes = payload["sources"]["recipes"]  # type: ignore[index]
    assert isinstance(recipes, Mapping)

    def fractions(values: Mapping[str, object]) -> dict[str, float]:
        total = sum(int(values[tank]) for tank in ("t100", "t200", "t400"))
        if total <= 0:
            raise ValueError("recipe total must be positive")
        return {tank: int(values[tank]) / total for tank in ("t100", "t200", "t400")}

    for recipe_id, recipe in recipes.items():
        assert isinstance(recipe, Mapping)
        programmed = recipe["programmed_ml"]
        assert isinstance(programmed, Mapping)
        programmed_fraction = fractions(programmed)
        for real_id, real in reserved.items():
            if (
                not isinstance(real_id, str)
                or not isinstance(real, Mapping)
                or set(real) != {"t100", "t200", "t400"}
                or any(type(real[tank]) is not int or real[tank] < 0 for tank in real)
            ):
                raise ValueError("reserved real recipe record is malformed")
            real_fraction = fractions(real)
            ml_near = all(
                abs(int(programmed[tank]) - int(real[tank])) <= 3
                for tank in ("t100", "t200", "t400")
            )
            fraction_near = all(
                abs(programmed_fraction[tank] - real_fraction[tank]) <= 0.05
                for tank in ("t100", "t200", "t400")
            )
            if ml_near or fraction_near:
                raise ValueError(
                    f"pilot recipe {recipe_id} is near reserved real recipe {real_id}"
                )


def load_pilot_config(
    path: str | Path = CONFIG_PATH, *, repo_root: str | Path = REPO
) -> dict[str, object]:
    """Load, hash, validate, and bind the tracked config to committed split rows."""

    payload, snapshot = load_strict_json_snapshot(path, max_bytes=1024 * 1024)
    validated = validate_pilot_config(payload)
    resolved_repo = Path(repo_root).resolve(strict=True)
    _verify_source_rows(validated, resolved_repo)
    _verify_split_contract(validated, resolved_repo)
    return {
        "path": str(Path(path).resolve(strict=True)),
        "sha256": snapshot["sha256"],
        "bytes": snapshot["bytes"],
        "payload": validated,
    }


def _require_root(path: str | Path, *, role: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        metadata = os.lstat(candidate)
    except OSError as error:
        raise ValueError(f"{role} is unavailable: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{role} must be an existing non-symlink directory")
    return candidate.resolve(strict=True)


def _safe_relative(value: object, *, prefix: str | None = None) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact path must be a nonempty relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or (prefix is not None and (not relative.parts or relative.parts[0] != prefix))
    ):
        raise ValueError("artifact path escapes its expected relative root")
    return relative


def _snapshot_relative_regular(
    root: Path, relative_text: str, *, prefix: str | None = None
) -> dict[str, object]:
    relative = _safe_relative(relative_text, prefix=prefix)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise ValueError(f"artifact is unavailable: {relative_text}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"artifact path contains a symlink: {relative_text}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"artifact is not a regular file: {relative_text}")
    snapshot = hash_regular_file(current)
    return {"path": relative_text, **snapshot}


def _write_exclusive(path: Path, data: bytes) -> Path:
    if path.parent.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise ValueError("output parent must be a canonical non-symlink directory")
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    temporary_created = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        temporary_created = True
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary_created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return path


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


_ATTEMPT_FIELDS = {
    "schema_version",
    "run_id",
    "hypothesis",
    "falsifier",
    "command_plan",
    "seed",
    "denominators",
    "stop_conditions",
}
_COMMAND_PLAN_FIELDS = {
    "entry_point",
    "implementation_commit",
    "environment_profile",
    "command",
    "appearance_layer",
    "requested_controls",
    "process_attempt_limit",
    "replacement_process_attempts",
}
_APPEARANCE_LAYER_FIELDS = {
    "schema",
    "variant_id",
    "stage_sha256",
    "stage_bytes",
    "source_stage",
    "source_stage_sha256",
    "material_path",
    "diffuse_color_linear",
    "opacity",
}
_INSPECTION_FIELDS = _APPEARANCE_LAYER_FIELDS | {
    "path",
    "uncalibrated",
    "timing",
    "non_session_usd_layer_stack",
}


def _current_git_head(repo_root: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("could not verify the implementation commit") from error


def _validate_inspection_record(
    payload: object,
    *,
    root: Path,
    row: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != _INSPECTION_FIELDS:
        raise ValueError("appearance inspection record has wrong exact fields")
    sample_id = str(row["sample_id"])
    layer = _snapshot_relative_regular(
        root, f"appearance-layers/{sample_id}.usda", prefix="appearance-layers"
    )
    appearance = config["appearances"][row["appearance_id"]]  # type: ignore[index]
    expected_plan = {
        "schema": "direct_appearance_v1",
        "variant_id": sample_id,
        "stage_sha256": layer["sha256"],
        "stage_bytes": layer["bytes"],
        "source_stage": "stations/fill_station/fill_station.usda",
        "source_stage_sha256": payload.get("source_stage_sha256"),
        "material_path": _MATERIAL_PATH,
        "diffuse_color_linear": appearance["linear_diffuse_rgb"],  # type: ignore[index]
        "opacity": appearance["opacity"],  # type: ignore[index]
    }
    plan = {key: payload[key] for key in _APPEARANCE_LAYER_FIELDS}
    if (
        payload.get("path") != str((root / str(layer["path"])).resolve(strict=True))
        or payload.get("uncalibrated") is not True
        or not _exact_equal(plan, expected_plan)
    ):
        raise ValueError("appearance inspection does not match the frozen pilot row")
    source_hash = plan["source_stage_sha256"]
    if not isinstance(source_hash, str) or _SHA256.fullmatch(source_hash) is None:
        raise ValueError("appearance inspection source-stage hash is malformed")
    timing = payload["timing"]
    if (
        not isinstance(timing, Mapping)
        or set(timing) != {
            "start_time_code",
            "end_time_code",
            "frames_per_second",
            "time_codes_per_second",
        }
        or any(type(value) is not float or not np.isfinite(value) for value in timing.values())
    ):
        raise ValueError("appearance inspection timing is malformed")
    stack = payload["non_session_usd_layer_stack"]
    if not isinstance(stack, list) or len(stack) < 2:
        raise ValueError("appearance inspection layer stack is incomplete")
    for ordinal, record in enumerate(stack, start=1):
        if (
            not isinstance(record, Mapping)
            or set(record) != {"ordinal", "resolved_path", "sha256", "bytes"}
            or record["ordinal"] != ordinal
            or type(record["ordinal"]) is not int
            or not isinstance(record["resolved_path"], str)
            or not Path(record["resolved_path"]).is_absolute()
            or not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or type(record["bytes"]) is not int
            or record["bytes"] <= 0
        ):
            raise ValueError("appearance inspection layer stack is malformed")
    if (
        stack[0]["resolved_path"] != payload["path"]
        or stack[0]["sha256"] != layer["sha256"]
        or stack[0]["bytes"] != layer["bytes"]
        or stack[1]["sha256"] != source_hash
        or not stack[1]["resolved_path"].endswith("/" + str(plan["source_stage"]))
    ):
        raise ValueError("appearance inspection layer stack does not bind layer/source")
    return deepcopy(plan)


def write_attempt_manifests(
    pilot_root: str | Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    implementation_commit: str,
    controller_python: str | Path,
    isaac_python: str | Path,
    runtime_lock: str | Path,
    twin: str | Path,
    environment_profile: str = DEFAULT_ENVIRONMENT_PROFILE,
    repo_root: str | Path = REPO,
) -> tuple[Path, ...]:
    """Deterministically author the exact eight no-replace attempt manifests."""

    root = _require_root(pilot_root, role="pilot root")
    if (
        type(environment_profile) is not str
        or environment_profile not in CHILD_ENVIRONMENT_PROFILES
    ):
        raise ValueError(
            f"environment profile must be one of {CHILD_ENVIRONMENT_PROFILES!r}"
        )
    if implementation_commit != _current_git_head(repo_root):
        raise ValueError("implementation commit does not match repository HEAD")
    config_record = load_pilot_config(config_path, repo_root=repo_root)
    config = config_record["payload"]
    assert isinstance(config, Mapping)
    rows = config["rows"]
    assert isinstance(rows, list)
    with os.scandir(root) as entries:
        if any(_ATTEMPT_NAME.fullmatch(entry.name) for entry in entries):
            raise ValueError("attempt manifests must be authored before attempt 1")
    inspection_dir = root / "layer-inspections"
    expected_inspections = sorted(f"{row['sample_id']}.json" for row in rows)
    try:
        inspection_names = sorted(entry.name for entry in os.scandir(inspection_dir))
    except OSError as error:
        raise ValueError("layer-inspections directory is unavailable") from error
    if inspection_names != expected_inspections:
        raise ValueError("layer-inspections must contain exactly the eight Task 3A records")

    runtime_values = [controller_python, isaac_python, runtime_lock, twin]
    normalized_runtime: list[str] = []
    for value in runtime_values:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise ValueError("manifest runtime paths must be absolute")
        normalized_runtime.append(str(Path(os.path.abspath(candidate))))
    controller_text, isaac_text, lock_text, twin_text = normalized_runtime

    manifest_dir = root / "attempt-manifests"
    if manifest_dir.exists() or manifest_dir.is_symlink():
        metadata = os.lstat(manifest_dir)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("attempt-manifests must be an ordinary directory")
        if any(os.scandir(manifest_dir)):
            raise FileExistsError(manifest_dir)
    else:
        manifest_dir.mkdir(mode=0o700)

    payloads: list[tuple[Path, dict[str, object]]] = []
    for row in rows:
        assert isinstance(row, Mapping)
        sample_id = str(row["sample_id"])
        inspection, _ = load_strict_json_snapshot(
            inspection_dir / f"{sample_id}.json", max_bytes=4 * 1024 * 1024
        )
        appearance_plan = _validate_inspection_record(
            inspection, root=root, row=row, config=config
        )
        manifest_path = manifest_dir / f"{sample_id}.json"
        payload = {
            "schema_version": 1,
            "run_id": sample_id,
            "hypothesis": "The fixed renderer emits one mechanically valid pilot capture.",
            "falsifier": "Any lifecycle, provenance, or capture gate fails.",
            "command_plan": {
                "entry_point": "cli/twin/render/render_twin_dataset.py",
                "implementation_commit": implementation_commit,
                "environment_profile": environment_profile,
                "command": [
                    controller_text,
                    "cli/twin/render/render_twin_dataset.py",
                    "--isaac-python",
                    isaac_text,
                    "--runtime-lock",
                    lock_text,
                    "--experiment-manifest",
                    str(manifest_path.resolve()),
                    "--twin",
                    twin_text,
                    "--run-id",
                    sample_id,
                    "--out-parent",
                    str(root),
                    "--appearance-stage",
                    str((root / "appearance-layers" / f"{sample_id}.usda").resolve()),
                    "--frames",
                    "686:687:1",
                    "--subframes",
                    "64",
                    "--warmup",
                    "60",
                    "--max-tries",
                    "20",
                    "--seed",
                    "310241",
                    "--require-liquid-pixels",
                    "--renderer",
                    "PathTracing",
                    "--timeout",
                    "1800",
                    "--environment-profile",
                    environment_profile,
                ],
                "appearance_layer": appearance_plan,
                "requested_controls": deepcopy(config["controls"]),
                "process_attempt_limit": 1,
                "replacement_process_attempts": 0,
            },
            "seed": 310241,
            "denominators": {"pilot_rows": 8, "requested_frames": 1, "process_attempts": 1},
            "stop_conditions": [
                "no replacement process attempt",
                "bounded timeout",
                "provenance or safety failure",
            ],
        }
        payloads.append((manifest_path, payload))
    for path, payload in payloads:
        _write_exclusive(path, _json_bytes(payload))
    return tuple(path for path, _ in payloads)


def _validate_attempt_manifest(
    payload: object,
    *,
    row: Mapping[str, object],
    config: Mapping[str, object],
    implementation_commit: str,
    layer: Mapping[str, object],
    layer_path: Path,
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != _ATTEMPT_FIELDS:
        raise ValueError("attempt manifest has wrong exact fields")
    sample_id = row["sample_id"]
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise ValueError("attempt manifest schema version must be exactly 1")
    if payload["run_id"] != sample_id:
        raise ValueError("attempt manifest run ID does not match pilot row")
    for field in ("hypothesis", "falsifier"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"attempt manifest {field} must be nonblank")
    plan = payload["command_plan"]
    if not isinstance(plan, Mapping) or set(plan) != _COMMAND_PLAN_FIELDS:
        raise ValueError("attempt command plan has wrong exact fields")
    if plan["entry_point"] != "cli/twin/render/render_twin_dataset.py":
        raise ValueError("attempt command plan entry point is wrong")
    if plan["implementation_commit"] != implementation_commit:
        raise ValueError("attempt command plan commit is wrong")
    environment_profile = plan["environment_profile"]
    if (
        type(environment_profile) is not str
        or environment_profile not in CHILD_ENVIRONMENT_PROFILES
    ):
        raise ValueError("attempt command plan environment profile is unsupported")
    command = plan["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(piece, str) or not piece for piece in command)
    ):
        raise ValueError("attempt command must be a nonempty argv list")
    root = layer_path.parent.parent
    expected_manifest = root / "attempt-manifests" / f"{sample_id}.json"
    expected_command = [
        command[0] if command else None,
        "cli/twin/render/render_twin_dataset.py",
        "--isaac-python",
        command[3] if len(command) > 3 else None,
        "--runtime-lock",
        command[5] if len(command) > 5 else None,
        "--experiment-manifest",
        str(expected_manifest.resolve(strict=True)),
        "--twin",
        command[9] if len(command) > 9 else None,
        "--run-id",
        sample_id,
        "--out-parent",
        str(root.resolve(strict=True)),
        "--appearance-stage",
        str(layer_path.resolve(strict=True)),
        "--frames",
        "686:687:1",
        "--subframes",
        "64",
        "--warmup",
        "60",
        "--max-tries",
        "20",
        "--seed",
        "310241",
        "--require-liquid-pixels",
        "--renderer",
        "PathTracing",
        "--timeout",
        "1800",
        "--environment-profile",
        environment_profile,
    ]
    if not _exact_equal(command, expected_command):
        raise ValueError("attempt command does not exactly encode the frozen invocation")
    for path_index in (0, 3, 5, 9):
        candidate = command[path_index]
        if not isinstance(candidate, str) or not Path(candidate).is_absolute():
            raise ValueError("attempt runtime/twin command paths must be absolute")
    if not _exact_equal(plan["requested_controls"], config["controls"]):
        raise ValueError("attempt command controls drifted from pilot config")
    if plan["process_attempt_limit"] != 1 or type(plan["process_attempt_limit"]) is not int:
        raise ValueError("attempt manifest must permit one process attempt")
    if plan["replacement_process_attempts"] != 0 or type(plan["replacement_process_attempts"]) is not int:
        raise ValueError("attempt manifest must forbid replacement process attempts")

    appearance_layer = plan["appearance_layer"]
    if not isinstance(appearance_layer, Mapping) or set(appearance_layer) != _APPEARANCE_LAYER_FIELDS:
        raise ValueError("appearance layer preregistration has wrong exact fields")
    appearance = config["appearances"][row["appearance_id"]]  # type: ignore[index]
    expected_layer = {
        "schema": "direct_appearance_v1",
        "variant_id": row["layer_id"],
        "stage_sha256": layer["sha256"],
        "stage_bytes": layer["bytes"],
        "source_stage": "stations/fill_station/fill_station.usda",
        "source_stage_sha256": appearance_layer.get("source_stage_sha256"),
        "material_path": _MATERIAL_PATH,
        "diffuse_color_linear": appearance["linear_diffuse_rgb"],  # type: ignore[index]
        "opacity": appearance["opacity"],  # type: ignore[index]
    }
    source_hash = appearance_layer.get("source_stage_sha256")
    if not isinstance(source_hash, str) or _SHA256.fullmatch(source_hash) is None:
        raise ValueError("appearance layer source-stage hash is malformed")
    if not _exact_equal(dict(appearance_layer), expected_layer):
        raise ValueError("appearance layer preregistration does not match the frozen row")
    if payload["seed"] != config["controls"]["seed"] or type(payload["seed"]) is not int:  # type: ignore[index]
        raise ValueError("attempt manifest seed drifted")
    if not _exact_equal(payload["denominators"], {
        "pilot_rows": 8,
        "requested_frames": 1,
        "process_attempts": 1,
    }):
        raise ValueError("attempt manifest denominators drifted")
    expected_stops = [
        "no replacement process attempt",
        "bounded timeout",
        "provenance or safety failure",
    ]
    if not _exact_equal(payload["stop_conditions"], expected_stops):
        raise ValueError("attempt manifest stop conditions drifted")
    return deepcopy(dict(payload))


def _compose_graph(
    root: Path,
    config_record: Mapping[str, object],
    implementation_commit: str,
    repo_root: Path,
) -> dict[str, object]:
    if _COMMIT.fullmatch(implementation_commit) is None:
        raise ValueError("implementation commit must be 40 lowercase hexadecimal characters")
    config = config_record["payload"]
    assert isinstance(config, Mapping)
    rows = config["rows"]
    assert isinstance(rows, list)
    expected_ids = [str(row["sample_id"]) for row in rows]
    for directory_name, suffix in (
        ("appearance-layers", ".usda"),
        ("attempt-manifests", ".json"),
    ):
        directory = root / directory_name
        try:
            directory_metadata = os.lstat(directory)
            names = sorted(entry.name for entry in os.scandir(directory))
        except OSError as error:
            raise ValueError(f"{directory_name} is unavailable") from error
        if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(directory_metadata.st_mode):
            raise ValueError(f"{directory_name} must be a non-symlink directory")
        expected_names = sorted(f"{sample_id}{suffix}" for sample_id in expected_ids)
        if names != expected_names:
            raise ValueError(f"{directory_name} must contain exactly the eight preregistered files")
    entries: list[dict[str, object]] = []
    source_stage_hashes: set[str] = set()
    environment_profiles: set[str] = set()
    for row in rows:
        assert isinstance(row, Mapping)
        sample_id = str(row["sample_id"])
        layer = _snapshot_relative_regular(
            root, f"appearance-layers/{sample_id}.usda", prefix="appearance-layers"
        )
        if not 0 < int(layer["bytes"]) <= 1_048_576:
            raise ValueError("appearance layer has an invalid bounded size")
        manifest_record = _snapshot_relative_regular(
            root,
            f"attempt-manifests/{sample_id}.json",
            prefix="attempt-manifests",
        )
        manifest, manifest_snapshot = load_strict_json_snapshot(
            root / str(manifest_record["path"]), max_bytes=1024 * 1024
        )
        if manifest_snapshot != {
            "sha256": manifest_record["sha256"],
            "bytes": manifest_record["bytes"],
        }:
            raise ValueError("attempt manifest changed during graph construction")
        _validate_attempt_manifest(
            manifest,
            row=row,
            config=config,
            implementation_commit=implementation_commit,
            layer=layer,
            layer_path=root / str(layer["path"]),
        )
        assert isinstance(manifest, Mapping)
        source_stage_hashes.add(
            str(manifest["command_plan"]["appearance_layer"]["source_stage_sha256"])  # type: ignore[index]
        )
        environment_profiles.add(
            str(manifest["command_plan"]["environment_profile"])  # type: ignore[index]
        )
        entries.append(
            {
                "sample_id": sample_id,
                "layer": layer,
                "attempt_manifest": manifest_record,
            }
        )
    if len(source_stage_hashes) != 1:
        raise ValueError("all pilot layers must bind one unchanged source stage")
    if len(environment_profiles) != 1:
        raise ValueError("all pilot attempts must bind one environment profile")
    return {
        "schema_version": 1,
        "graph_id": "twin-appearance-pilot-graph-v1",
        "dataset_version": DATASET_VERSION,
        "config": {
            "config_id": CONFIG_ID,
            "sha256": config_record["sha256"],
            "bytes": config_record["bytes"],
        },
        "implementation_commit": implementation_commit,
        "environment_profile": environment_profiles.pop(),
        "implementation_files": [
            _snapshot_relative_regular(repo_root, relative)
            for relative in _IMPLEMENTATION_PATHS
        ],
        "ordered_sample_ids": [entry["sample_id"] for entry in entries],
        "controls": deepcopy(config["controls"]),
        "denominators": {
            "process_attempts": 8,
            "replacement_process_attempts": 0,
            "shared_capture_tries_per_process": 20,
        },
        "entries": entries,
    }


def build_experiment_graph(
    pilot_root: str | Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    implementation_commit: str,
    repo_root: str | Path = REPO,
) -> Path:
    """Exclusively freeze the eight pre-attempt layer/manifest snapshots."""

    root = _require_root(pilot_root, role="pilot root")
    with os.scandir(root) as entries:
        attempt_started = any(_ATTEMPT_NAME.fullmatch(entry.name) for entry in entries)
    if attempt_started:
        raise ValueError("experiment graph must be frozen before attempt 1 starts")
    try:
        actual_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("could not verify the graph implementation commit") from error
    if implementation_commit != actual_commit:
        raise ValueError("implementation commit does not match repository HEAD")
    config = load_pilot_config(config_path, repo_root=repo_root)
    graph = _compose_graph(root, config, implementation_commit, Path(repo_root).resolve(strict=True))
    return _write_exclusive(root / GRAPH_FILE, _json_bytes(graph))


def validate_experiment_graph(
    path: str | Path,
    *,
    pilot_root: str | Path,
    config_path: str | Path = CONFIG_PATH,
    repo_root: str | Path = REPO,
) -> dict[str, object]:
    """Verify a frozen graph and re-hash every bound preregistration input."""

    root = _require_root(pilot_root, role="pilot root")
    candidate = Path(path)
    if candidate.resolve(strict=True) != root / GRAPH_FILE:
        raise ValueError("experiment graph must be the canonical pilot-root file")
    payload, snapshot = load_strict_json_snapshot(candidate, max_bytes=4 * 1024 * 1024)
    if not isinstance(payload, Mapping):
        raise ValueError("experiment graph must be an object")
    commit = payload.get("implementation_commit")
    if not isinstance(commit, str):
        raise ValueError("experiment graph implementation commit is malformed")
    config = load_pilot_config(config_path, repo_root=repo_root)
    expected = _compose_graph(
        root, config, commit, Path(repo_root).resolve(strict=True)
    )
    if not _exact_equal(dict(payload), expected):
        raise ValueError("experiment graph does not exactly match its bound inputs")
    return {
        "path": str(candidate.resolve(strict=True)),
        "sha256": snapshot["sha256"],
        "bytes": snapshot["bytes"],
        "payload": deepcopy(expected),
    }


def launch_pilot_attempt(
    pilot_root: str | Path,
    sample_id: str,
    *,
    config_path: str | Path = CONFIG_PATH,
    repo_root: str | Path = REPO,
) -> int:
    """Validate the graph and spawn one preregistered outer argv verbatim."""

    root = _require_root(pilot_root, role="pilot root")
    repository = Path(repo_root).resolve(strict=True)
    graph_record = validate_experiment_graph(
        root / GRAPH_FILE,
        pilot_root=root,
        config_path=config_path,
        repo_root=repository,
    )
    graph = graph_record["payload"]
    assert isinstance(graph, Mapping)
    ordered_ids = graph["ordered_sample_ids"]
    entries = graph["entries"]
    if (
        not isinstance(sample_id, str)
        or not isinstance(ordered_ids, list)
        or sample_id not in ordered_ids
        or not isinstance(entries, list)
    ):
        raise ValueError("launch sample ID is not in the frozen graph")
    ordinal = ordered_ids.index(sample_id)
    expected_prior = sorted(str(value) for value in ordered_ids[:ordinal])
    with os.scandir(root) as root_entries:
        observed_attempts = sorted(
            entry.name for entry in root_entries if _ATTEMPT_NAME.fullmatch(entry.name)
        )
    if observed_attempts != expected_prior:
        raise ValueError("pilot launch order or no-replacement contract was violated")

    entry = entries[ordinal]
    if not isinstance(entry, Mapping) or entry.get("sample_id") != sample_id:
        raise ValueError("launch graph entry is inconsistent")
    manifest_record = entry.get("attempt_manifest")
    if not isinstance(manifest_record, Mapping):
        raise ValueError("launch manifest record is malformed")
    manifest, snapshot = load_strict_json_snapshot(
        root / str(manifest_record["path"]), max_bytes=1024 * 1024
    )
    if snapshot != {
        "sha256": manifest_record.get("sha256"),
        "bytes": manifest_record.get("bytes"),
    } or not isinstance(manifest, Mapping):
        raise ValueError("launch manifest drifted from the frozen graph")
    command = manifest["command_plan"]["command"]  # type: ignore[index]
    if not isinstance(command, list):
        raise ValueError("launch command is malformed")
    result = subprocess.run(list(command), cwd=repository, check=False)
    if type(result.returncode) is not int:
        raise ValueError("launcher returned a malformed process status")
    return result.returncode


_REVIEW_FIELDS = {
    "schema_version",
    "dataset_version",
    "config_sha256",
    "experiment_graph_sha256",
    "status",
    "reviewer_id",
    "independent_from_implementation",
    "reviewed_utc",
    "reviewed_sample_ids",
    "reviewed_artifacts",
    "checks",
}
_REVIEW_CHECKS = {
    "recognizable_station_glass_liquid",
    "invariant_mask_alignment",
    "coherent_ordered_appearances",
    "broken_geometry_absent",
}


def _relative_file_record(root: Path, path: str | Path, *, prefix: str | None = None) -> dict[str, object]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    lexical = Path(os.path.abspath(candidate))
    if not lexical.is_relative_to(root):
        raise ValueError("bound artifact must remain inside the pilot root")
    relative = lexical.relative_to(root).as_posix()
    return _snapshot_relative_regular(root, relative, prefix=prefix)


def _load_visual_review(
    path: str | Path,
    *,
    root: Path,
    config: Mapping[str, object],
    config_sha256: str,
    graph_sha256: str,
) -> dict[str, object]:
    record = _relative_file_record(root, path, prefix="reviews")
    payload, snapshot = load_strict_json_snapshot(root / str(record["path"]), max_bytes=1024 * 1024)
    if snapshot != {"sha256": record["sha256"], "bytes": record["bytes"]}:
        raise ValueError("visual-review record changed while reading")
    if not isinstance(payload, Mapping) or set(payload) != _REVIEW_FIELDS:
        raise ValueError("visual-review record has wrong exact fields")
    checks = payload["checks"]
    if not isinstance(checks, Mapping) or set(checks) != _REVIEW_CHECKS:
        raise ValueError("visual-review checks have wrong exact fields")
    expected_ids = [row["sample_id"] for row in config["rows"]]  # type: ignore[index]
    reviewed_artifacts = payload["reviewed_artifacts"]
    artifact_fields = {
        "sample_id",
        "rgb_sha256",
        "semantic_sha256",
        "instance_sha256",
        "mask_sha256",
        "summary_sha256",
        "attempt_sha256",
    }
    artifacts_valid = (
        isinstance(reviewed_artifacts, list)
        and len(reviewed_artifacts) == 8
        and all(
            isinstance(item, Mapping)
            and set(item) == artifact_fields
            and item["sample_id"] == sample_id
            and all(
                isinstance(item[field], str)
                and _SHA256.fullmatch(item[field]) is not None
                for field in artifact_fields - {"sample_id"}
            )
            for item, sample_id in zip(reviewed_artifacts, expected_ids, strict=True)
        )
    )
    if (
        payload["schema_version"] != 1
        or type(payload["schema_version"]) is not int
        or payload["dataset_version"] != DATASET_VERSION
        or payload["config_sha256"] != config_sha256
        or payload["experiment_graph_sha256"] != graph_sha256
        or payload["status"] != "accepted"
        or not isinstance(payload["reviewer_id"], str)
        or not payload["reviewer_id"].strip()
        or payload["independent_from_implementation"] is not True
        or not isinstance(payload["reviewed_utc"], str)
        or not payload["reviewed_utc"].strip()
        or payload["reviewed_sample_ids"] != expected_ids
        or not artifacts_valid
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("visual review is not an exact accepted independent review")
    return {**record, "payload": deepcopy(dict(payload))}


def _read_relative_bytes(
    root: Path,
    record: Mapping[str, object],
    *,
    prefix: str | None = None,
    max_bytes: int = 128 * 1024 * 1024,
) -> bytes:
    if set(record) != {"path", "sha256"}:
        raise ValueError("artifact locator must contain only path and sha256")
    path_text = record["path"]
    expected_hash = record["sha256"]
    if not isinstance(path_text, str) or not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
        raise ValueError("artifact locator is malformed")
    snapshot = _snapshot_relative_regular(root, path_text, prefix=prefix)
    if snapshot["sha256"] != expected_hash:
        raise ValueError(f"artifact hash mismatch: {path_text}")
    if int(snapshot["bytes"]) > max_bytes:
        raise ValueError(f"artifact exceeds the bounded size: {path_text}")
    candidate = root / path_text
    data = candidate.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash or len(data) != snapshot["bytes"]:
        raise ValueError(f"artifact changed while reading: {path_text}")
    after = _snapshot_relative_regular(root, path_text, prefix=prefix)
    if after != snapshot:
        raise ValueError(f"artifact changed while verifying: {path_text}")
    return data


def _decode_rgb(data: bytes, *, role: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB" or image.size != (640, 480):
                raise ValueError(f"{role} must be an RGB 8-bit 640x480 PNG")
            array = np.array(image, dtype=np.uint8, copy=True)
    except (OSError, SyntaxError) as error:
        raise ValueError(f"{role} is not a decodable PNG") from error
    if array.shape != (480, 640, 3) or array.dtype != np.dtype(np.uint8):
        raise ValueError(f"{role} decoded to the wrong array contract")
    return array


def _decode_mask(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.format != "PNG" or image.mode != "L" or image.size != (640, 480):
                raise ValueError("mask must be a single-channel 8-bit 640x480 PNG")
            array = np.array(image, dtype=np.uint8, copy=True)
    except (OSError, SyntaxError) as error:
        raise ValueError("mask is not a decodable PNG") from error
    values = set(int(value) for value in np.unique(array))
    if not values.issubset({0, 255}):
        raise ValueError("mask must be exactly binary 0/255")
    return array == 255


def _decode_integer_npy(data: bytes, *, role: str) -> np.ndarray:
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{role} is not a safe NumPy array") from error
    if (
        array.shape != (480, 640)
        or np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{role} must be an integer 480x640 array")
    return array


def _artifact_locator(root: Path, capture_root: Path, artifact: Mapping[str, object], kind: str) -> dict[str, str]:
    if set(artifact) != {"path", "sha256"}:
        raise ValueError("capture artifact record is malformed")
    capture_relative = _safe_relative(artifact["path"], prefix=kind)
    candidate = capture_root.joinpath(*capture_relative.parts)
    relative = candidate.resolve(strict=True).relative_to(root).as_posix()
    expected_prefix = f"{capture_root.parent.name}/capture/{kind}/"
    if not relative.startswith(expected_prefix):
        raise ValueError("capture artifact escaped its neutral attempt/kind role")
    return {"path": relative, "sha256": str(artifact["sha256"])}


def _expected_features(config: Mapping[str, object], row: Mapping[str, object]) -> dict[str, object]:
    return {
        "view_id": config["controls"]["view_id"],  # type: ignore[index]
        "current_programmed_ml": deepcopy(row["current_programmed_ml"]),
        "previous_programmed_ml": deepcopy(row["previous_programmed_ml"]),
        "programmed_dispense_order": deepcopy(config["programmed_dispense_order"]),
        "run_position": row["run_position"],
        "recipe_changed": row["recipe_changed"],
    }


def _inspect_attempt(
    root: Path,
    *,
    config: Mapping[str, object],
    config_record: Mapping[str, object],
    graph_record: Mapping[str, object],
    graph_entry: Mapping[str, object],
    row: Mapping[str, object],
    review_record: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, object], np.ndarray, np.ndarray, dict[str, str]]:
    sample_id = str(row["sample_id"])
    attempt_relative = f"{sample_id}/attempt.json"
    attempt_record = _snapshot_relative_regular(root, attempt_relative, prefix=sample_id)
    attempt, attempt_snapshot = load_strict_json_snapshot(root / attempt_relative, max_bytes=16 * 1024 * 1024)
    if attempt_snapshot != {"sha256": attempt_record["sha256"], "bytes": attempt_record["bytes"]}:
        raise ValueError("attempt record changed while reading")
    if not isinstance(attempt, Mapping):
        raise ValueError("attempt record must be an object")
    required = {
        "schema_version",
        "run_id",
        "status",
        "visual_check_status",
        "failure_reasons",
        "requested_settings",
        "timeout_seconds",
        "safe_command",
        "environment_policy",
        "capture",
        "provenance",
    }
    if not required.issubset(attempt):
        raise ValueError("attempt record is missing required launcher fields")
    if (
        attempt["schema_version"] != 1
        or type(attempt["schema_version"]) is not int
        or attempt["run_id"] != sample_id
        or attempt["status"] != "mechanical_gates_passed"
        or attempt["visual_check_status"] != "pending"
        or attempt["failure_reasons"] != []
        or attempt["timeout_seconds"] != config["controls"]["timeout_seconds"]  # type: ignore[index]
    ):
        raise ValueError("attempt did not pass the frozen mechanical contract")

    layer = graph_entry["layer"]
    manifest = graph_entry["attempt_manifest"]
    assert isinstance(layer, Mapping) and isinstance(manifest, Mapping)
    layer_path = root / str(layer["path"])
    capture_root = root / sample_id / "capture"
    verified = verify_capture_summary(capture_root, None)
    summary = verified["summary"]
    actual_settings = capture_requested_settings(summary)
    expected_settings = {
        "camera": "FillCam",
        "stage": str(layer_path.resolve(strict=True)),
        "frames": config["controls"]["frame_spec"],  # type: ignore[index]
        "subframes": config["controls"]["subframes"],  # type: ignore[index]
        "warmup": config["controls"]["warmup"],  # type: ignore[index]
        "max_tries": config["controls"]["max_tries"],  # type: ignore[index]
        "seed": config["controls"]["seed"],  # type: ignore[index]
        "require_liquid_pixels": True,
        "renderer": config["controls"]["renderer"],  # type: ignore[index]
    }
    if not _exact_equal(actual_settings, expected_settings) or not _exact_equal(attempt["requested_settings"], expected_settings):
        raise ValueError("attempt requested controls do not match the frozen graph/config")
    render = summary.get("render_settings")
    if not isinstance(render, Mapping) or render.get("resolution") != [640, 480]:
        raise ValueError("capture resolution is not the frozen 640x480 contract")
    inputs = summary.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("stage"), Mapping):
        raise ValueError("capture stage provenance is malformed")
    stage = inputs["stage"]
    if (
        stage.get("resolved_path") != str(layer_path.resolve(strict=True))
        or stage.get("sha256") != layer["sha256"]
    ):
        raise ValueError("capture root-stage provenance does not match the graph layer")
    stack = inputs.get("non_session_usd_layer_stack")
    if not isinstance(stack, list) or not stack or not isinstance(stack[0], Mapping):
        raise ValueError("capture layer stack provenance is missing")
    if (
        stack[0].get("ordinal") != 1
        or stack[0].get("resolved_path") != str(layer_path.resolve(strict=True))
        or stack[0].get("sha256") != layer["sha256"]
        or stack[0].get("bytes") != layer["bytes"]
    ):
        raise ValueError("capture layer stack is not rooted in the graph layer")

    capture_record = attempt["capture"]
    provenance = attempt["provenance"]
    if not isinstance(capture_record, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("attempt capture/provenance is malformed")
    if (
        capture_record.get("root") != str(capture_root.resolve(strict=True))
        or capture_record.get("verified") is not True
        or capture_record.get("appearance_provenance_verified") is not True
        or capture_record.get("inner_summary_sha256") != verified["summary_sha256"]
        or capture_record.get("artifact_count") != 4
    ):
        raise ValueError("attempt capture binding is inconsistent")
    git = provenance.get("git")
    experiment = provenance.get("experiment_manifest")
    appearance_layer = provenance.get("appearance_layer")
    if not isinstance(git, Mapping) or git.get("commit") != graph_record["payload"]["implementation_commit"] or git.get("dirty") is not False:  # type: ignore[index]
        raise ValueError("attempt Git provenance does not match the frozen graph")
    if not isinstance(experiment, Mapping) or any(
        (
            experiment.get("sha256") != manifest["sha256"],
            experiment.get("bytes") != manifest["bytes"],
            experiment.get("sha256_after") != manifest["sha256"],
            experiment.get("bytes_after") != manifest["bytes"],
            experiment.get("unchanged") is not True,
        )
    ):
        raise ValueError("attempt manifest provenance drifted")
    if not isinstance(appearance_layer, Mapping) or any(
        (
            appearance_layer.get("required") is not True,
            appearance_layer.get("path") != str(layer_path.resolve(strict=True)),
            appearance_layer.get("sha256") != layer["sha256"],
            appearance_layer.get("bytes") != layer["bytes"],
            appearance_layer.get("sha256_after") != layer["sha256"],
            appearance_layer.get("bytes_after") != layer["bytes"],
            appearance_layer.get("unchanged") is not True,
        )
    ):
        raise ValueError("attempt appearance-layer provenance drifted")
    inspection_before = appearance_layer.get("inspection_before")
    inspection_after = appearance_layer.get("inspection_after")
    inspection_fields = {
        "path",
        "schema",
        "variant_id",
        "stage_sha256",
        "stage_bytes",
        "source_stage",
        "source_stage_sha256",
        "material_path",
        "diffuse_color_linear",
        "opacity",
        "uncalibrated",
        "timing",
        "non_session_usd_layer_stack",
    }
    manifest_payload, _ = load_strict_json_snapshot(
        root / str(manifest["path"]), max_bytes=1024 * 1024
    )
    if not isinstance(manifest_payload, Mapping):
        raise ValueError("attempt manifest payload is malformed")
    preregistration = manifest_payload["command_plan"]["appearance_layer"]  # type: ignore[index]
    planned_command = manifest_payload["command_plan"]["command"]  # type: ignore[index]
    inspection_plan = (
        {
            key: inspection_before[key]
            for key in _APPEARANCE_LAYER_FIELDS
        }
        if isinstance(inspection_before, Mapping)
        and _APPEARANCE_LAYER_FIELDS.issubset(inspection_before)
        else None
    )
    if (
        not isinstance(inspection_before, Mapping)
        or set(inspection_before) != inspection_fields
        or not _exact_equal(dict(inspection_before), inspection_after)
        or not isinstance(preregistration, Mapping)
        or not _exact_equal(inspection_plan, dict(preregistration))
        or inspection_before.get("path") != str(layer_path.resolve(strict=True))
        or inspection_before.get("uncalibrated") is not True
        or not isinstance(inspection_before.get("timing"), Mapping)
        or not isinstance(inspection_before.get("non_session_usd_layer_stack"), list)
    ):
        raise ValueError("attempt appearance-layer inspection provenance drifted")
    normalized_stack: list[dict[str, object]] = []
    for stack_record in stack:
        if not isinstance(stack_record, Mapping) or set(stack_record) != {
            "ordinal",
            "identifier",
            "resolved_path",
            "sha256",
            "bytes",
        }:
            raise ValueError("capture non-session layer-stack record is malformed")
        if stack_record["identifier"] != stack_record["resolved_path"]:
            raise ValueError("capture layer identifier is not its resolved path")
        normalized_stack.append(
            {
                "ordinal": stack_record["ordinal"],
                "resolved_path": stack_record["resolved_path"],
                "sha256": stack_record["sha256"],
                "bytes": stack_record["bytes"],
            }
        )
    if not _exact_equal(
        normalized_stack, inspection_before["non_session_usd_layer_stack"]
    ):
        raise ValueError("capture layer stack does not exactly match inspected preflight")
    inspected_stack = inspection_before["non_session_usd_layer_stack"]
    if (
        len(inspected_stack) < 2
        or not isinstance(inspected_stack[1], Mapping)
        or inspected_stack[1].get("ordinal") != 2
        or inspected_stack[1].get("sha256") != preregistration["source_stage_sha256"]
        or not str(inspected_stack[1].get("resolved_path", "")).endswith(
            "/" + str(preregistration["source_stage"])
        )
    ):
        raise ValueError("appearance layer stack does not bind the canonical source second")
    timing = inspection_before["timing"]
    if (
        set(timing) != {
            "start_time_code",
            "end_time_code",
            "frames_per_second",
            "time_codes_per_second",
        }
        or any(
            type(value) is not float or not np.isfinite(value)
            for value in timing.values()
        )
        or render.get("stage_time_code_bounds")
        != [timing["start_time_code"], timing["end_time_code"]]
        or render.get("time_codes_per_second") != timing["time_codes_per_second"]
    ):
        raise ValueError("appearance timing does not match the capture stage contract")
    if not isinstance(provenance.get("twin"), Mapping) or provenance["twin"].get("unchanged") is not True:  # type: ignore[index]
        raise ValueError("attempt public-twin provenance is not unchanged")
    files = provenance.get("files")
    if not isinstance(files, Mapping) or not files or any(
        not isinstance(value, Mapping) or value.get("unchanged") is not True
        for value in files.values()
    ):
        raise ValueError("attempt source-file provenance is not unchanged")
    if not isinstance(planned_command, list):
        raise ValueError("attempt manifest command plan is malformed")
    runtime_lock = files.get("runtime_lock")
    isaac_record = files.get("isaac_python")
    renderer_record = files.get("renderer")
    environment_policy = attempt["environment_policy"]
    twin_record = provenance["twin"]
    if (
        not isinstance(runtime_lock, Mapping)
        or runtime_lock.get("path") != str(Path(planned_command[5]).resolve(strict=False))
        or not isinstance(isaac_record, Mapping)
        or not isinstance(isaac_record.get("before"), Mapping)
        or isaac_record["before"].get("lexical_path") != planned_command[3]  # type: ignore[index]
        or not isinstance(renderer_record, Mapping)
        or not isinstance(renderer_record.get("path"), str)
        or not isinstance(environment_policy, Mapping)
        or environment_policy.get("profile") != planned_command[-1]
        or twin_record.get("root") != str(Path(planned_command[9]).resolve(strict=False))
        or experiment.get("path") != str(Path(planned_command[7]).resolve(strict=False))
    ):
        raise ValueError("actual launcher runtime provenance drifted from command plan")
    expected_inner_command = [
        planned_command[3],
        renderer_record["path"],
        "--twin",
        twin_record["root"],
        "--stage",
        str(layer_path.resolve(strict=True)),
        "--camera",
        "FillCam",
        "--frames",
        "686:687:1",
        "--subframes",
        "64",
        "--warmup",
        "60",
        "--max-tries",
        "20",
        "--seed",
        "310241",
        "--renderer",
        "PathTracing",
        "--require-liquid-pixels",
        "--out",
        str(capture_root.resolve(strict=False)),
    ]
    if not _exact_equal(attempt["safe_command"], expected_inner_command):
        raise ValueError("actual renderer argv drifted from the frozen command plan")

    frames = summary.get("frames")
    if not isinstance(frames, list) or len(frames) != 1 or not isinstance(frames[0], Mapping):
        raise ValueError("pilot capture must contain exactly one frame")
    frame = frames[0]
    if frame.get("time_code") != 686 or frame.get("reason_codes") != []:
        raise ValueError("pilot frame time/reasons drifted")
    rgb_facts = frame.get("rgb")
    semantic_facts = frame.get("semantic")
    instance_facts = frame.get("instance")
    if (
        not isinstance(rgb_facts, Mapping)
        or rgb_facts.get("shape") != [480, 640, 3]
        or rgb_facts.get("dtype") != "uint8"
    ):
        raise ValueError("pilot frame RGB facts drifted")
    for field, facts in (("semantic", semantic_facts), ("instance", instance_facts)):
        if (
            not isinstance(facts, Mapping)
            or facts.get("shape") != [480, 640]
            or not isinstance(facts.get("dtype"), str)
            or re.fullmatch(r"u?int(?:8|16|32|64)", facts["dtype"]) is None
        ):
            raise ValueError(f"pilot frame {field} facts drifted")
    class_ids = frame.get("class_ids")
    counts = frame.get("class_pixel_counts")
    agreement = frame.get("instance_agreement")
    if (
        not isinstance(class_ids, Mapping)
        or set(class_ids) != {"liquid", "vial"}
        or not isinstance(counts, Mapping)
        or not isinstance(agreement, Mapping)
        or agreement.get("passed") is not True
    ):
        raise ValueError("pilot class/instance facts are malformed")
    liquid_ids = class_ids["liquid"]
    vial_ids = class_ids["vial"]
    selected_instance_ids = agreement.get("selected_liquid_instance_ids")
    if any(
        not isinstance(values, list)
        or not values
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
        for values in (liquid_ids, vial_ids, selected_instance_ids)
    ):
        raise ValueError("pilot class/instance IDs are invalid")

    artifacts = frame.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"rgb", "semantic", "instance", "liquid"}:
        raise ValueError("pilot artifacts are incomplete")
    locators = {
        kind: _artifact_locator(root, capture_root, artifacts[kind], kind)
        for kind in artifacts
    }
    if review_record is not None:
        review_payload = review_record["payload"]
        assert isinstance(review_payload, Mapping)
        reviewed_artifacts = review_payload["reviewed_artifacts"]
        assert isinstance(reviewed_artifacts, list)
        reviewed = next(
            item for item in reviewed_artifacts if item["sample_id"] == sample_id
        )
        expected_reviewed = {
            "sample_id": sample_id,
            "rgb_sha256": locators["rgb"]["sha256"],
            "semantic_sha256": locators["semantic"]["sha256"],
            "instance_sha256": locators["instance"]["sha256"],
            "mask_sha256": locators["liquid"]["sha256"],
            "summary_sha256": verified["summary_sha256"],
            "attempt_sha256": attempt_record["sha256"],
        }
        if not _exact_equal(reviewed, expected_reviewed):
            raise ValueError("accepted visual review does not bind the current artifacts")
    rgb = _decode_rgb(_read_relative_bytes(root, locators["rgb"], prefix=sample_id), role="RGB input")
    semantic = _decode_integer_npy(
        _read_relative_bytes(root, locators["semantic"], prefix=sample_id), role="semantic oracle"
    )
    instance = _decode_integer_npy(
        _read_relative_bytes(root, locators["instance"], prefix=sample_id), role="instance oracle"
    )
    mask = _decode_mask(_read_relative_bytes(root, locators["liquid"], prefix=sample_id))
    if str(semantic.dtype) != semantic_facts["dtype"] or str(instance.dtype) != instance_facts["dtype"]:  # type: ignore[index]
        raise ValueError("semantic/instance summary dtype does not match the artifact")
    if not np.any(rgb != rgb[0, 0]):
        raise ValueError("RGB input is spatially constant")
    total = 640 * 480
    liquid_count = int(np.count_nonzero(mask))
    semantic_mask = np.isin(semantic, liquid_ids)
    instance_mask = np.isin(instance, selected_instance_ids)
    vial_count = int(np.count_nonzero(np.isin(semantic, vial_ids)))
    if not 0 < liquid_count < total or not 0 < vial_count < total:
        raise ValueError("liquid and vial pixels must be nonempty and non-full")
    if not np.array_equal(mask, semantic_mask) or not np.array_equal(mask, instance_mask):
        raise ValueError("mask, semantic liquid, and instance liquid do not agree exactly")
    if (
        counts.get("liquid") != liquid_count
        or counts.get("vial") != vial_count
        or agreement.get("semantic_liquid_pixels") != liquid_count
        or agreement.get("instance_liquid_pixels") != liquid_count
    ):
        raise ValueError("capture summary pixel counts do not match artifact arrays")

    assert isinstance(manifest_payload, Mapping)
    source_hash = manifest_payload["command_plan"]["appearance_layer"]["source_stage_sha256"]  # type: ignore[index]
    recipe_sources = config["sources"]["recipes"]  # type: ignore[index]
    transition_source = config["sources"]["transition"]  # type: ignore[index]
    appearance = config["appearances"][row["appearance_id"]]  # type: ignore[index]
    rgb_record = locators["rgb"]
    model_record = {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "sample_id": sample_id,
        "artifact": rgb_record,
        "features": _expected_features(config, row),
    }
    label_record = {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "sample_id": sample_id,
        "partition": row["partition"],
        "identifiers": {
            "sequence_id": row["sequence_id"],
            "event_id": row["event_id"],
            "pair_id": row["pair_id"],
            "counterfactual_id": row["counterfactual_id"],
        },
        "recipes": {
            "current_sim_recipe_id": row["current_sim_recipe_id"],
            "previous_sim_recipe_id": row["previous_sim_recipe_id"],
            "current_recipe_row_sha256": recipe_sources[row["current_sim_recipe_id"]]["row_sha256"],  # type: ignore[index]
            "previous_recipe_row_sha256": recipe_sources[row["previous_sim_recipe_id"]]["row_sha256"],  # type: ignore[index]
            "transition_row_sha256": transition_source["row_sha256"],
        },
        "label": {
            "injected_appearance": row["injected_appearance"],
            "appearance_blend_alpha": row["appearance_blend_alpha"],
            "recipe_appearance_mapping_validated": False,
        },
        "appearance": {
            "endpoint_a_id": "A",
            "endpoint_a_linear_diffuse_rgb": deepcopy(config["endpoints"]["A"]["linear_diffuse_rgb"]),  # type: ignore[index]
            "endpoint_b_id": "B",
            "endpoint_b_linear_diffuse_rgb": deepcopy(config["endpoints"]["B"]["linear_diffuse_rgb"]),  # type: ignore[index]
            "authored_appearance_id": row["appearance_id"],
            "authored_linear_diffuse_rgb": deepcopy(appearance["linear_diffuse_rgb"]),  # type: ignore[index]
            "opacity": appearance["opacity"],  # type: ignore[index]
        },
        "oracle_artifacts": {
            "semantic": locators["semantic"],
            "instance": locators["instance"],
            "mask": locators["liquid"],
        },
        "taxonomy": {
            "liquid_class_ids": deepcopy(liquid_ids),
            "vial_class_ids": deepcopy(vial_ids),
            "liquid_instance_ids": deepcopy(selected_instance_ids),
        },
        "capture": {
            "renderer": config["controls"]["renderer"],  # type: ignore[index]
            "time_code": config["controls"]["time_code"],  # type: ignore[index]
            "seed": config["controls"]["seed"],  # type: ignore[index]
            "nuisance_id": row["nuisance_id"],
            "layer": deepcopy(dict(layer)),
            "config": {
                "config_id": CONFIG_ID,
                "sha256": config_record["sha256"],
                "bytes": config_record["bytes"],
            },
            "source_stage": {
                "path": "stations/fill_station/fill_station.usda",
                "sha256": source_hash,
            },
            "attempt": attempt_record,
            "summary": {
                "path": (capture_root / "summary.json").relative_to(root).as_posix(),
                "sha256": verified["summary_sha256"],
                "bytes": (capture_root / "summary.json").stat().st_size,
            },
            "experiment_manifest": deepcopy(dict(manifest)),
            "experiment_graph": {
                "path": GRAPH_FILE,
                "sha256": graph_record["sha256"],
                "bytes": graph_record["bytes"],
            },
        },
        "validity": {
            "mechanical_gates_passed": True,
            "rgb_spatially_nonconstant": True,
            "liquid_nonempty_nonfull": True,
            "vial_nonempty_nonfull": True,
            "mask_matches_semantic_liquid": True,
            "semantic_instance_liquid_agreement": True,
        },
        "visual_review": {
            "status": "accepted" if review_record is not None else "pending",
            "record_sha256": review_record["sha256"] if review_record is not None else None,
        },
    }
    invariant = {
        "mask": locators["liquid"]["sha256"],
        "semantic": locators["semantic"]["sha256"],
        "instance": locators["instance"]["sha256"],
    }
    return model_record, label_record, rgb, mask, invariant


def _inspect_all(
    root: Path,
    *,
    config_record: Mapping[str, object],
    graph_record: Mapping[str, object],
    review_record: Mapping[str, object] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[np.ndarray], list[np.ndarray], dict[str, str]]:
    config = config_record["payload"]
    graph = graph_record["payload"]
    assert isinstance(config, Mapping) and isinstance(graph, Mapping)
    rows = config["rows"]
    entries = graph["entries"]
    if not isinstance(rows, list) or not isinstance(entries, list) or len(rows) != 8 or len(entries) != 8:
        raise ValueError("pilot matrix/graph denominator is not eight")
    with os.scandir(root) as root_entries:
        attempt_names = sorted(
            entry.name
            for entry in root_entries
            if _ATTEMPT_NAME.fullmatch(entry.name)
            or (entry.is_dir(follow_symlinks=False) and (root / entry.name / "attempt.json").exists())
        )
    expected_attempt_names = sorted(str(row["sample_id"]) for row in rows)
    if attempt_names != expected_attempt_names:
        raise ValueError("pilot root must contain the exact eight attempts; replacement attempts are forbidden")
    for name in expected_attempt_names:
        metadata = os.lstat(root / name)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("each pilot attempt must be a non-symlink directory")
    models: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    rgbs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    invariants: list[dict[str, str]] = []
    for row, entry in zip(rows, entries, strict=True):
        if not isinstance(row, Mapping) or not isinstance(entry, Mapping) or entry.get("sample_id") != row.get("sample_id"):
            raise ValueError("pilot graph/config ordered join is inconsistent")
        model, label, rgb, mask, invariant = _inspect_attempt(
            root,
            config=config,
            config_record=config_record,
            graph_record=graph_record,
            graph_entry=entry,
            row=row,
            review_record=review_record,
        )
        models.append(model)
        labels.append(label)
        rgbs.append(rgb)
        masks.append(mask)
        invariants.append(invariant)
    for kind in ("mask", "semantic", "instance"):
        values = {record[kind] for record in invariants}
        if len(values) != 1:
            raise ValueError(f"all eight {kind} artifacts must have identical SHA-256")
    return models, labels, rgbs, masks, invariants[0]


def load_prepublication_diagnostic_inputs(
    pilot_root: str | Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    graph_path: str | Path | None = None,
    repo_root: str | Path = REPO,
) -> tuple[dict[str, object], ...]:
    """Verify graph/captures and return RGB plus oracle mask before final publication."""

    root = _require_root(pilot_root, role="pilot root")
    config_record = load_pilot_config(config_path, repo_root=repo_root)
    graph_record = validate_experiment_graph(
        root / GRAPH_FILE if graph_path is None else graph_path,
        pilot_root=root,
        config_path=config_path,
        repo_root=repo_root,
    )
    models, labels, rgbs, masks, _ = _inspect_all(
        root,
        config_record=config_record,
        graph_record=graph_record,
        review_record=None,
    )
    return tuple(
        {
            "sample_id": model["sample_id"],
            "rgb": np.array(rgb, copy=True),
            "mask": np.array(mask, dtype=np.bool_, copy=True),
            "rgb_sha256": model["artifact"]["sha256"],
            "mask_sha256": label["oracle_artifacts"]["mask"]["sha256"],
        }
        for model, label, rgb, mask in zip(models, labels, rgbs, masks, strict=True)
    )


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _artifact_receipt(path: str, data: bytes, rows: int) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "rows": rows,
    }


def _receipt_payload(
    *,
    config_record: Mapping[str, object],
    graph_record: Mapping[str, object],
    review_record: Mapping[str, object],
    model_data: bytes,
    label_data: bytes,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "purpose": "acceptance_only",
        "training_eligible": False,
        "config": {
            "config_id": CONFIG_ID,
            "sha256": config_record["sha256"],
            "bytes": config_record["bytes"],
        },
        "experiment_graph": {
            "path": GRAPH_FILE,
            "sha256": graph_record["sha256"],
            "bytes": graph_record["bytes"],
        },
        "visual_review": {
            "path": review_record["path"],
            "sha256": review_record["sha256"],
            "bytes": review_record["bytes"],
            "status": "accepted",
        },
        "artifacts": {
            "model_input": _artifact_receipt(MODEL_INPUT_FILE, model_data, 8),
            "label_provenance": _artifact_receipt(LABEL_PROVENANCE_FILE, label_data, 8),
        },
    }


def build_pilot_dataset(
    pilot_root: str | Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    graph_path: str | Path | None = None,
    visual_review_path: str | Path,
    repo_root: str | Path = REPO,
) -> dict[str, Path]:
    """Publish the two joined JSONLs and receipt only after accepted review."""

    root = _require_root(pilot_root, role="pilot root")
    outputs = [root / MODEL_INPUT_FILE, root / LABEL_PROVENANCE_FILE, root / RECEIPT_FILE]
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    config_record = load_pilot_config(config_path, repo_root=repo_root)
    graph_record = validate_experiment_graph(
        root / GRAPH_FILE if graph_path is None else graph_path,
        pilot_root=root,
        config_path=config_path,
        repo_root=repo_root,
    )
    config = config_record["payload"]
    assert isinstance(config, Mapping)
    review_record = _load_visual_review(
        visual_review_path,
        root=root,
        config=config,
        config_sha256=str(config_record["sha256"]),
        graph_sha256=str(graph_record["sha256"]),
    )
    models, labels, _, _, _ = _inspect_all(
        root,
        config_record=config_record,
        graph_record=graph_record,
        review_record=review_record,
    )
    model_data = _jsonl_bytes(models)
    label_data = _jsonl_bytes(labels)
    receipt = _receipt_payload(
        config_record=config_record,
        graph_record=graph_record,
        review_record=review_record,
        model_data=model_data,
        label_data=label_data,
    )
    _write_exclusive(outputs[0], model_data)
    _write_exclusive(outputs[1], label_data)
    _write_exclusive(outputs[2], _json_bytes(receipt))
    return {"model_input": outputs[0], "label_provenance": outputs[1], "receipt": outputs[2]}


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSONL row contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_jsonl_bytes(data: bytes, *, role: str) -> list[dict[str, object]]:
    try:
        text = data.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{role} is not UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line for line in lines) or not text.endswith("\n"):
        raise ValueError(f"{role} must be nonempty canonical line-delimited JSON")
    rows: list[dict[str, object]] = []
    for ordinal, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as error:
            raise ValueError(f"{role} row {ordinal} is invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{role} row {ordinal} must be an object")
        rows.append(payload)
    return rows


def _validated_dataset(
    pilot_root: str | Path,
    *,
    config_path: str | Path,
    repo_root: str | Path,
) -> dict[str, object]:
    root = _require_root(pilot_root, role="pilot root")
    receipt_path = root / RECEIPT_FILE
    receipt, receipt_snapshot = load_strict_json_snapshot(receipt_path, max_bytes=1024 * 1024)
    if not isinstance(receipt, Mapping):
        raise ValueError("dataset receipt must be an object")
    config_record = load_pilot_config(config_path, repo_root=repo_root)
    graph_record = validate_experiment_graph(
        root / GRAPH_FILE,
        pilot_root=root,
        config_path=config_path,
        repo_root=repo_root,
    )
    visual = receipt.get("visual_review")
    if not isinstance(visual, Mapping) or not isinstance(visual.get("path"), str):
        raise ValueError("dataset receipt visual-review binding is malformed")
    config = config_record["payload"]
    assert isinstance(config, Mapping)
    review_record = _load_visual_review(
        root / str(visual["path"]),
        root=root,
        config=config,
        config_sha256=str(config_record["sha256"]),
        graph_sha256=str(graph_record["sha256"]),
    )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"model_input", "label_provenance"}:
        raise ValueError("dataset receipt artifact bindings are malformed")
    loaded_data: dict[str, bytes] = {}
    for key, expected_path in (
        ("model_input", MODEL_INPUT_FILE),
        ("label_provenance", LABEL_PROVENANCE_FILE),
    ):
        record = artifacts[key]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes", "rows"}:
            raise ValueError("dataset receipt artifact record has wrong exact fields")
        if record["path"] != expected_path or record["rows"] != 8:
            raise ValueError("dataset receipt artifact path/row count drifted")
        data = _read_relative_bytes(
            root,
            {"path": record["path"], "sha256": record["sha256"]},
            max_bytes=16 * 1024 * 1024,
        )
        if len(data) != record["bytes"]:
            raise ValueError("dataset receipt byte count drifted")
        loaded_data[key] = data
    expected_receipt = _receipt_payload(
        config_record=config_record,
        graph_record=graph_record,
        review_record=review_record,
        model_data=loaded_data["model_input"],
        label_data=loaded_data["label_provenance"],
    )
    if not _exact_equal(dict(receipt), expected_receipt):
        raise ValueError("dataset receipt does not exactly bind the current artifacts")
    if load_strict_json_snapshot(receipt_path, max_bytes=1024 * 1024)[1] != receipt_snapshot:
        raise ValueError("dataset receipt changed during verification")
    models = _load_jsonl_bytes(loaded_data["model_input"], role="model input")
    labels = _load_jsonl_bytes(loaded_data["label_provenance"], role="label provenance")
    expected_models, expected_labels, rgbs, masks, invariant = _inspect_all(
        root,
        config_record=config_record,
        graph_record=graph_record,
        review_record=review_record,
    )
    if not _exact_equal(models, expected_models):
        raise ValueError("model input schema/content does not exactly match the frozen pilot")
    if not _exact_equal(labels, expected_labels):
        raise ValueError("label provenance schema/content does not exactly match the frozen pilot")
    if len({row["sample_id"] for row in models}) != 8 or len({row["sample_id"] for row in labels}) != 8:
        raise ValueError("dataset contains duplicate sample IDs")
    if [row["sample_id"] for row in models] != [row["sample_id"] for row in labels]:
        raise ValueError("model/label neutral-ID join is inconsistent")
    return {
        "root": root,
        "models": models,
        "labels": labels,
        "rgbs": rgbs,
        "masks": masks,
        "invariant": invariant,
    }


def check_pilot_dataset(
    pilot_root: str | Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    repo_root: str | Path = REPO,
) -> dict[str, object]:
    """Run all receipt, leakage, matrix, artifact, and separation gates."""

    validated = _validated_dataset(
        pilot_root, config_path=config_path, repo_root=repo_root
    )
    invariant = validated["invariant"]
    assert isinstance(invariant, Mapping)
    return {
        "dataset_version": DATASET_VERSION,
        "purpose": "acceptance_only",
        "training_eligible": False,
        "rows": 8,
        "mechanical_rows_passed": 8,
        "visual_review_status": "accepted",
        "mask_sha256": invariant["mask"],
        "semantic_sha256": invariant["semantic"],
        "instance_sha256": invariant["instance"],
    }


def load_inference_rgb(
    pilot_root: str | Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    repo_root: str | Path = REPO,
) -> tuple[np.ndarray, ...]:
    """Return verified RGB arrays only; IDs, paths, labels, and AOVs stay hidden."""

    validated = _validated_dataset(pilot_root, config_path=config_path, repo_root=repo_root)
    return tuple(np.array(array, copy=True) for array in validated["rgbs"])  # type: ignore[arg-type]


def load_training_samples(
    pilot_root: str | Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    repo_root: str | Path = REPO,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return `(RGB, binary target)` pairs with the mask in a target-only channel."""

    validated = _validated_dataset(pilot_root, config_path=config_path, repo_root=repo_root)
    return tuple(
        (np.array(rgb, copy=True), np.array(mask, dtype=np.bool_, copy=True))
        for rgb, mask in zip(validated["rgbs"], validated["masks"], strict=True)  # type: ignore[arg-type]
    )


def load_context_features(
    pilot_root: str | Path,
    *,
    config_path: str | Path = CONFIG_PATH,
    repo_root: str | Path = REPO,
) -> tuple[dict[str, object], ...]:
    """Return only the reviewed deployable programmed-context allowlist."""

    validated = _validated_dataset(pilot_root, config_path=config_path, repo_root=repo_root)
    models = validated["models"]
    assert isinstance(models, list)
    return tuple(deepcopy(row["features"]) for row in models)  # type: ignore[misc]
