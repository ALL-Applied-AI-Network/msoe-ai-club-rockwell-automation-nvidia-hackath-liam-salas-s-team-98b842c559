"""Publish CPU-only twin artifacts for reference-bank plan assembly."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import secrets

from twin.cohorts.one_run_manifest_v1 import build_one_run_source_artifacts
from twin.physx.run_variation import (
    bind_tank_colour_resolution_to_artifacts,
    build_tank_colour_resolution,
)


_RECIPES = {
    "amber_teal": (
        ("S0439", {"t100": 26.0, "t200": 28.0, "t400": 0.0}),
        ("S0440", {"t100": 11.0, "t200": 31.0, "t400": 20.0}),
        ("S0441", {"t100": 11.0, "t200": 11.0, "t400": 43.0}),
    ),
    "crimson_indigo": (
        ("S0442", {"t100": 56.0, "t200": 24.0, "t400": 0.0}),
        ("S0443", {"t100": 10.0, "t200": 50.0, "t400": 16.0}),
        ("S0444", {"t100": 11.0, "t200": 10.0, "t400": 39.0}),
    ),
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def build_reference_bank_twin_artifact_manifest(
    repo_root: str | Path,
) -> dict[str, object]:
    """Build the published twin half of the INN-67 plan without rendering."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    recipe_manifest = root / "data" / "sim_recipes_v3.csv"
    split_spec = root / "data" / "sim_split_spec_v3.json"
    rows: list[dict[str, object]] = []
    for batch_ordinal in (1, 2, 3):
        for palette in ("amber_teal", "crimson_indigo"):
            palette_code = "AT" if palette == "amber_teal" else "CI"
            run_id = f"INN67-DEV-RB-B{batch_ordinal:02d}-{palette_code}-RUN"
            subrun_id = f"INN67-DEV-RB-B{batch_ordinal:02d}-{palette_code}-SUBRUN"
            recipes = _RECIPES[palette]
            seed_base = 6_700_000 + batch_ordinal * 100 + (
                0 if palette_code == "AT" else 10
            )
            source = {
                "schema_version": 1,
                "artifact_type": "fluidmix-one-run-source-spec-v1",
                "run_id": run_id,
                "split_role": "development",
                "split_version": "inn67-development-reference-bank-v1",
                "source_sha256": {
                    "recipe_manifest": _file_sha256(recipe_manifest),
                    "split_spec": _file_sha256(split_spec),
                },
                "recipes": [
                    {"recipe_id": recipe_id, "programmed_ml": deepcopy(volumes)}
                    for recipe_id, volumes in recipes
                ],
                "initial_condition": "clean_start",
                "initial_predecessor_recipe_id": None,
                "vial_recipe_ids": [recipe_id for recipe_id, _ in recipes],
                "residual_volume_ml_by_vial": [0.0, 0.0, 0.0],
                "render_seed_by_vial": [
                    seed_base + offset for offset in (1, 2, 3)
                ],
                "nuisance_group_id_by_vial": [
                    f"INN67-DEV-RB-B{batch_ordinal:02d}-{palette_code}-V{ordinal:02d}"
                    for ordinal in (1, 2, 3)
                ],
            }
            artifacts = bind_tank_colour_resolution_to_artifacts(
                build_one_run_source_artifacts(source),
                build_tank_colour_resolution(palette),
            )
            for filename, payload in artifacts.items():
                rows.append(
                    {
                        "relative_path": f"subruns/{subrun_id}/{filename}",
                        "payload_sha256": _canonical_sha256(payload),
                        "payload": payload,
                    }
                )
    return {
        "schema_version": 1,
        "artifact_type": "inn67_reference_bank_twin_artifact_manifest",
        "status": "published",
        "artifacts": rows,
    }


def publish_reference_bank_twin_artifact_manifest(
    repo_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Atomically publish one new manifest and return its external trust hash."""

    output = Path(output_path).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"published twin manifest already exists: {output}")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("published twin manifest parent must be a directory")
    value = build_reference_bank_twin_artifact_manifest(repo_root)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = parent / f".{output.name}.tmp-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, output, follow_symlinks=False)
        published = True
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not published:
        raise OSError("published twin manifest was not published")
    return {
        "status": "published",
        "path": str(output),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "artifact_count": len(value["artifacts"]),
    }


__all__ = [
    "build_reference_bank_twin_artifact_manifest",
    "publish_reference_bank_twin_artifact_manifest",
]
