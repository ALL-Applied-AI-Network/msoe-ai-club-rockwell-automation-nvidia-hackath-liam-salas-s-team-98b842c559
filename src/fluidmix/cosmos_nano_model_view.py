"""Stage and verify an offline, regular-file Cosmos3-Nano reasoner view."""

import hashlib
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path

from fluidmix.cosmos_contract_runner import sha256_path
from fluidmix.explanation_contract import ContractError, canonical_json


MODEL_REPOSITORY = "nvidia/Cosmos3-Nano"
MODEL_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
EXPECTED_FILE_COUNT = 48
_REPO_CACHE_NAME = "models--nvidia--Cosmos3-Nano"
_REASONER_PATTERNS = (
    "*.json",
    "*.jinja",
    "*.txt",
    "*.md",
    "transformer/*",
    "vision_encoder/*",
    "text_tokenizer/*",
)


def _is_reasoner_path(relative):
    value = relative.as_posix()
    return any(fnmatchcase(value, pattern) for pattern in _REASONER_PATTERNS)


def _manifest_payload(manifest):
    return {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}


def _manifest_bytes(manifest):
    return (
        json.dumps(manifest, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _validate_weight_index(root, paths, *, allow_snapshot_symlink=False):
    index_path = root / "model.safetensors.index.json"
    if (
        not index_path.is_file()
        or (index_path.is_symlink() and not allow_snapshot_symlink)
    ):
        raise ContractError("model view lacks a regular root weight index")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("root weight index is not valid JSON") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ContractError("root weight index has no weight_map")
    references = set(weight_map.values())
    if any(not isinstance(path, str) for path in references):
        raise ContractError("root weight index has a non-string path")
    missing = sorted(references - paths)
    if missing:
        raise ContractError(f"root weight index references absent files: {missing}")


def stage_model_view(snapshot_dir, view_dir, manifest_path):
    """Hard-link the exact documented reasoner selection from pinned cache blobs."""
    snapshot = Path(snapshot_dir)
    view = Path(view_dir)
    manifest_path = Path(manifest_path)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ContractError("cache snapshot must be a real directory")
    snapshot = snapshot.resolve()
    if snapshot.name != MODEL_REVISION or snapshot.parent.name != "snapshots":
        raise ContractError("cache snapshot is not the pinned revision")
    repo_cache = snapshot.parent.parent
    if repo_cache.name != _REPO_CACHE_NAME:
        raise ContractError("cache snapshot is not the pinned model repository")
    blobs = (repo_cache / "blobs").resolve()
    if not blobs.is_dir() or blobs.is_symlink():
        raise ContractError("pinned repository blobs directory is invalid")
    if view.exists() or view.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise ContractError("model view or manifest path is already occupied")
    if not view.parent.is_dir() or view.parent.is_symlink():
        raise ContractError("model view parent must be an existing real directory")
    if not manifest_path.parent.is_dir() or manifest_path.parent.is_symlink():
        raise ContractError("manifest parent must be an existing real directory")

    selected = []
    for candidate in snapshot.rglob("*"):
        relative = candidate.relative_to(snapshot)
        if not _is_reasoner_path(relative):
            continue
        if not candidate.is_symlink():
            raise ContractError(f"selected cache entry is not a snapshot symlink: {relative}")
        target = candidate.resolve(strict=True)
        if blobs not in target.parents or not target.is_file() or target.is_symlink():
            raise ContractError(f"selected cache target escapes pinned blobs: {relative}")
        if target.stat().st_dev != view.parent.stat().st_dev:
            raise ContractError("cache blob and model view are on different filesystems")
        selected.append((relative, target))
    selected.sort(key=lambda item: item[0].as_posix())
    if len(selected) != EXPECTED_FILE_COUNT:
        raise ContractError(
            f"reasoner selection has {len(selected)} files, expected {EXPECTED_FILE_COUNT}"
        )
    selected_paths = {relative.as_posix() for relative, _ in selected}
    _validate_weight_index(
        snapshot, selected_paths, allow_snapshot_symlink=True
    )

    rows = []
    for relative, target in selected:
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_path(target),
            }
        )
    core = {
        "schema_version": "cosmos_nano_model_view_manifest.v1",
        "repo_id": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "selection": list(_REASONER_PATTERNS),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    manifest = {
        **core,
        "manifest_payload_sha256": hashlib.sha256(
            canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }

    view.mkdir(mode=0o700, exist_ok=False)
    for relative, target in selected:
        destination = view / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.link(target, destination, follow_symlinks=False)
    with manifest_path.open("xb") as stream:
        stream.write(_manifest_bytes(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def _extract_chat_template_text(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("chat_template.json is not valid JSON") from exc
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("chat_template"), str):
        return value["chat_template"]
    raise ContractError("chat_template.json has no single template text")


def verify_model_view(view_dir, manifest_path, config):
    """Rehash every staged file and reject drift, links, extras, or native-config changes."""
    view = Path(view_dir)
    manifest_path = Path(manifest_path)
    if view.is_symlink() or not view.is_dir():
        raise ContractError("model view must be a real directory")
    view = view.resolve()
    if sha256_path(manifest_path) != config["model_view_manifest_sha256"]:
        raise ContractError("model-view manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("model-view manifest is not valid JSON") from exc
    core = _manifest_payload(manifest)
    expected_payload_hash = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    if manifest.get("manifest_payload_sha256") != expected_payload_hash:
        raise ContractError("model-view manifest self-hash mismatch")
    fixed = {
        "schema_version": "cosmos_nano_model_view_manifest.v1",
        "repo_id": config["model_repository"],
        "revision": config["model_revision"],
        "file_count": config["model_view_file_count"],
        "total_bytes": config["model_view_total_bytes"],
    }
    for key, expected in fixed.items():
        if manifest.get(key) != expected:
            raise ContractError(f"model-view manifest mismatch: {key}")
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != config["model_view_file_count"]:
        raise ContractError("model-view manifest file rows are incomplete")
    manifest_paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(manifest_paths) != len(rows) or manifest_paths != sorted(set(manifest_paths)):
        raise ContractError("model-view manifest paths are invalid or duplicated")
    for candidate in view.rglob("*"):
        if candidate.is_symlink():
            raise ContractError("model view contains a symlink")
    actual_paths = sorted(
        path.relative_to(view).as_posix() for path in view.rglob("*") if path.is_file()
    )
    if actual_paths != manifest_paths:
        raise ContractError("model view has missing or extra files")
    for row in rows:
        if set(row) != {"path", "bytes", "sha256"}:
            raise ContractError("invalid model-view manifest row")
        path = view / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_path(path) != row["sha256"]:
            raise ContractError(f"model-view file drift: {row['path']}")
    _validate_weight_index(view, set(actual_paths))

    config_json = json.loads((view / "config.json").read_text(encoding="utf-8"))
    if config_json.get("model_type") != config["model_type"]:
        raise ContractError("Nano model_type changed")
    generation = json.loads(
        (view / "generation_config.json").read_text(encoding="utf-8")
    )
    if generation.get("eos_token_id") != config["native_eos_token_ids"]:
        raise ContractError("Nano native EOS token IDs changed")
    if generation.get("pad_token_id") != config["native_pad_token_id"]:
        raise ContractError("Nano native pad token ID changed")
    template_path = view / "chat_template.json"
    if "chat_template_file_sha256" in config and sha256_path(template_path) != config[
        "chat_template_file_sha256"
    ]:
        raise ContractError("Nano chat-template file changed")
    template_text = _extract_chat_template_text(template_path)
    if "chat_template_text_sha256" in config and hashlib.sha256(
        template_text.encode("utf-8")
    ).hexdigest() != config["chat_template_text_sha256"]:
        raise ContractError("Nano chat-template text changed")
    return {
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_sha256": sha256_path(manifest_path),
        "chat_template_text": template_text,
    }
