"""Frozen attempt planning and recording helpers for the local Cosmos contract run."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import time

from fluidmix.explanation_contract import (
    ContractError,
    build_prompt,
    canonical_json,
    reference_explanation,
    validate_eval_config,
    validate_fixture_set,
)


def build_attempt_plan(fixture_set, config, prompt_template):
    """Build the exact record-by-mode matrix without reading the scoring oracle."""
    validate_fixture_set(fixture_set)
    validate_eval_config(config)
    attempts = []
    for record in fixture_set["fixtures"]:
        for mode in config["modes"]:
            camera_included = mode != "image_recipe"
            image_attached = mode in {"image_evidence_recipe", "image_recipe"}
            prompt = None
            prompt_sha256 = None
            template_response = None
            if mode == "template_reference":
                template_response = canonical_json(reference_explanation(record))
            else:
                prompt = build_prompt(
                    prompt_template,
                    record,
                    camera_evidence_included=camera_included,
                )
                prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            attempts.append({
                "attempt_index": len(attempts),
                "record_id": record["record_id"],
                "mode": mode,
                "record": record,
                "camera_evidence_included": camera_included,
                "image_attached": image_attached,
                "prompt": prompt,
                "prompt_sha256": prompt_sha256,
                "template_response": template_response,
            })
    if len(attempts) != config["planned_attempts"]:
        raise RuntimeError("attempt plan does not match frozen denominator")
    return attempts


def execute_attempt_plan(attempt_plan, infer):
    """Execute every planned attempt once; errors become retained attempt records."""
    results = []
    for spec in attempt_plan:
        started = time.monotonic()
        base = {
            "attempt_index": spec["attempt_index"],
            "record_id": spec["record_id"],
            "mode": spec["mode"],
            "prompt_sha256": spec["prompt_sha256"],
            "image_attached": spec["image_attached"],
        }
        try:
            if spec["mode"] == "template_reference":
                response = spec["template_response"]
                generated_tokens = None
            else:
                response, generated_tokens = infer(spec["prompt"], spec["image_attached"])
                if not isinstance(response, str):
                    raise TypeError("inference response must be text")
                if (
                    isinstance(generated_tokens, bool)
                    or not isinstance(generated_tokens, int)
                    or generated_tokens < 0
                ):
                    raise TypeError("generated token count must be a nonnegative integer")
            result = {
                **base,
                "attempt_status": "completed",
                "response_raw": response,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "generated_token_count": generated_tokens,
                "error_type": None,
                "error": None,
            }
        except Exception as exc:  # every frozen attempt remains in the denominator
            result = {
                **base,
                "attempt_status": "error",
                "response_raw": None,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "generated_token_count": None,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
        results.append(result)
    return results


def sha256_path(path):
    """Hash one file in bounded chunks."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_snapshot(model_dir, manifest_path, config):
    """Verify the pinned model manifest and every selected local file."""
    model_dir = Path(model_dir).resolve()
    manifest_path = Path(manifest_path)
    if sha256_path(manifest_path) != config["model_manifest_sha256"]:
        raise ContractError("model manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "verified":
        raise ContractError("model manifest is not verified")
    if manifest.get("repo_id") != config["model_repository"]:
        raise ContractError("model repository mismatch")
    if manifest.get("revision") != config["model_revision"]:
        raise ContractError("model revision mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("model manifest has no selected files")
    seen = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ContractError("invalid model manifest row")
        rel = row["path"]
        if (
            not isinstance(rel, str)
            or not rel
            or Path(rel).is_absolute()
            or ".." in Path(rel).parts
            or rel in seen
        ):
            raise ContractError("unsafe or duplicate model manifest path")
        seen.add(rel)
        candidate = model_dir / rel
        relative_chain = [candidate]
        parent = candidate.parent
        while parent != model_dir:
            relative_chain.append(parent)
            parent = parent.parent
        if any(item.is_symlink() for item in relative_chain):
            raise ContractError(f"model manifest path uses a symlink: {rel}")
        path = candidate.resolve()
        if model_dir not in path.parents:
            raise ContractError("model manifest path escapes model directory")
        expected_bytes = row["bytes"]
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ContractError("invalid model file byte count")
        if not path.is_file() or path.is_symlink() or path.stat().st_size != expected_bytes:
            raise ContractError(f"model file size/type mismatch: {rel}")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ContractError("invalid model file SHA-256")
        if sha256_path(path) != row["sha256"]:
            raise ContractError(f"model file SHA-256 mismatch: {rel}")


def verify_git_freeze(repo_root, tag):
    """Return HEAD after proving that the named tag points to a clean checkout."""
    repo_root = Path(repo_root)

    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ContractError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    head = git("rev-parse", "HEAD")
    tag_commit = git("rev-parse", f"refs/tags/{tag}^{{commit}}")
    if head != tag_commit:
        raise ContractError("freeze tag does not point to HEAD")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ContractError("repository is not clean at frozen tag")
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ContractError("HEAD is not a full commit hash")
    return head


def verify_runtime_environment(
    lock_path,
    config,
    *,
    installed_versions=None,
    actual_python=None,
    actual_machine=None,
):
    """Match every lock pin and the frozen interpreter/platform to the running environment."""
    lock_path = Path(lock_path)
    if sha256_path(lock_path) != config["runtime_lock_sha256"]:
        raise ContractError("runtime lock SHA-256 mismatch")

    def canonical_name(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    locked = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", line)
        if not match:
            continue
        name = canonical_name(match.group(1))
        if name in locked:
            raise ContractError(f"duplicate runtime lock package: {name}")
        locked[name] = match.group(2)
    if not locked:
        raise ContractError("runtime lock contains no exact package pins")

    if installed_versions is None:
        installed_versions = {
            canonical_name(distribution.metadata["Name"]): distribution.version
            for distribution in metadata.distributions()
            if distribution.metadata.get("Name")
        }
    else:
        installed_versions = {
            canonical_name(name): version for name, version in installed_versions.items()
        }
    observed = {}
    for name, expected in locked.items():
        actual = installed_versions.get(name)
        if actual != expected:
            raise ContractError(
                f"runtime package mismatch for {name}: {actual!r} != {expected!r}"
            )
        observed[name] = actual

    actual_python = actual_python or platform.python_version()
    actual_machine = actual_machine or platform.machine()
    if actual_python != config["python_version"]:
        raise ContractError("runtime Python version mismatch")
    if actual_machine != config["platform_machine"]:
        raise ContractError("runtime platform-machine mismatch")
    expected_versions = config.get("runtime_versions")
    if not isinstance(expected_versions, dict) or not expected_versions:
        raise ContractError("runtime_versions pin set is missing")
    for raw_name, expected in expected_versions.items():
        name = canonical_name(raw_name)
        if locked.get(name) != expected or observed.get(name) != expected:
            raise ContractError(f"runtime summary pin mismatch for {name}")
    return dict(sorted(observed.items()))


def create_frozen_output_parent(repo_root, output_relative):
    """Create the one ignored run directory without following repository-path symlinks."""
    repo_root = Path(repo_root)
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise ContractError("repository root must be a real directory")
    repo_root = repo_root.resolve()
    relative = Path(output_relative)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 3
        or relative.parts[0] != "outputs"
    ):
        raise ContractError("frozen output must be outputs/<run>/<file>")
    output = repo_root / relative
    current = repo_root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"output path component is a symlink: {current}")
        if current.exists() and not current.is_dir():
            raise ContractError(f"output path component is not a directory: {current}")
    if output.exists() or output.is_symlink() or output.parent.exists():
        raise ContractError("frozen output path is already occupied")
    try:
        output.parent.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:  # covers a local race after the checks
        raise ContractError("frozen output directory became occupied") from exc
    return output
