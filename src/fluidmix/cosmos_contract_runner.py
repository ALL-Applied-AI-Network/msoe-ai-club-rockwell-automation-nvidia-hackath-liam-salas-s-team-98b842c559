"""Shared fail-closed preflight helpers for a frozen local Cosmos run."""

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess

from fluidmix.cosmos_nano_contract import validate_nano_eval_config
from fluidmix.explanation_contract import (
    ContractError,
    build_prompt,
    canonical_json,
    reference_explanation,
    validate_fixture_set,
)


def sha256_path(path):
    """Hash one non-symlink regular file in bounded chunks."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_attempt_plan(fixture_set, config, prompt_template):
    """Build the exact fixture-major public-input matrix."""
    validate_fixture_set(fixture_set)
    validate_nano_eval_config(config)
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
            attempts.append(
                {
                    "attempt_index": len(attempts),
                    "record_id": record["record_id"],
                    "mode": mode,
                    "camera_evidence_included": camera_included,
                    "image_attached": image_attached,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "template_response": template_response,
                }
            )
    if len(attempts) != config["planned_attempts"]:
        raise ContractError("attempt plan does not match the frozen denominator")
    return attempts


def verify_git_freeze(repo_root, tag):
    """Return HEAD after proving that the named tag identifies a clean checkout."""
    repo_root = Path(repo_root)

    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ContractError(
                f"git {' '.join(args)} failed: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    head = git("rev-parse", "HEAD")
    tag_commit = git("rev-parse", f"refs/tags/{tag}^{{commit}}")
    if head != tag_commit:
        raise ContractError("freeze tag does not point to HEAD")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ContractError("repository is not clean at frozen tag")
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ContractError("HEAD is not a full commit hash")
    return head


def _canonical_distribution_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def verify_runtime_environment(
    lock_path,
    config,
    *,
    installed_versions=None,
    actual_python=None,
    actual_machine=None,
):
    """Require every lock pin plus exactly the explicitly allowed extra packages."""
    lock_path = Path(lock_path)
    if sha256_path(lock_path) != config["runtime_lock_sha256"]:
        raise ContractError("runtime lock SHA-256 mismatch")
    locked = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", line)
        if not match:
            continue
        name = _canonical_distribution_name(match.group(1))
        if name in locked:
            raise ContractError(f"duplicate runtime lock package: {name}")
        locked[name] = match.group(2)
    if len(locked) != config["runtime_lock_pin_count"]:
        raise ContractError("runtime lock pin count mismatch")

    if installed_versions is None:
        installed_versions = {
            _canonical_distribution_name(distribution.metadata["Name"]): distribution.version
            for distribution in metadata.distributions()
            if distribution.metadata.get("Name")
        }
    else:
        installed_versions = {
            _canonical_distribution_name(name): version
            for name, version in installed_versions.items()
        }
    allowed_extras = {
        _canonical_distribution_name(name): version
        for name, version in config["allowed_extra_distributions"].items()
    }
    expected_names = set(locked) | set(allowed_extras)
    if set(installed_versions) != expected_names:
        missing = sorted(expected_names - set(installed_versions))
        extra = sorted(set(installed_versions) - expected_names)
        raise ContractError(
            f"runtime distribution set mismatch: missing={missing}, extra={extra}"
        )
    for name, expected in {**locked, **allowed_extras}.items():
        actual = installed_versions[name]
        if actual != expected:
            raise ContractError(
                f"runtime package mismatch for {name}: {actual!r} != {expected!r}"
            )

    actual_python = actual_python or platform.python_version()
    actual_machine = actual_machine or platform.machine()
    if actual_python != config["python_version"]:
        raise ContractError("runtime Python version mismatch")
    if actual_machine != config["platform_machine"]:
        raise ContractError("runtime platform-machine mismatch")
    for raw_name, expected in config["runtime_versions"].items():
        name = _canonical_distribution_name(raw_name)
        if locked.get(name) != expected or installed_versions.get(name) != expected:
            raise ContractError(f"runtime summary pin mismatch for {name}")
    return {
        "locked": dict(sorted(locked.items())),
        "extras": dict(sorted(allowed_extras.items())),
    }


def verify_source_hashes(repo_root, expected, required_paths):
    """Verify the exact reviewed source path set and return its observed hashes."""
    repo_root = Path(repo_root).resolve()
    if not isinstance(expected, dict) or set(expected) != set(required_paths):
        raise ContractError("source hash path set is not frozen")
    observed = {}
    for relative in required_paths:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ContractError("unsafe source hash path")
        digest = sha256_path(repo_root / path)
        if digest != expected[relative]:
            raise ContractError(f"frozen source SHA-256 mismatch: {relative}")
        observed[relative] = digest
    return observed


def _resolve_output(repo_root, output_relative):
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
    current = repo_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"output path component is a symlink: {current}")
    return repo_root, repo_root / relative


def require_output_absent(repo_root, output_relative):
    """Fail preflight if the final artifact path is already occupied."""
    _, output = _resolve_output(repo_root, output_relative)
    if output.exists() or output.is_symlink():
        raise ContractError("frozen output path is already occupied")
    if output.parent.exists() and not output.parent.is_dir():
        raise ContractError("frozen output parent is not a directory")
    return output


def publish_immutable_json(repo_root, output_relative, value):
    """Atomically publish one exclusive read-only JSON artifact after it is complete."""
    repo_root, output = _resolve_output(repo_root, output_relative)
    if output.exists() or output.is_symlink():
        raise ContractError("frozen output path is already occupied")
    payload = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    if output.parent.exists():
        if not output.parent.is_dir() or output.parent.is_symlink():
            raise ContractError("frozen output parent is not a real directory")
    else:
        try:
            output.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ContractError("frozen output directory became occupied") from exc
    temporary = output.parent / f".{output.name}.incomplete"
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
        temporary.unlink()
        output.chmod(0o444)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return output
