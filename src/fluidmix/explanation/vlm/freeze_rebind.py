"""Fail-closed proof and artifact builder for a move-only VLM freeze rebind."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


class FreezeRebindError(RuntimeError):
    """The proposed source move or freeze rebind is not exactly admissible."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _contains_file_name(node: ast.AST) -> bool:
    return any(isinstance(item, ast.Name) and item.id == "__file__" for item in ast.walk(node))


class _MoveOnlyNormalizer(ast.NodeTransformer):
    """Erase only import module targets and __file__-derived read paths."""

    def visit_Import(self, node: ast.Import) -> ast.AST:
        node = deepcopy(node)
        node.names = [ast.alias(name="__IMPORT_TARGET__", asname=item.asname) for item in node.names]
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        node = deepcopy(node)
        node.module = "__IMPORT_TARGET__"
        node.level = 0
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        if (
            any(isinstance(target, ast.Name) and target.id == "tracker_source" for target in node.targets)
            and _contains_file_name(node.value)
        ):
            node.value = ast.Constant(value="__FILE_DERIVED_MODULE_PATH__")
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_bytes"
            and _contains_file_name(node.func.value)
        ):
            node.func.value = ast.Constant(value="__FILE_DERIVED_MODULE_PATH__")
        return node


def _parse(raw: bytes, label: str) -> ast.Module:
    try:
        return ast.parse(raw.decode("utf-8"), filename=label)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise FreezeRebindError(f"{label} is not valid UTF-8 Python source") from error


def _dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False, indent=2)


def _normalized_dump(tree: ast.Module) -> str:
    normalized = _MoveOnlyNormalizer().visit(deepcopy(tree))
    ast.fix_missing_locations(normalized)
    return _dump(normalized)


def _import_targets(tree: ast.Module) -> list[tuple[object, ...]]:
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.append((node.lineno, "import", tuple(item.name for item in node.names)))
        elif isinstance(node, ast.ImportFrom):
            targets.append((node.lineno, "from", node.level, node.module))
    return targets


def _file_path_expressions(tree: ast.Module) -> list[tuple[int, str]]:
    result = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "tracker_source" for target in node.targets)
            and _contains_file_name(node.value)
        ):
            result.append((node.lineno, _dump(node.value)))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_bytes"
            and _contains_file_name(node.func.value)
        ):
            result.append((node.lineno, _dump(node.func.value)))
    return result


def _differing_count(old: Sequence[tuple[object, ...]], new: Sequence[tuple[object, ...]], label: str) -> int:
    if len(old) != len(new):
        raise FreezeRebindError(f"{label} structure differs")
    count = 0
    for left, right in zip(old, new, strict=True):
        if left != right:
            count += 1
    return count


def _top_level_nodes(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node
    return result


_COMPONENTS = {
    "prompt_strings": ("SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE"),
    "response_schema": ("RESPONSE_SCHEMA",),
    "guard_logic": (
        "_FORBIDDEN", "_PATH", "_PROSE_CITATION", "_DELTA_CLAIM",
        "_CONFIDENCE_CLAIM", "_THRESHOLD_COMPARISON", "_HIDDEN_SIMULATOR",
        "_CAUSE_CLAIM", "_EVIDENCE_UNAVAILABLE_EXPLANATION", "_ACTION",
        "_OPERATOR_ACTION", "_KEEP_QUALITY_CLAIM", "_DISCARD_QUALITY_CLAIM",
        "_NO_PREVIOUS_RECIPE_DIRECTION_SUPPORT", "parse_model_response",
        "_deterministic_explanation", "_prose_guard_failures", "guard_model_response",
    ),
    "thresholds_and_aggregation": (
        "AGGREGATION", "DETERMINISTIC_CONFIG_SHA256", "DETERMINISTIC_POLICY",
        "build_deterministic_decision", "aggregate_decision",
    ),
    "generation_config": ("GENERATION_CONFIG",),
    "model_pins": ("_MODEL_RUNTIME_POLICIES", "MODELS", "MODEL_REASONING_MODES"),
}


def _component_hashes(tree: ast.Module) -> dict[str, str]:
    nodes = _top_level_nodes(tree)
    result = {}
    for group, names in _COMPONENTS.items():
        missing = [name for name in names if name not in nodes]
        if missing:
            raise FreezeRebindError(f"proof component {group} is incomplete: {missing}")
        raw = "\n".join(_dump(nodes[name]) for name in names).encode("utf-8")
        result[group] = _sha256(raw)
    return result


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FreezeRebindError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def rebind_freeze(
    *,
    accepted_freeze: Path,
    accepted_freeze_sha256: str,
    accepted_source: bytes,
    accepted_source_commit: str,
    accepted_source_path: str,
    accepted_source_sha256: str,
    onion_source_sha256: str,
    output_freeze: Path,
    output_receipt: Path,
    onion_source: Path | None = None,
    onion_source_bytes: bytes | None = None,
    onion_source_path: str | None = None,
) -> dict[str, object]:
    """Prove a move-only source rebind and emit the minimally changed freeze."""
    old_freeze_raw = Path(accepted_freeze).read_bytes()
    if _sha256(old_freeze_raw) != accepted_freeze_sha256:
        raise FreezeRebindError("accepted freeze SHA-256 differs")
    if _sha256(accepted_source) != accepted_source_sha256:
        raise FreezeRebindError("accepted decision source SHA-256 differs")
    if (onion_source is None) == (onion_source_bytes is None):
        raise FreezeRebindError("provide exactly one onion source input")
    if onion_source is not None:
        onion_path_text = str(onion_source)
        if onion_source.is_symlink() or not onion_source.is_file():
            raise FreezeRebindError("onion decision source is not a regular file")
        new_source = onion_source.read_bytes()
    else:
        onion_path_text = onion_source_path or "<provided-bytes>"
        new_source = onion_source_bytes or b""
    if _sha256(new_source) != onion_source_sha256:
        raise FreezeRebindError("onion decision source SHA-256 differs")

    old_tree = _parse(accepted_source, accepted_source_path)
    new_tree = _parse(new_source, onion_path_text)
    old_normalized, new_normalized = _normalized_dump(old_tree), _normalized_dump(new_tree)
    normalized_diff = list(difflib.unified_diff(
        old_normalized.splitlines(), new_normalized.splitlines(),
        fromfile=accepted_source_path, tofile=onion_path_text, lineterm="",
    ))
    if normalized_diff:
        raise FreezeRebindError("normalized AST differs outside admitted move-only bindings")

    old_components = _component_hashes(old_tree)
    new_components = _component_hashes(new_tree)
    if old_components != new_components:
        raise FreezeRebindError("required decision component differs")
    import_differences = _differing_count(
        _import_targets(old_tree), _import_targets(new_tree), "import target",
    )
    file_path_differences = _differing_count(
        _file_path_expressions(old_tree), _file_path_expressions(new_tree),
        "__file__-derived module path",
    )

    try:
        old_freeze = json.loads(old_freeze_raw)
    except json.JSONDecodeError as error:
        raise FreezeRebindError("accepted freeze is not JSON") from error
    if not isinstance(old_freeze, dict):
        raise FreezeRebindError("accepted freeze is not an object")
    if old_freeze.get("decision_source_sha256") != accepted_source_sha256:
        raise FreezeRebindError("accepted freeze does not bind accepted decision source")
    new_freeze = deepcopy(old_freeze)
    new_freeze["decision_source_sha256"] = onion_source_sha256
    changed_keys = sorted(
        key for key in set(old_freeze) | set(new_freeze)
        if old_freeze.get(key) != new_freeze.get(key)
    )
    if changed_keys != ["decision_source_sha256"]:
        raise FreezeRebindError("freeze rebind changed fields beyond decision source")
    _write_new(Path(output_freeze), new_freeze)
    new_freeze_raw = Path(output_freeze).read_bytes()

    tool_path = Path(__file__).resolve()
    receipt: dict[str, object] = {
        "schema_version": "wp7-vlm-freeze-onion-rebind-receipt-v1",
        "sources": {
            "accepted": {
                "git_commit": accepted_source_commit,
                "path": accepted_source_path,
                "sha256": accepted_source_sha256,
            },
            "onion": {"path": onion_path_text, "sha256": onion_source_sha256},
        },
        "artifacts": {
            "accepted_freeze_path": str(accepted_freeze),
            "old_freeze_sha256": accepted_freeze_sha256,
            "new_freeze_path": str(output_freeze),
            "new_freeze_sha256": _sha256(new_freeze_raw),
        },
        "proof": {
            "policy": "ignore-only-import-targets-and-file-derived-module-path-expressions",
            "normalized_ast_sha256": _sha256(old_normalized.encode("utf-8")),
            "normalized_ast_diff": normalized_diff,
            "ignored_differences": {
                "import_targets": import_differences,
                "file_derived_module_paths": file_path_differences,
            },
            "identical_components": old_components,
            "freeze_changed_keys": changed_keys,
            "tool_path": str(tool_path),
            "tool_sha256": _sha256(tool_path.read_bytes()),
        },
    }
    _write_new(Path(output_receipt), receipt)
    return receipt


def _git_blob(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"], check=True, stdout=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise FreezeRebindError("accepted git source blob is unavailable") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-freeze", type=Path, required=True)
    parser.add_argument("--accepted-freeze-sha256", required=True)
    parser.add_argument("--accepted-source-commit", required=True)
    parser.add_argument("--accepted-source-path", required=True)
    parser.add_argument("--accepted-source-sha256", required=True)
    parser.add_argument("--onion-source", type=Path, required=True)
    parser.add_argument("--onion-source-sha256", required=True)
    parser.add_argument("--output-freeze", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    rebind_freeze(
        accepted_freeze=args.accepted_freeze,
        accepted_freeze_sha256=args.accepted_freeze_sha256,
        accepted_source=_git_blob(args.accepted_source_commit, args.accepted_source_path),
        accepted_source_commit=args.accepted_source_commit,
        accepted_source_path=args.accepted_source_path,
        accepted_source_sha256=args.accepted_source_sha256,
        onion_source=args.onion_source,
        onion_source_sha256=args.onion_source_sha256,
        output_freeze=args.output_freeze,
        output_receipt=args.output_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
