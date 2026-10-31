from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "fluidmix"
INNER_RINGS = ("core", "colour", "perception", "measurement", "decision")
FORBIDDEN_IMPORT_ROOTS = {
    "aiohttp",
    "ftplib",
    "http",
    "httpx",
    "requests",
    "socket",
    "sqlite3",
    "subprocess",
    "urllib",
}


def _import_roots(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.partition(".")[0] for alias in node.names}
    if node.module is None:
        return set()
    return {node.module.partition(".")[0]}


def _open_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id == "open":
        return "open"
    if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
        return "open"
    return None


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for root in sorted(_import_roots(node) & FORBIDDEN_IMPORT_ROOTS):
                findings.append(f"line {node.lineno}: imports {root}")
        elif isinstance(node, ast.Call):
            name = _open_call_name(node)
            if name is not None:
                findings.append(f"line {node.lineno}: calls {name}()")
    return sorted(findings, key=lambda finding: int(finding.split(":", 1)[0][5:]))


INNER_RING_MODULES = tuple(
    path
    for ring in INNER_RINGS
    for path in sorted((SOURCE_ROOT / ring).rglob("*.py"))
)


@pytest.mark.parametrize(
    "module_path",
    INNER_RING_MODULES,
    ids=lambda path: path.relative_to(SOURCE_ROOT).as_posix(),
)
def test_inner_ring_modules_do_not_perform_io(module_path: Path) -> None:
    violations = _violations(module_path)
    assert not violations, "\n" + "\n".join(violations)
