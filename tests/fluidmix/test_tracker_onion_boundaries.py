from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _tree(relative_path: str) -> ast.AST:
    return ast.parse((ROOT / relative_path).read_text(), filename=relative_path)


def test_measurement_calibration_does_not_depend_on_twin() -> None:
    tree = _tree("src/fluidmix/measurement/fill_step_calibration.py")
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert not any(name == "twin" or name.startswith("twin.") for name in imported)


def test_perception_tracker_modules_do_not_perform_file_io() -> None:
    forbidden_methods = {
        "exists", "is_file", "is_symlink", "lstat", "mkdir", "open",
        "read_bytes", "read_text", "write_bytes", "write_text",
    }
    for relative_path in (
        "src/fluidmix/perception/tracker/fill_step_production.py",
        "src/fluidmix/perception/tracker/tracker_runtime.py",
    ):
        tree = _tree(relative_path)
        calls = {
            (node.func.value.id, node.func.attr)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        }

        assert ("os", "open") not in calls, relative_path
        assert ("Image", "open") not in calls, relative_path
        assert not ({method for _owner, method in calls} & forbidden_methods), relative_path
