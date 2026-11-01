"""Review-only contact sheet tests for Task 2A lossless captures."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "cli/twin/render/twin_contact_sheet.py"

sys.path.insert(0, str(ROOT / "tests"))
from tests.twin.render.test_render_process import write_complete_capture
sys.path.pop(0)


def load_contact_sheet():
    name = "fluidmix_test_twin_contact_sheet"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _write_summary(capture, summary):
    (capture / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")


def test_liquid_tint_changes_only_emitted_mask_coordinates():
    module = load_contact_sheet()
    rgb = np.array(
        [[[255, 0, 255], [1, 2, 3]], [[4, 5, 6], [7, 8, 9]]], dtype=np.uint8
    )
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)

    overlay = module.tint_liquid_mask(rgb, mask)

    changed = np.any(overlay != rgb, axis=2)
    assert np.array_equal(changed, mask == 255)
    assert np.array_equal(rgb[0, 0], [255, 0, 255])


def test_contact_sheet_uses_verified_rgb_and_liquid_mask_at_same_coordinates(tmp_path):
    module = load_contact_sheet()
    capture = tmp_path / "capture"
    write_complete_capture(capture)
    output = tmp_path / "sheet.png"

    result = module.create_contact_sheet(capture, output)

    assert result == output.resolve()
    with Image.open(output) as decoded:
        decoded.load()
        sheet = np.asarray(decoded.convert("RGB"))
    with Image.open(capture / "rgb/0001.png") as source:
        rgb = np.asarray(source.convert("RGB"))
    with Image.open(capture / "liquid/0001.png") as source:
        mask = np.asarray(source.convert("L"))
    height, width = mask.shape
    assert sheet.shape == (module.CAPTION_HEIGHT + height * 2, width, 3)
    upper = sheet[module.CAPTION_HEIGHT : module.CAPTION_HEIGHT + height]
    lower = sheet[module.CAPTION_HEIGHT + height :]
    assert np.array_equal(upper, rgb)
    assert np.array_equal(np.any(lower != upper, axis=2), mask == 255)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mechanical_gates_passed", False),
        ("visual_check_status", "accepted"),
    ],
)
def test_contact_sheet_rejects_failed_or_published_capture(tmp_path, field, value):
    module = load_contact_sheet()
    capture = tmp_path / "capture"
    summary = write_complete_capture(capture)
    summary[field] = value
    _write_summary(capture, summary)

    with pytest.raises(ValueError):
        module.create_contact_sheet(capture, tmp_path / "sheet.png")


def test_contact_sheet_rejects_wrong_artifact_hash(tmp_path):
    module = load_contact_sheet()
    capture = tmp_path / "capture"
    write_complete_capture(capture)
    (capture / "liquid/0001.png").write_bytes(b"not the captured mask")

    with pytest.raises(ValueError, match="artifact"):
        module.create_contact_sheet(capture, tmp_path / "sheet.png")


def test_contact_sheet_refuses_existing_and_dangling_output(tmp_path):
    module = load_contact_sheet()
    capture = tmp_path / "capture"
    write_complete_capture(capture)
    existing = tmp_path / "existing.png"
    existing.write_text("keep")
    dangling = tmp_path / "dangling.png"
    dangling.symlink_to(tmp_path / "missing")

    for output in (existing, dangling):
        with pytest.raises(FileExistsError):
            module.create_contact_sheet(capture, output)
    assert existing.read_text() == "keep"


def test_contact_sheet_exclusive_publish_preserves_competing_output(
    tmp_path, monkeypatch
):
    module = load_contact_sheet()
    capture = tmp_path / "capture"
    write_complete_capture(capture)
    output = tmp_path / "raced.png"
    original_link = module.os.link

    def competing_link(source, destination, **kwargs):
        Path(destination).write_bytes(b"competitor")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "link", competing_link)
    with pytest.raises(FileExistsError):
        module.create_contact_sheet(capture, output)

    assert output.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".raced.png.tmp-*"))


def test_contact_sheet_is_import_and_help_safe():
    module = load_contact_sheet()
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert hasattr(module, "main")
    assert completed.returncode == 0, completed.stderr
    assert "RUN_DIR OUTPUT [LIMIT]" in completed.stdout.replace("\n", " ")
    assert "--frames" in completed.stdout
