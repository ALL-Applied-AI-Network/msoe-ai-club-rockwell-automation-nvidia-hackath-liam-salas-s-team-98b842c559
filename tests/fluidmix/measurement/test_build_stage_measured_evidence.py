"""Public CLI tests for assembling measured stage evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tests.fluidmix.storage.artifacts.test_capture_manifest_adapter import _manifest, _model_plan


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "cli/fluidmix/measurement" / "build_stage_measured_evidence.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _inputs(tmp_path: Path) -> tuple[Path, list[Path]]:
    plan_path = _write_json(tmp_path / "model-input-plan.json", _model_plan("TARGET", ["REF-1"]))
    capture_paths: list[Path] = []
    for batch_id, rgb in (("TARGET", (20, 40, 80)), ("REF-1", (20, 40, 80))):
        capture_dir = tmp_path / batch_id
        manifest = _manifest(capture_dir, batch_id=batch_id, rgb_value=rgb)
        capture_paths.append(_write_json(capture_dir / "capture-manifest.json", manifest))
    return plan_path, capture_paths


def _run(
    plan: Path, captures: list[Path], output: Path, locator_output: Path
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--model-input-plan",
        str(plan),
    ]
    for capture in captures:
        command.extend(("--capture-manifest", str(capture)))
    command.extend(("--out", str(output), "--image-locators-out", str(locator_output)))
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def test_cli_publishes_measured_evidence_with_source_hashes(tmp_path):
    plan, captures = _inputs(tmp_path)
    output_dir = tmp_path / "published"
    output_dir.mkdir(mode=0o755)
    output = output_dir / "measured-evidence.json"
    locator_output = output_dir / "image-locators.json"

    result = _run(plan, captures, output, locator_output)

    assert result.returncode == 0, result.stderr
    measured = json.loads(output.read_text(encoding="utf-8"))
    assert measured["artifact_type"] == "measured_evidence"
    assert measured["source_model_input_sha256"] == _sha256(plan)
    assert measured["source_capture_manifest_sha256"] == [
        _sha256(path) for path in captures
    ]
    assert measured["decision_records"][0]["disposition"] == "insufficient_evidence"
    assert measured["denominators"] == {
        "image_measurements": 18,
        "valid_image_measurements": 18,
        "invalid_image_measurements": 0,
        "invalid_image_measurement_reasons": {},
        "vial_decisions": 1,
        "valid_vial_decisions": 0,
        "abstained_vial_decisions": 1,
    }
    assert "latent_parameters" not in json.dumps(measured)
    assert "expected_disposition" not in json.dumps(measured)
    locators = json.loads(locator_output.read_text(encoding="utf-8"))
    assert set(locators) == {"schema_version", "data_origin", "images"}
    assert locators["schema_version"] == 1
    assert locators["data_origin"] == "twin"
    assert set(locators["images"]) == {
        f"{batch}-{vial}-{stage}"
        for batch in ("TARGET", "REF-1")
        for vial in ("V1", "V2", "V3")
        for stage in ("after-t200", "after-t400", "after-t100")
    }
    for locator in locators["images"].values():
        assert Path(locator["rgb_path"]).is_file()
        assert locator["rgb_sha256"] == _sha256(Path(locator["rgb_path"]))
        assert locator["mask_sha256"] == _sha256(Path(locator["mask_path"]))


def test_cli_does_not_clobber_existing_output(tmp_path):
    plan, captures = _inputs(tmp_path)
    output = tmp_path / "measured-evidence.json"
    locator_output = tmp_path / "image-locators.json"
    output.write_text("sentinel\n", encoding="utf-8")

    result = _run(plan, captures, output, locator_output)

    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == "sentinel\n"
    assert not locator_output.exists()
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))


def test_cli_rejects_duplicate_json_key_without_partial_output(tmp_path):
    plan, captures = _inputs(tmp_path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"artifact_type": "x", "artifact_type": "y"}\n', encoding="utf-8")
    output = tmp_path / "measured-evidence.json"
    locator_output = tmp_path / "image-locators.json"

    result = _run(plan, [duplicate], output, locator_output)

    assert result.returncode != 0
    assert "JSON contains duplicate key 'artifact_type'" in result.stderr
    assert not output.exists()
    assert not locator_output.exists()


def test_cli_rejects_duplicate_capture_batch_without_partial_output(tmp_path):
    plan, captures = _inputs(tmp_path)
    output = tmp_path / "measured-evidence.json"
    locator_output = tmp_path / "image-locators.json"

    result = _run(plan, [captures[0], captures[0], captures[1]], output, locator_output)

    assert result.returncode != 0
    assert "duplicate_capture_batch: TARGET" in result.stderr
    assert not output.exists()
    assert not locator_output.exists()


def test_cli_rejects_pixel_hash_drift_without_partial_output(tmp_path):
    plan, captures = _inputs(tmp_path)
    manifest = json.loads(captures[0].read_text(encoding="utf-8"))
    rgb_path = Path(manifest["rows"][0]["rgb"]["path"])
    rgb_path.write_bytes(b"tampered")
    output = tmp_path / "measured-evidence.json"
    locator_output = tmp_path / "image-locators.json"

    result = _run(plan, captures, output, locator_output)

    assert result.returncode != 0
    assert "rgb_hash_mismatch" in result.stderr
    assert not output.exists()
    assert not locator_output.exists()


def test_cli_rejects_existing_locator_without_publishing_either_output(tmp_path):
    plan, captures = _inputs(tmp_path)
    output = tmp_path / "measured-evidence.json"
    locator_output = tmp_path / "image-locators.json"
    locator_output.write_text("sentinel\n", encoding="utf-8")

    result = _run(plan, captures, output, locator_output)

    assert result.returncode != 0
    assert not output.exists()
    assert locator_output.read_text(encoding="utf-8") == "sentinel\n"
