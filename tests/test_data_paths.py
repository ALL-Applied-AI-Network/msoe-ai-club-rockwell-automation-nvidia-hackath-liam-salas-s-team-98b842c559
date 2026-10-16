"""Data-root lookup and manifest check, on a tiny fake dataset."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix import paths


def test_root_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    monkeypatch.setattr(paths, "_dotenv_value", lambda name: None)
    assert paths.data_root() == ROOT / "external"
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "env"))
    assert paths.data_root() == tmp_path / "env"
    assert paths.data_root(tmp_path / "arg") == tmp_path / "arg"


def test_manifest_round_trip_and_mismatch(tmp_path):
    vial = tmp_path / "data/csi-vial-fill-dataset/vial_fill"
    vial.mkdir(parents=True)
    (vial / "image_index.csv").write_text("video_id,filename\n")
    manifest = tmp_path / "manifest.csv"

    def run(*extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_data.py"), str(tmp_path / "data"),
             "--manifest", str(manifest), *extra], text=True, capture_output=True)

    assert run("--write").returncode == 0
    assert run().returncode == 0
    (vial / "image_index.csv").write_text("video_id,filename\nchanged\n")
    result = run()
    assert result.returncode != 0 and "size" in result.stdout
