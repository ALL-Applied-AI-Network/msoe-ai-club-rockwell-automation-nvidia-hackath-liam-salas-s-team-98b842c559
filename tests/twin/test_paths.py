from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def test_twin_repository_root_is_owned_by_twin_package():
    from twin.paths import REPO

    assert REPO == ROOT
