"""CPU-only checks for renderer summary integrity rules."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.render_fill_level_dataset import frame_integrity_ok  # noqa: E402


def test_positive_foamy_frame_may_have_no_visible_liquid_pixels():
    record = {
        "level_ml": 0.065,
        "foam": {"enabled": True, "visible_in_camera": True},
        "label_stats": {
            "class_pixels": {
                "empty": 16850,
                "liquid": 0,
                "foam": 1391,
                "blocked": 3411,
                "ignore": 285548,
            }
        },
    }
    assert frame_integrity_ok(record)


def test_zero_level_without_visible_empty_pixels_fails_integrity():
    record = {
        "level_ml": 0.0,
        "foam": {"enabled": False, "visible_in_camera": False},
        "label_stats": {
            "class_pixels": {
                "empty": 0,
                "liquid": 10,
                "foam": 0,
                "blocked": 4,
                "ignore": 100,
            }
        },
    }
    assert not frame_integrity_ok(record)
