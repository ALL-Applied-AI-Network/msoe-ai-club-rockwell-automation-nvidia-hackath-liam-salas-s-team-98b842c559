"""Synthetic checks for the SAM evaluation helpers. No GPU, torch or SAM needed."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sam_eval import MODELS, ROW0, ROW1, FloodFill, contact_sheet, is_stripe, mask_stats


def stripe(width, x=303, rows=(ROW0, ROW1)):
    mask = np.zeros((480, 640), bool)
    mask[rows[0]:rows[1], x:x + width] = True
    return mask


def test_mask_stats_of_a_box():
    mask = np.zeros((480, 640), bool)
    mask[100:150, 300:306] = True
    s = mask_stats(mask)
    assert (s["px"], s["top"], s["bottom"], s["height"], s["width"]) == (300, 100, 149, 50, 6.0)
    assert mask_stats(np.zeros((4, 4), bool))["px"] == 0


def test_stream_test_accepts_a_thin_stripe_under_the_nozzle_only():
    assert is_stripe(stripe(6))
    assert not is_stripe(stripe(40))              # the whole vial, not a stream
    assert not is_stripe(stripe(6, x=100))        # thin, but not under the nozzle
    assert not is_stripe(stripe(6, rows=(ROW0, ROW0 + 10)))   # too short


def test_flood_fill_stand_in_stays_inside_a_uniform_region_and_the_box():
    frame = np.full((480, 640, 3), 200, np.uint8)
    frame[:, 320:] = 50
    mask = FloodFill().image(frame, [(100, 100)])
    assert mask[:, :320].all() and not mask[:, 320:].any()
    boxed = FloodFill().image(frame, [(100, 100)], box=(50, 50, 150, 150))
    assert boxed.sum() == 100 * 100


def test_every_model_has_a_checkpoint_and_a_config():
    for ckpt, cfg in MODELS.values():
        assert ckpt.endswith(".pt") and cfg.startswith("configs/sam2.1/")


def test_contact_sheet_tiles_images():
    sheet = contact_sheet([np.zeros((480, 640, 3), np.uint8)] * 7, cols=3, width=160)
    assert sheet.shape == (3 * 120, 3 * 160, 3)
