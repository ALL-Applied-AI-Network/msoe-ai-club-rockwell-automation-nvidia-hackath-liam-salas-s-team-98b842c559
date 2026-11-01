"""Six-channel learned liquid-mask candidate contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def test_resnet18_unet_has_exact_six_channel_input_and_pixel_output():
    from fluidmix.perception.cnn.cnn_predictor import ResNet18UNet

    model = ResNet18UNet(weights=None)
    current = torch.rand(2, 3, 64, 96)
    reference = torch.rand(2, 3, 64, 96)

    output = model(current, reference)

    assert model.encoder.conv1.in_channels == 6
    assert output.shape == (2, 1, 64, 96)
    assert not any("coord" in name.lower() for name, _module in model.named_modules())
    with pytest.raises(ValueError, match="three RGB channels"):
        model(torch.rand(2, 4, 64, 96), reference)


def test_six_channel_pretrained_stem_duplicates_rgb_weights_at_half_scale():
    from fluidmix.perception.cnn.cnn_predictor import adapt_resnet18_stem_weight

    rgb = torch.arange(64 * 3 * 7 * 7, dtype=torch.float32).reshape(64, 3, 7, 7)
    adapted = adapt_resnet18_stem_weight(rgb)

    assert adapted.shape == (64, 6, 7, 7)
    assert torch.equal(adapted[:, :3], rgb / 2.0)
    assert torch.equal(adapted[:, 3:], rgb / 2.0)


def test_paired_augmentation_preserves_alignment_and_moves_absolute_position():
    from fluidmix.perception.cnn.cnn_predictor import paired_augment

    current = torch.zeros(3, 40, 50)
    reference = torch.zeros(3, 40, 50)
    mask = torch.zeros(1, 40, 50)
    current[:, 15:25, 20:30] = 0.7
    reference[:, 15:25, 20:30] = 0.7
    mask[:, 15:25, 20:30] = 1.0

    augmented_current, augmented_reference, augmented_mask = paired_augment(
        current,
        reference,
        mask,
        generator=torch.Generator().manual_seed(8),
        max_translation_px=8,
    )

    assert torch.equal(augmented_current, augmented_reference)
    assert augmented_mask.shape == mask.shape
    assert int(augmented_mask.sum()) == 100
    assert not torch.equal(augmented_mask, mask)
    assert float(augmented_current.max()) != pytest.approx(0.7)


def test_probability_mask_keeps_approach_a_roi_and_clamp_exclusion():
    from fluidmix.perception.cnn.cnn_predictor import probability_to_visible_prediction
    from fluidmix.perception.baselines.empty_reference_rgb_predictor import fillcam_masks

    probability = np.ones((480, 640), dtype=np.float32)
    prediction = probability_to_visible_prediction(probability, threshold=0.5)
    roi, clamp = fillcam_masks()

    assert prediction.status == "visible_measured"
    assert np.array_equal(prediction.mask, roi & ~clamp)
    assert not np.any(prediction.mask & clamp)
    assert prediction.surface_row == 194


def test_identical_control_bypasses_model_and_abstains():
    from fluidmix.perception.cnn.cnn_predictor import predict_pair

    class MustNotRun(torch.nn.Module):
        def forward(self, current, reference):  # pragma: no cover - must not execute
            raise AssertionError("model must not run on identical control")

    image = np.full((480, 640, 3), 100, dtype=np.uint8)
    prediction = predict_pair(MustNotRun(), image, image.copy())

    assert prediction.status == "identical_control_abstain"
    assert prediction.surface_row is None
    assert not np.any(prediction.mask)


def test_acceptance_requires_every_gate_and_paired_win_with_low_fill_stratum():
    from fluidmix.perception.cnn.cnn_predictor import score_holdout_comparison

    cnn = [
        {
            "case_id": f"C{index}",
            "dice": dice,
            "visible_height_error_px": error,
            "control_kind": "positive",
            "low_fill_first_dispense": index < 2,
        }
        for index, (dice, error) in enumerate(
            zip([0.91, 0.88, 0.90, 0.93, 0.86, 0.92, 0.89, 0.94, 0.87], [2, 5, 1, 3, 4, 2, 2, 1, 3], strict=True)
        )
    ]
    cnn.extend(
        [
            {"case_id": "empty", "dice": 1.0, "visible_height_error_px": None, "control_kind": "empty_vial", "abstained": True, "mask_pixel_count": 0},
            {"case_id": "same", "dice": 1.0, "visible_height_error_px": None, "control_kind": "identical", "abstained": True, "mask_pixel_count": 0},
        ]
    )
    approach_a = [{"case_id": f"C{index}", "dice": 0.84} for index in range(9)]

    result = score_holdout_comparison(cnn, approach_a)

    assert result["decision"] == "replace_approach_a"
    assert result["all_gates_pass"] is True
    assert result["low_fill_first_dispense"]["count"] == 2
    failed = [dict(row) for row in cnn]
    failed[0]["visible_height_error_px"] = 6
    assert score_holdout_comparison(failed, approach_a)["decision"] == "retain_approach_a"
