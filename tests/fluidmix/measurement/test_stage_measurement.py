"""Evidence-only per-stage colour measurements with explicit peer references."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.measurement.stage_measurement import StageImage, measure_stage_evidence


def _record(
    record_id: str,
    rgb_value: tuple[int, int, int],
    *,
    run_id: str = "RUN-1",
    recipe_id: str = "S0380",
    data_origin: str = "twin",
    split_role: str = "val",
    stage_id: str = "after-t400",
    stage_ordinal: int = 2,
    stage_tank_id: str = "t400",
    dispense_order: tuple[str, ...] = ("t200", "t400", "t100"),
    mask: np.ndarray | None = None,
) -> StageImage:
    rgb = np.empty((3, 4, 3), dtype=np.uint8)
    rgb[:] = rgb_value
    if mask is None:
        mask = np.array(
            [[0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]],
            dtype=np.uint8,
        )
    return StageImage(
        record_id=record_id,
        image_id=f"IMG-{record_id}",
        run_id=run_id,
        recipe_id=recipe_id,
        data_origin=data_origin,
        split_role=split_role,
        vial_id=f"VIAL-{record_id}",
        stage_id=stage_id,
        stage_ordinal=stage_ordinal,
        stage_tank_id=stage_tank_id,
        dispense_order=dispense_order,
        rgb=rgb,
        liquid_mask=mask,
    )


def test_two_explicit_references_produce_grounded_delta_evidence_only():
    target = _record("TARGET", (30, 60, 90))
    refs = [
        _record("REF-1", (28, 58, 88), run_id="RUN-1"),
        _record("REF-2", (32, 62, 92), run_id="RUN-2"),
    ]

    result = measure_stage_evidence(
        target, refs, minimum_reference_count=2
    )

    assert result["validity"] == "valid"
    assert result["limit_codes"] == []
    assert result["unit"] == "delta_e76"
    assert result["target_delta_e76"] > 0
    assert result["reference"]["count"] == 2
    assert result["reference"]["record_ids"] == ["REF-1", "REF-2"]
    assert result["reference"]["run_ids"] == ["RUN-1", "RUN-2"]
    assert result["reference"]["same_run_count"] == 1
    assert result["reference"]["other_run_count"] == 1
    assert result["reference"]["median_radius_delta_e76"] >= 0
    assert result["reference"]["max_radius_delta_e76"] >= 0
    assert result["reference"]["pairwise_diameter_delta_e76"] >= 0
    assert all(math.isfinite(value) for value in result["target"]["lab"])
    assert "decision" not in result
    assert "cause" not in result


def test_one_reference_is_valid_only_when_the_frozen_minimum_allows_it():
    target = _record("TARGET", (10, 20, 30))
    reference = _record("REF", (11, 21, 31), run_id="RUN-2")

    allowed = measure_stage_evidence(
        target, [reference], minimum_reference_count=1
    )
    thin = measure_stage_evidence(
        target, [reference], minimum_reference_count=2
    )

    assert allowed["validity"] == "valid"
    assert allowed["reference"]["count"] == 1
    assert allowed["reference"]["median_radius_delta_e76"] == 0.0
    assert allowed["reference"]["pairwise_diameter_delta_e76"] == 0.0
    assert thin["validity"] == "insufficient_evidence"
    assert thin["limit_codes"] == ["reference_count_below_minimum"]
    assert thin["reference"]["count"] == 1
    assert thin["target_delta_e76"] is None


def test_two_later_vials_in_the_same_three_vial_run_are_supported():
    target = _record("V1", (40, 70, 100), run_id="BATCH-1")
    references = [
        _record("V2", (41, 71, 101), run_id="BATCH-1"),
        _record("V3", (42, 72, 102), run_id="BATCH-1"),
    ]

    result = measure_stage_evidence(
        target, references, minimum_reference_count=2
    )

    assert result["validity"] == "valid"
    assert result["reference"]["same_run_count"] == 2
    assert result["reference"]["other_run_count"] == 0


def test_reference_center_and_spread_use_fixed_cie76_definitions():
    result = measure_stage_evidence(
        _record("TARGET", (127, 127, 127)),
        [
            _record("BLACK", (0, 0, 0)),
            _record("WHITE", (255, 255, 255), run_id="RUN-2"),
        ],
        minimum_reference_count=2,
    )

    assert result["reference"]["center_lab"] == pytest.approx(
        [50.0, -0.00122747, 0.00232671], abs=1e-5
    )
    assert result["reference"]["median_radius_delta_e76"] == pytest.approx(
        50.0, abs=1e-5
    )
    assert result["reference"]["max_radius_delta_e76"] == pytest.approx(
        50.0, abs=1e-5
    )
    assert result["reference"]["mad_radius_delta_e76"] == pytest.approx(
        0.0, abs=1e-5
    )
    assert result["reference"]["pairwise_diameter_delta_e76"] == pytest.approx(
        100.0, abs=1e-5
    )
    assert result["target_delta_e76"] == pytest.approx(3.192778, abs=1e-5)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"data_origin": "real_check"}, "data_origin_mismatch"),
        ({"recipe_id": "S9999"}, "recipe_mismatch"),
        ({"split_role": "test"}, "split_mismatch"),
        ({"stage_id": "after-t200"}, "stage_id_mismatch"),
        ({"stage_ordinal": 1, "stage_tank_id": "t200"}, "stage_ordinal_mismatch"),
        ({"stage_tank_id": "t100"}, "stage_tank_mismatch"),
        (
            {
                "dispense_order": ("t400", "t200", "t100"),
                "stage_tank_id": "t200",
            },
            "stage_tank_mismatch",
        ),
    ],
)
def test_reference_context_mismatch_abstains(change, expected):
    target = _record("TARGET", (10, 20, 30))
    reference = _record("REF", (11, 21, 31), **change)

    result = measure_stage_evidence(
        target, [reference], minimum_reference_count=1
    )

    assert result["validity"] == "insufficient_evidence"
    assert any(expected in code for code in result["limit_codes"])
    assert result["target_delta_e76"] is None


def test_target_record_image_and_exact_rgb_bytes_cannot_be_references():
    target = _record("TARGET", (10, 20, 30))
    same_record = _record("TARGET", (11, 21, 31))
    same_image = _record("OTHER", (12, 22, 32))
    object.__setattr__(same_image, "image_id", target.image_id)
    same_bytes = _record("SAME-BYTES", (10, 20, 30))

    for reference, code in (
        (same_record, "target_record_reused_as_reference"),
        (same_image, "target_image_reused_as_reference"),
        (same_bytes, "target_rgb_bytes_reused_as_reference"),
    ):
        result = measure_stage_evidence(
            target, [reference], minimum_reference_count=1
        )
        assert result["validity"] == "insufficient_evidence"
        assert code in result["limit_codes"]


def test_duplicate_reference_identity_or_rgb_bytes_cannot_inflate_count():
    target = _record("TARGET", (10, 20, 30))
    first = _record("REF-1", (11, 21, 31))
    same_id = _record("REF-1", (12, 22, 32), run_id="RUN-2")
    same_bytes = _record("REF-2", (11, 21, 31), run_id="RUN-2")

    identity = measure_stage_evidence(
        target, [first, same_id], minimum_reference_count=2
    )
    pixels = measure_stage_evidence(
        target, [first, same_bytes], minimum_reference_count=2
    )

    assert identity["validity"] == "insufficient_evidence"
    assert "duplicate_reference_identity" in identity["limit_codes"]
    assert pixels["validity"] == "insufficient_evidence"
    assert "duplicate_reference_rgb_bytes" in pixels["limit_codes"]


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (np.zeros((2, 2), dtype=np.uint8), "mask_shape_or_dtype_invalid"),
        (np.zeros((3, 4), dtype=np.uint8), "mask_empty"),
        (np.full((3, 4), 7, dtype=np.uint8), "mask_not_binary"),
    ],
)
def test_invalid_target_masks_abstain(mask, expected):
    target = _record("TARGET", (10, 20, 30), mask=mask)
    reference = _record("REF", (11, 21, 31))

    result = measure_stage_evidence(
        target, [reference], minimum_reference_count=1
    )

    assert result["validity"] == "insufficient_evidence"
    assert result["limit_codes"] == [f"target:{expected}"]


def test_nonfinite_or_non_uint8_rgb_is_rejected_before_conversion():
    target = _record("TARGET", (10, 20, 30))
    object.__setattr__(
        target,
        "rgb",
        np.full((3, 4, 3), np.nan, dtype=np.float64),
    )

    result = measure_stage_evidence(
        target, [_record("REF", (11, 21, 31))], minimum_reference_count=1
    )

    assert result["validity"] == "insufficient_evidence"
    assert result["limit_codes"] == ["target:rgb_shape_or_dtype_invalid"]


def test_target_vial_cannot_be_reused_under_new_record_and_image_ids():
    target = _record("TARGET", (10, 20, 30))
    reference = _record("REF", (11, 21, 31))
    object.__setattr__(reference, "vial_id", target.vial_id)

    result = measure_stage_evidence(
        target, [reference], minimum_reference_count=1
    )

    assert result["validity"] == "insufficient_evidence"
    assert result["limit_codes"] == ["target_vial_reused_as_reference"]


def test_one_vial_cannot_supply_two_reference_records():
    target = _record("TARGET", (10, 20, 30))
    first = _record("REF-1", (11, 21, 31))
    second = _record("REF-2", (12, 22, 32))
    object.__setattr__(second, "vial_id", first.vial_id)

    result = measure_stage_evidence(
        target, [first, second], minimum_reference_count=2
    )

    assert result["validity"] == "insufficient_evidence"
    assert result["limit_codes"] == ["duplicate_reference_identity"]


def test_reference_spread_nonfinite_abstains(monkeypatch):
    import fluidmix.measurement.stage_measurement as measurement

    target = _record("TARGET", (10, 20, 30))
    references = [
        _record("REF-1", (11, 21, 31)),
        _record("REF-2", (12, 22, 32), run_id="RUN-2"),
    ]
    real_norm = measurement.np.linalg.norm

    def nonfinite_radius(values, axis=None):
        if axis == 1:
            return np.full(values.shape[0], np.inf)
        return real_norm(values, axis=axis)

    monkeypatch.setattr(measurement.np.linalg, "norm", nonfinite_radius)
    result = measure_stage_evidence(
        target, references, minimum_reference_count=2
    )

    assert result["validity"] == "insufficient_evidence"
    assert result["limit_codes"] == ["reference_spread_nonfinite"]
    assert result["target_delta_e76"] is None


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_minimum_reference_count_must_be_a_frozen_positive_integer(minimum):
    with pytest.raises(ValueError, match="positive integer"):
        measure_stage_evidence(
            _record("TARGET", (10, 20, 30)),
            [_record("REF", (11, 21, 31))],
            minimum_reference_count=minimum,
        )
