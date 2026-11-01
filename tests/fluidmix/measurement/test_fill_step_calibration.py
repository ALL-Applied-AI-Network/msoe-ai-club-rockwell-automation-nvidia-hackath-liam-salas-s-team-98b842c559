from __future__ import annotations

import pytest

from fluidmix.measurement.fill_step_calibration import build_development_calibration
from fluidmix.perception.tracker.fill_step_tracker import calibration_payload_sha256


def test_development_builder_freezes_reviewed_projection_and_curve() -> None:
    # Rows are independent USD/FillCam projection measurements already covered
    # by the adjacent volume-gate tests: approximately 10 mL and 20 mL.
    calibration = build_development_calibration(
        calibration_id="V3-V4-DEVELOPMENT-1",
        fill_height_surface_row_samples=(
            (44.257340, 360.742660),
            (68.493011, 336.506989),
        ),
    )

    assert calibration.visible_height_m_samples == pytest.approx(
        (0.019122235, 0.038244470), abs=1.0e-8
    )
    assert calibration.twin_geometry_equivalent_volume_ml_samples == pytest.approx(
        (10.404817, 20.809634), abs=0.000002
    )
    assert calibration.artifact_sha256 == calibration_payload_sha256(calibration)
