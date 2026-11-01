"""CPU-only contracts for Step 8 FillCam ruler calibration."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluidmix import fill_level_ruler as ruler


SCRIPT = ROOT / "scripts" / "calibrate_fill_level_ruler.py"
SPEC = importlib.util.spec_from_file_location("calibrate_fill_level_ruler", SCRIPT)
CALIBRATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATE)


COMBINED_VALIDATION_CASE_COUNTS = {
    "clean": 78,
    "foam": 177,
    "sticker": 207,
    "edge": 182,
    "hidden": 90,
}
VALIDATION_PROVENANCE = {
    300: {
        "batch_summary_sha256": (
            "e736d17ee7db384fa9b15cadc97422bb095b2a82ef1fb6b879040df0424d88c0"
        ),
        "exception_review_sha256": (
            "6ef12c4adce78e1c5f0fe89c5a37043d19722b003e7232c79f02f2efb366130f"
        ),
        "ordered_hash_ledger_sha256": (
            "8969e9727149dc2f987b2afb0524fc042f46d5396adbb199411ce1c2ca2066dd"
        ),
        "samples_checked": 300,
        "files_checked": 600,
    },
    360: {
        "batch_summary_sha256": (
            "bfe6650e2d60311411284e4abbf63b71592eb22904b88c50a2da02dfa3ebd6b5"
        ),
        "exception_review_sha256": (
            "e0efefdcd3e603dbb6f88404d8301e20db3d431728b94b19e544cd1c9a189eea"
        ),
        "ordered_hash_ledger_sha256": (
            "680d6d382ed8d55f0d87eb76c5605a290242b95d6e1c2ba5acad32e634204f4c"
        ),
        "samples_checked": 360,
        "files_checked": 720,
    },
}
HOLDER_RULE_CONSTANTS = {
    "holder_band_rows_full_inclusive": [217, 275],
    "surface_oval_height_rows": 17,
    "visibility_columns_full_half_open": [279, 340],
    "minimum_liquid_pixels_per_visibility_row": 3,
}
HOLDER_RULE_EVIDENCE = {
    "selection_source": "synthetic_validation_labels_and_manifest_geometry",
    "role": "val",
    "frame_count": 300,
    "ordered_hash_ledger_sha256": (
        "8969e9727149dc2f987b2afb0524fc042f46d5396adbb199411ce1c2ca2066dd"
    ),
    "source_component": "step6",
    "source_batch_summary_sha256": (
        "e736d17ee7db384fa9b15cadc97422bb095b2a82ef1fb6b879040df0424d88c0"
    ),
    **HOLDER_RULE_CONSTANTS,
}


def _write_index(root):
    windows = []
    frames = []
    for vial, volume, recipe in ((1, 30.0, "5"), (13, 80.0, "9")):
        start, end = ruler.CALIBRATION_FRAME_SPANS[vial]
        windows.append(
            {
                "vial": vial,
                "recipe_id": recipe,
                "level_ml": volume,
                "role": "calibration",
                "source_role": "real_validation",
                "window_start_s": 1.0,
                "window_end_s": 2.0,
                "start_frame": start,
                "end_frame": end,
                "n_frames": end - start + 1,
            }
        )
        vial_dir = root / f"vial-{vial:02d}"
        vial_dir.mkdir()
        for frame_index in range(start, end + 1):
            path = vial_dir / f"frame-{frame_index:06d}.png"
            path.write_bytes(f"synthetic-{vial}-{frame_index}".encode())
            frames.append(
                {
                    "vial": vial,
                    "recipe_id": recipe,
                    "level_ml": volume,
                    "role": "calibration",
                    "source_window_role": "real_validation",
                    "frame_index": frame_index,
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "width": 640,
                    "height": 480,
                }
            )
    value = {
        "schema_version": 1,
        "selection_policy": "all frames in each committed inclusive window",
        "source": {
            "video": {
                "path": "C:/ignored/raw/video.mp4",
                "sha256": CALIBRATE.EXPECTED_VIDEO_SHA256,
            },
            "windows_csv": {
                "path": "C:/repo/data/real_check_windows.csv",
                "sha256": CALIBRATE.EXPECTED_WINDOWS_SHA256,
            },
        },
        "image_contract": {
            "encoding": "PNG",
            "lossless": True,
            "width": 640,
            "height": 480,
        },
        "windows": windows,
        "frame_count": len(frames),
        "frames": frames,
    }
    path = root / "index.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def _validation_rule(
    checkpoint_hash,
    *,
    frame_count=300,
    case_counts=None,
    holder_rule=False,
):
    if case_counts is None:
        case_counts = dict(CALIBRATE.EXPECTED_VALIDATION_CASE_COUNTS)
    else:
        case_counts = dict(case_counts)
    provenance = VALIDATION_PROVENANCE[frame_count]
    bin_counts = {
        f"{lower:02d}-{lower + 10:02d}": frame_count // 9
        for lower in range(0, 90, 10)
    }
    for index, name in enumerate(bin_counts):
        if index < frame_count % 9:
            bin_counts[name] += 1

    def aggregate(frames):
        return {
            "frames": frames,
            "visible_predictions": frames,
            "visible_prediction_rate": 1.0,
            "abstained": 0,
            "abstention_rate": 0.0,
            "predicted_hidden": 0,
            "predicted_hidden_rate": 0.0,
            "geometry_in_predicted_range": 0,
            "true_in_range_rate": None,
            "true_in_range_over_all_frames_rate": 0.0,
            "mae_rows": 0.0,
            "median_absolute_error_rows": 0.0,
            "p95_absolute_error_rows": 0.0,
            "max_absolute_error_rows": 0.0,
            "mae_ml": 0.0,
            "median_absolute_error_ml": 0.0,
            "p95_absolute_error_ml": 0.0,
            "max_absolute_error_ml": 0.0,
            "hidden_interval_median_distance_to_label_rows": None,
            "hidden_interval_p95_distance_to_label_rows": None,
            "hidden_interval_max_distance_to_label_rows": None,
            "hidden_interval_median_distance_to_label_ml": None,
            "hidden_interval_p95_distance_to_label_ml": None,
            "hidden_interval_max_distance_to_label_ml": None,
        }

    rule_config = {
        "selection_source": "synthetic_validation_only",
        "crop_xyxy": list(CALIBRATE.training.CROP_XYXY),
        "class_ids": {"liquid": 1, "blocked": 3},
        "measurement_columns_full_half_open": [298, 316],
        "measurement_rows_full_half_open": [83, 427],
        "measurement_row_derivation": {
            "measurement_rows_full_half_open": [83, 427],
            "selection_source": (
                "synthetic_validation_label_outline_intersection"
            ),
            "frame_count": frame_count,
            "role": "val",
            "ordered_hash_ledger_sha256": provenance[
                "ordered_hash_ledger_sha256"
            ],
        },
        "minimum_liquid_pixels_per_row": 1,
        "minimum_blocked_pixels_per_row": 1,
    }
    if holder_rule:
        rule_config.update(
            {
                "hidden_rule_version": "holder_band_v2",
                "hidden_rule_derivation": dict(HOLDER_RULE_EVIDENCE),
                **HOLDER_RULE_CONSTANTS,
            }
        )

    return {
        "status": "complete",
        "inputs": {
            "checkpoint_sha256": checkpoint_hash,
            "batch_summary_sha256": provenance["batch_summary_sha256"],
            "exception_review_sha256": provenance[
                "exception_review_sha256"
            ],
        },
        "dataset": {
            "role": "val",
            "frame_count": frame_count,
            "case_counts": case_counts,
            "preflight": {
                "samples_checked": provenance["samples_checked"],
                "files_checked": provenance["files_checked"],
                "ordered_hash_ledger_sha256": provenance[
                    "ordered_hash_ledger_sha256"
                ],
            },
            "train_manifests_opened": 0,
            "train_images_loaded": 0,
            "test_manifests_opened": 0,
            "test_images_loaded": 0,
            "real_frames_loaded": 0,
        },
        "aggregate": {
            "by_case": {
                "all": aggregate(frame_count),
                **{
                    case: aggregate(count)
                    for case, count in case_counts.items()
                },
            },
            "by_level_bin_10ml": {
                name: aggregate(count) for name, count in bin_counts.items()
            },
        },
        "rule_config": rule_config,
    }


def _write_rule(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_median_uses_only_visible_rows_and_preserves_status_counts():
    result = ruler.median_visible_row(
        [
            {"status": "visible", "liquid_top_row": 330.0},
            {"status": "hidden", "liquid_top_row": 329.0},
            {"status": "visible", "liquid_top_row": 328.0},
            {"status": "abstain", "liquid_top_row": None},
            {"status": "visible", "liquid_top_row": 329.0},
        ]
    )
    assert result["median_row"] == 329.0
    assert result["visible_row_count"] == 3
    assert result["status_counts"] == {"visible": 3, "hidden": 1, "abstain": 1}


def test_median_requires_three_visible_rows():
    with pytest.raises(ValueError, match="at least 3 visible"):
        ruler.median_visible_row(
            [
                {"status": "visible", "liquid_top_row": 329.0},
                {"status": "visible", "liquid_top_row": 330.0},
                {"status": "hidden", "liquid_top_row": 328.0},
            ]
        )


def test_two_point_ruler_matches_reference_geometry_and_converts_ranges():
    fitted = ruler.fit_affine_ruler({1: 329.0, 13: 177.0})
    assert fitted["rows_per_ml"] == pytest.approx(3.04)
    assert ruler.row_to_ml(fitted, 329.0) == pytest.approx(30.0)
    assert ruler.row_to_ml(fitted, 177.0) == pytest.approx(80.0)
    assert ruler.row_range_to_ml(fitted, [177.0, 329.0]) == pytest.approx([30.0, 80.0])
    assert fitted["crosscheck"]["rows_per_ml_absolute_delta"] == pytest.approx(0.0)
    assert all(
        item["absolute_delta_rows"] == pytest.approx(0.0)
        for item in fitted["crosscheck"]["twin_reference_points"]
    )


def test_calibration_target_gate_is_inclusive_and_rejects_drift():
    passed = ruler.validate_calibration_targets({1: 333.0, 13: 173.0})
    assert passed["status"] == "passed"
    assert passed["tolerance_rows_inclusive"] == 4.0
    assert all(item["within_tolerance"] for item in passed["checks"])

    with pytest.raises(ValueError, match="outside.*4-row target tolerance"):
        ruler.validate_calibration_targets({1: 333.01, 13: 177.0})
    with pytest.raises(ValueError, match="30 mL row must be below"):
        ruler.validate_calibration_targets({1: 170.0, 13: 330.0})


def test_ruler_rejects_reversed_calibration_rows():
    with pytest.raises(ValueError, match="30 mL row must be below"):
        ruler.fit_affine_ruler({1: 170.0, 13: 330.0})


def test_index_gate_requires_exact_windows_all_frames_and_safe_paths(tmp_path):
    index_path, value = _write_index(tmp_path)
    indexed = CALIBRATE.validate_frame_index(index_path)

    assert indexed["frame_count"] == 98
    assert [item["vial"] for item in indexed["vials"]] == [1, 13]
    assert indexed["source_video"] == {
        "logical_path": CALIBRATE.LOGICAL_VIDEO_PATH,
        "sha256": CALIBRATE.EXPECTED_VIDEO_SHA256,
    }
    assert indexed["windows"]["path"] == CALIBRATE.LOGICAL_WINDOWS_PATH

    value["frames"][0]["path"] = "../escape.png"
    index_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe calibration frame path"):
        CALIBRATE.validate_frame_index(index_path)


def test_index_gate_rejects_missing_window_frame(tmp_path):
    index_path, value = _write_index(tmp_path)
    value["frames"].pop()
    value["frame_count"] -= 1
    index_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="every committed window frame"):
        CALIBRATE.validate_frame_index(index_path)


def test_index_gate_rejects_wrong_committed_windows_hash(tmp_path):
    index_path, value = _write_index(tmp_path)
    value["source"]["windows_csv"]["sha256"] = "a" * 64
    index_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="committed settled windows"):
        CALIBRATE.validate_frame_index(index_path)


def test_validation_rule_is_bound_to_validation_and_checkpoint(tmp_path):
    checkpoint_hash = "b" * 64
    rule_path = tmp_path / "validation-rule.json"
    rule_hash = _write_rule(
        rule_path, _validation_rule(checkpoint_hash)
    )
    result = CALIBRATE.validate_frozen_rule(
        rule_path, rule_hash, checkpoint_hash
    )
    assert result["rule_options"] == {
        "measurement_columns_full": (298, 316),
        "measurement_rows_full": (83, 427),
        "min_liquid_pixels": 1,
        "min_blocked_pixels": 1,
    }

    mask = CALIBRATE.np.zeros((416, 112), dtype=CALIBRATE.np.uint8)
    mask[0, 42:60] = 1
    mask[35, 42:60] = 1
    prediction = CALIBRATE.surface_eval.surface_prediction(
        mask, **result["rule_options"]
    )
    assert prediction["liquid_top_row"] == 83

    with pytest.raises(ValueError, match="checkpoint hash"):
        CALIBRATE.validate_frozen_rule(rule_path, rule_hash, "c" * 64)


def test_validation_rule_accepts_combined_360_frame_holder_rule(tmp_path):
    checkpoint_hash = "b" * 64
    rule_path = tmp_path / "combined-validation-rule.json"
    rule_hash = _write_rule(
        rule_path,
        _validation_rule(
            checkpoint_hash,
            frame_count=360,
            case_counts=COMBINED_VALIDATION_CASE_COUNTS,
            holder_rule=True,
        ),
    )

    result = CALIBRATE.validate_frozen_rule(
        rule_path, rule_hash, checkpoint_hash
    )

    assert result["rule_options"] == {
        "measurement_columns_full": (298, 316),
        "measurement_rows_full": (83, 427),
        "min_liquid_pixels": 1,
        "min_blocked_pixels": 1,
        "holder_band_rows_full": (217, 275),
        "surface_oval_height_rows": 17,
        "visibility_columns_full": (279, 340),
        "min_visibility_liquid_pixels": 3,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: artifact["inputs"].__setitem__(
            "batch_summary_sha256", "a" * 64
        ),
        lambda artifact: artifact["inputs"].__setitem__(
            "exception_review_sha256", "a" * 64
        ),
        lambda artifact: artifact["dataset"]["preflight"].__setitem__(
            "samples_checked", 359
        ),
        lambda artifact: artifact["dataset"]["preflight"].__setitem__(
            "files_checked", 719
        ),
        lambda artifact: (
            artifact["dataset"]["preflight"].__setitem__(
                "ordered_hash_ledger_sha256", "f" * 64
            ),
            artifact["rule_config"]["measurement_row_derivation"].__setitem__(
                "ordered_hash_ledger_sha256", "f" * 64
            ),
        ),
    ],
)
def test_combined_validation_rule_rejects_unapproved_provenance(
    tmp_path, mutation
):
    checkpoint_hash = "b" * 64
    artifact = _validation_rule(
        checkpoint_hash,
        frame_count=360,
        case_counts=COMBINED_VALIDATION_CASE_COUNTS,
        holder_rule=True,
    )
    mutation(artifact)
    rule_path = tmp_path / "combined-validation-rule.json"
    rule_hash = _write_rule(rule_path, artifact)

    with pytest.raises(ValueError, match="approved validation provenance"):
        CALIBRATE.validate_frozen_rule(
            rule_path, rule_hash, checkpoint_hash
        )


def test_holder_rule_options_reach_calibration_surface_prediction(
    tmp_path, monkeypatch
):
    checkpoint_hash = "b" * 64
    rule_path = tmp_path / "holder-validation-rule.json"
    rule_hash = _write_rule(
        rule_path,
        _validation_rule(checkpoint_hash, holder_rule=True),
    )
    frozen_rule = CALIBRATE.validate_frozen_rule(
        rule_path, rule_hash, checkpoint_hash
    )
    observed_options = []

    monkeypatch.setattr(
        CALIBRATE,
        "_load_rgb",
        lambda _frame: CALIBRATE.np.zeros(
            (416, 112, 3), dtype=CALIBRATE.np.uint8
        ),
    )

    def capture_prediction(_mask, **options):
        observed_options.append(options)
        return {
            "status": "visible",
            "liquid_top_row": 329.0,
            "range_top_row": 329.0,
            "range_bottom_row": 329.0,
        }

    monkeypatch.setattr(
        CALIBRATE.surface_eval, "surface_prediction", capture_prediction
    )

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def float(self):
            return self

        def div_(self, _divisor):
            return self

        def to(self, _device):
            return self

        def argmax(self, *, dim):
            assert dim == 1
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, _error_type, _error, _traceback):
            return False

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return InferenceMode()

        @staticmethod
        def from_numpy(value):
            return FakeTensor(value)

    class FakeModel:
        def __call__(self, _tensor):
            return FakeTensor(
                CALIBRATE.np.zeros((1, 416, 112), dtype=CALIBRATE.np.uint8)
            )

    indexed = {
        "vials": [
            {
                "vial": 1,
                "volume_ml": 30.0,
                "frames": [
                    {
                        "frame": 816,
                        "path": "vial-01/frame-000816.png",
                        "sha256": "a" * 64,
                    }
                ],
            }
        ]
    }
    records = CALIBRATE.infer_calibration_frames(
        FakeTorch,
        FakeModel(),
        "cpu",
        indexed,
        frozen_rule["rule_options"],
        batch_size=1,
    )

    assert records[0]["liquid_top_row"] == 329.0
    assert observed_options == [
        {
            "measurement_columns_full": (298, 316),
            "measurement_rows_full": (83, 427),
            "min_liquid_pixels": 1,
            "min_blocked_pixels": 1,
            "holder_band_rows_full": (217, 275),
            "surface_oval_height_rows": 17,
            "visibility_columns_full": (279, 340),
            "min_visibility_liquid_pixels": 3,
        }
    ]


def test_validation_rule_rejects_wrong_expected_artifact_hash(tmp_path):
    checkpoint_hash = "b" * 64
    rule_path = tmp_path / "validation-rule.json"
    _write_rule(rule_path, _validation_rule(checkpoint_hash))

    with pytest.raises(ValueError, match="frozen hash"):
        CALIBRATE.validate_frozen_rule(
            rule_path, "a" * 64, checkpoint_hash
        )


def test_validation_rule_rejects_handwritten_partial_artifact(tmp_path):
    checkpoint_hash = "b" * 64
    rule_path = tmp_path / "validation-rule.json"
    rule_hash = _write_rule(
        rule_path,
        {
            "status": "complete",
            "inputs": {"checkpoint_sha256": checkpoint_hash},
            "rule_config": _validation_rule(checkpoint_hash)["rule_config"],
        },
    )

    with pytest.raises(ValueError, match="complete leak-free"):
        CALIBRATE.validate_frozen_rule(rule_path, rule_hash, checkpoint_hash)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda artifact: artifact["dataset"].__setitem__(
                "test_images_loaded", 1
            ),
            "complete leak-free",
        ),
        (
            lambda artifact: artifact["inputs"].pop("checkpoint_sha256"),
            "checkpoint hash binding",
        ),
        (
            lambda artifact: artifact["aggregate"][
                "by_level_bin_10ml"
            ].pop("80-90"),
            "level-bin aggregates",
        ),
        (
            lambda artifact: artifact["aggregate"]["by_case"]["hidden"].pop(
                "true_in_range_rate"
            ),
            "hidden metrics are incomplete",
        ),
        (
            lambda artifact: artifact["rule_config"].pop(
                "measurement_row_derivation"
            ),
            "synthetic measurement-row derivation",
        ),
        (
            lambda artifact: artifact["rule_config"][
                "measurement_row_derivation"
            ].__setitem__("measurement_rows_full_half_open", [82, 427]),
            "complete synthetic validation derivation",
        ),
        (
            lambda artifact: artifact["rule_config"][
                "measurement_row_derivation"
            ].__setitem__("ordered_hash_ledger_sha256", "not-a-hash"),
            "validation ledger hash",
        ),
        (
            lambda artifact: (
                artifact["rule_config"].__setitem__(
                    "measurement_rows_full_half_open", [84, 427]
                ),
                artifact["rule_config"]["measurement_row_derivation"].__setitem__(
                    "measurement_rows_full_half_open", [84, 427]
                ),
            ),
            "approved synthetic validation interval",
        ),
        (
            lambda artifact: artifact["dataset"]["preflight"].__setitem__(
                "ordered_hash_ledger_sha256", "e" * 64
            ),
            "does not match the validation preflight ledger",
        ),
    ],
)
def test_validation_rule_rejects_incomplete_provenance(
    tmp_path, mutation, message
):
    checkpoint_hash = "b" * 64
    artifact = _validation_rule(checkpoint_hash)
    mutation(artifact)
    rule_path = tmp_path / "validation-rule.json"
    rule_hash = _write_rule(rule_path, artifact)

    with pytest.raises(ValueError, match=message):
        CALIBRATE.validate_frozen_rule(rule_path, rule_hash, checkpoint_hash)
