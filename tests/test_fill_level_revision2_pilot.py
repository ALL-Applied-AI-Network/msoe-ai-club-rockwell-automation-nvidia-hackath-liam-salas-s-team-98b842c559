"""CPU-only contracts for the Step 8 Revision-2 pilot analysis."""

from __future__ import annotations

import colorsys
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluidmix import fill_level_revision2_pilot as pilot


SCRIPT = ROOT / "scripts" / "summarize_fill_level_revision2_pilot.py"
SPEC = importlib.util.spec_from_file_location(
    "summarize_fill_level_revision2_pilot", SCRIPT
)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def _write_png(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return {
        "path": path.as_posix(),
        "sha256": pilot.file_sha256(path),
    }


def _stats(low=0.0, high=1.0):
    return {
        "min": low,
        "max": high,
        "mean": (low + high) / 2.0,
        "median": (low + high) / 2.0,
        "p05": low,
        "p95": high,
    }


def _real_appearance():
    broad = _stats(0.0, 1.0)
    return {
        "schema_version": 1,
        "purpose": "synthetic unit fixture",
        "evidence_label": "measured_on_real_calibration_vials_1_and_13_only",
        "training_policy": "appearance measurements only",
        "other_real_frames_opened": 0,
        "provenance": {
            "frame_count": 98,
            "vial_counts": {"1": 28, "13": 70},
            "allowed_vials": [1, 13],
            "index_sha256": "f" * 64,
        },
        "liquids": {
            "vial_1_30ml": {
                "per_frame_value": {**broad, "mean": 0.155038},
            },
            "vial_13_80ml": {
                "per_frame_value": {**broad, "mean": 0.087990},
            },
        },
        "brady_label": {
            "bbox_xyxy_half_open": [285, 168, 341, 240],
            "width_px": 56,
            "height_px": 72,
            "center_xy": [313, 204],
            "grayscale_p10": broad,
            "grayscale_p25": broad,
            "grayscale_p50": broad,
            "grayscale_p75": broad,
            "grayscale_p90": broad,
            "dark_print_fraction_proxy": broad,
            "paper_fraction_proxy": broad,
            "median_rgb": {"r": broad, "g": broad, "b": broad},
        },
        "global_look": {
            "scene_without_timestamp": {
                "rows_half_open": [0, 40],
                "grayscale_p05": broad,
                "grayscale_p50": broad,
                "grayscale_p95": broad,
                "robust_contrast_p95_minus_p05": broad,
                "grayscale_standard_deviation": broad,
            },
            "backdrop": {
                "roi_xyxy_half_open": [2, 5, 20, 25],
                "median_luma": broad,
                "luma_p05": broad,
                "luma_p95": broad,
                "robust_contrast_p95_minus_p05": broad,
                "median_saturation": broad,
                "median_rgb": {"r": broad, "g": broad, "b": broad},
                "tint_r_minus_g": _stats(-1.0, 1.0),
                "tint_b_minus_g": _stats(-1.0, 1.0),
            },
            "blur_proxies": {
                "laplacian_variance_8bit": _stats(0.0, 100000.0),
                "tenengrad_normalized": _stats(0.0, 10.0),
            },
            "noise_proxy": {"normalized_sigma": broad},
            "haze_proxy": {
                "backdrop_luma_floor_p05": broad,
                "backdrop_local_robust_contrast": broad,
            },
        },
    }


def _expanded_real_appearance():
    value = _real_appearance()
    value.update(
        {
            "schema_version": 2,
            "evidence_label": (
                "measured_real_appearance_without_real_model_training"
            ),
            "additional_video5_frames_opened": 0,
            "other_real_frames_opened_definition": (
                "legacy video-5-only counter: no video-5 calibration or real-check "
                "frame outside the approved closed 98-frame vial-1/vial-13 set was "
                "opened; videos-1--4 appearance-reference counts are recorded "
                "separately"
            ),
            "calibration_top_surface_and_foam": {
                "surface_oval_height_reference_px": [17, 24],
                "vial_1_30ml": {"fixture": True},
                "vial_13_80ml": {"fixture": True},
            },
            "external_reference_sample": {
                "provenance": {
                    "video5_paths_resolved_hashed_or_decoded": 0,
                    "video5_vial_colors_csv_opened": 0,
                    "candidate_pair_count": 68,
                    "candidate_image_count_hashed_before_first_decode": 136,
                    "vial_candidate_frames_decoded": 68,
                    "tank_selected_frames_decoded": 34,
                    "selected_pair_count": 34,
                    "allowed_video_ids": [1, 2, 3, 4],
                    "forbidden_video_ids": [5],
                    "vial_candidate_allowlist_sha256": "a" * 64,
                    "tank_candidate_allowlist_sha256": "b" * 64,
                    "candidate_pairs": [{"index": index} for index in range(68)],
                    "selected_pairs": [{"index": index} for index in range(34)],
                },
                "vial_fill": {
                    "selected_nominal_cycle_count": 34,
                    "selection_records": [
                        {"index": index} for index in range(34)
                    ],
                    "settled_proxy_pass_count": 31,
                    "liquid": {"colored_or_dark_settled_count": 10},
                    "labels": {},
                    "top_surface_and_foam": {},
                    "global_look": {},
                    "settled_proxy": {},
                },
                "tank_level_dye_colors": {
                    "tanks": {
                        f"tank_{index}": {"fixture": True}
                        for index in range(1, 5)
                    }
                },
            },
            "renderer_guidance": {
                "liquid": {},
                "labels": {},
                "dark_liquid_top_surface": {},
            },
        }
    )
    return value


def _write_pilot(tmp_path, *, low_count=7, saturated_count=10):
    root = tmp_path / "pilot"
    root.mkdir(parents=True)
    real_path = tmp_path / "real-appearance.json"
    real_path.write_text(
        json.dumps(_real_appearance(), sort_keys=True) + "\n", encoding="utf-8"
    )
    low_body_values = [10, 14, 19, 24, 29, 34, 39]
    records = []
    for index, (seed, level_ml) in enumerate(pilot.EXPECTED_PILOT_GRID):
        y, x = np.mgrid[:480, :640]
        rgb = np.empty((480, 640, 3), dtype=np.uint8)
        rgb[..., 0] = 80 + (x % 25)
        rgb[..., 1] = 85 + (y % 25)
        rgb[..., 2] = 82 + ((x + y) % 25)
        hue_family = (
            "olive",
            "dark_green",
            "brown",
            "blue",
            "near_black",
        )[index % 5]
        material_hue = {
            "olive": 0.21,
            "dark_green": 0.40,
            "brown": 0.08,
            "blue": 0.60,
            "near_black": 0.02,
        }[hue_family]
        mask = np.full((480, 640), 255, dtype=np.uint8)
        mask[8:46, 18:46] = 0
        mask[8:12, 18:22] = 3
        if index:
            mask[20:44, 20:44] = 1
            if index <= low_count:
                body = low_body_values[index - 1]
            else:
                body = 90 + index
            body_saturation = 0.6 if index <= saturated_count else 0.1
            body_rgb = np.rint(
                np.asarray(
                    colorsys.hsv_to_rgb(
                        material_hue, body_saturation, body / 255.0
                    )
                )
                * 255.0
            ).astype(np.uint8)
            rgb[20:44, 20:44] = body_rgb
            rgb[20:22, 20:44] = [80, 80, 80]
        foam = {
            "enabled": index in (1, 2, 3),
            "tint_rgb": [0.30, 0.34, 0.31],
            "opacity": 0.7,
            "bumpiness": 0.5,
            "thickness_mm": 1.0 if index in (1, 2, 3) else 0.0,
            "visible_in_camera": index in (1, 2, 3),
            "authored_small_bubble_count": 0,
        }
        if index in (1, 2, 3):
            mask[16:20, 22:42] = 2
            mask[12:14, 28:30] = 2
            rgb[16:20, 22:42] = [75, 85, 76]
            rgb[12:14, 28:30] = [100, 105, 101]
            foam["authored_small_bubble_count"] = index + 2
        secondary_sticker = index in (3, 4)
        if secondary_sticker:
            sticker_x = 100 if index == 3 else 200
            mask[100:130, sticker_x:sticker_x + 50] = 3
            rgb[100:130, sticker_x:sticker_x + 50] = [210, 205, 195]
            rgb[104:127:4, sticker_x + 5:sticker_x + 45] = [20, 20, 20]
        rgb_path = root / "rgb" / f"{index:05d}.png"
        mask_path = root / "mask" / f"{index:05d}.png"
        _write_png(rgb_path, rgb)
        _write_png(mask_path, mask)
        outputs = {
            "rgb": {
                "path": rgb_path.relative_to(root).as_posix(),
                "sha256": pilot.file_sha256(rgb_path),
            },
            "mask": {
                "path": mask_path.relative_to(root).as_posix(),
                "sha256": pilot.file_sha256(mask_path),
            },
        }
        label_stats = {
            "sticker_bbox_xyxy": (
                [sticker_x, 100, sticker_x + 50, 130]
                if secondary_sticker
                else None
            ),
            "class_pixels": {
                name: int(np.count_nonzero(mask == identifier))
                for name, identifier in (
                    ("empty", 0),
                    ("liquid", 1),
                    ("foam", 2),
                    ("blocked", 3),
                    ("ignore", 255),
                )
            },
            "vial_pixels": int(np.count_nonzero(mask != 255)),
            "vial_bbox_xyxy": [18, 8, 46, 46],
            "sticker_blocked_pixels": 1500 if secondary_sticker else 0,
        }
        if index == 0:
            sticker_mask = np.zeros((480, 640), dtype=np.uint8)
            sticker_mask[10:25, 45:60] = 1
            sticker_mask[12:23:3, 48:57] = 2
            rgb[sticker_mask == 1] = [210, 205, 195]
            rgb[sticker_mask == 2] = [20, 20, 20]
            _write_png(rgb_path, rgb)
            outputs["rgb"]["sha256"] = pilot.file_sha256(rgb_path)
            sticker_path = root / "sticker" / "00000.png"
            _write_png(sticker_path, sticker_mask)
            outputs["sticker_mask"] = {
                "path": sticker_path.relative_to(root).as_posix(),
                "sha256": pilot.file_sha256(sticker_path),
            }
            label_stats["sticker_bbox_xyxy"] = [45, 10, 60, 25]
            label_stats["sticker_blocked_pixels"] = int(
                np.count_nonzero(sticker_mask)
            )
        records.append(
            {
                "schema_version": 1,
                "seed": seed,
                "role": "pilot",
                "variation_profile": "real-look-v2",
                "level_ml": level_ml,
                "liquid": {
                    "hsv": [
                        material_hue,
                        body_saturation,
                        max(0.04, body / 255.0),
                    ]
                    if index
                    else [0.0, 0.0, 0.04],
                    "hue_family": hue_family,
                    "opacity": 0.8,
                },
                "foam": foam,
                "sticker_pose": {
                    "present": index in (0, 3, 4),
                    "visible_in_camera": index in (0, 3, 4),
                },
                "surface": {"rendered_oval_height_px": 2},
                "outputs": outputs,
                "label_stats": label_stats,
            }
        )
    manifest_path = root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "frame_count": 20,
        "passed": True,
        "source_unchanged": True,
        "integrity_passed": True,
        "real_look_requested": True,
        "manifest_sha256": pilot.file_sha256(manifest_path),
        "renderer_script_sha256": "1" * 64,
        "variation_module_sha256": "2" * 64,
        "specs_sha256": "3" * 64,
        "session_layer_sha256": "4" * 64,
        "active_vial_glass_material_params": {
            "diffuse_rgb": [0.92, 0.96, 0.98],
            "opacity": 0.14,
            "roughness": 0.08,
            "ior": 1.46,
            "specular_rgb": [0.034965959415691715] * 3,
            "use_specular_workflow": 1,
            "metallic": 0.0,
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root, real_path


def _write_allowed_real_root(tmp_path):
    root = tmp_path / "allowed-real"
    frames = []
    selected = {830, 19735}
    for vial, first, last, level, prefix in (
        (1, 816, 843, 30.0, "vial-01-030ml"),
        (13, 19700, 19769, 80.0, "vial-13-080ml"),
    ):
        for frame_index in range(first, last + 1):
            relative = f"{prefix}/frame-{frame_index:06d}.png"
            sha256 = "a" * 64
            if frame_index in selected:
                y, x = np.mgrid[:480, :640]
                image = np.empty((480, 640, 3), dtype=np.uint8)
                image[..., 0] = (x // 4 + vial) % 256
                image[..., 1] = (y // 3 + vial) % 256
                image[..., 2] = ((x + y) // 6 + vial) % 256
                path = root / relative
                _write_png(path, image)
                sha256 = pilot.file_sha256(path)
            frames.append(
                {
                    "frame_index": frame_index,
                    "level_ml": level,
                    "path": relative,
                    "role": "calibration",
                    "sha256": sha256,
                    "source_window_role": "real_validation",
                    "vial": vial,
                }
            )
    index = {"schema_version": 1, "frame_count": 98, "frames": frames}
    (root / "index.json").write_text(
        json.dumps(index, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _read_manifest(root):
    return [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
    ]


def _rewrite_manifest_and_bind_summary(root, records):
    manifest_path = root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["manifest_sha256"] = pilot.file_sha256(manifest_path)
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )


def _replace_mask(root, records, index, transform):
    descriptor = records[index]["outputs"]["mask"]
    mask_path = root / descriptor["path"]
    with Image.open(mask_path) as image:
        mask = np.asarray(image).copy()
    mask = transform(mask)
    Image.fromarray(mask.astype(np.uint8)).save(mask_path)
    descriptor["sha256"] = pilot.file_sha256(mask_path)
    records[index]["label_stats"]["class_pixels"] = {
        name: int(np.count_nonzero(mask == identifier))
        for name, identifier in (
            ("empty", 0),
            ("liquid", 1),
            ("foam", 2),
            ("blocked", 3),
            ("ignore", 255),
        )
    }
    records[index]["label_stats"]["vial_pixels"] = int(
        np.count_nonzero(mask != 255)
    )


def test_liquid_hsv_metrics_use_weighted_circular_hue_and_mark_grey_unavailable():
    rgb = np.asarray(
        [[
            np.rint(np.asarray(colorsys.hsv_to_rgb(0.99, 1.0, 0.5)) * 255),
            np.rint(np.asarray(colorsys.hsv_to_rgb(0.01, 1.0, 0.5)) * 255),
            [64, 64, 64],
        ]],
        dtype=np.uint8,
    )
    metrics = pilot.liquid_hsv_metrics(rgb, np.ones((1, 3), dtype=bool))
    assert pilot._circular_hue_distance(
        metrics["mean_hue_circular_saturation_weighted_0_to_1"], 0.0
    ) < 0.01
    assert metrics["mean_saturation"] == pytest.approx(2.0 / 3.0, abs=0.01)
    comparison = pilot._material_rendered_hsv_comparison(
        metrics, [0.0, 0.8, 0.5]
    )
    assert comparison["circular_hue_error"] < 0.01
    assert comparison["saturation_retention_ratio"] == pytest.approx(
        metrics["mean_saturation"] / 0.8
    )

    grey = np.full((2, 2, 3), 50, dtype=np.uint8)
    grey_metrics = pilot.liquid_hsv_metrics(
        grey, np.ones((2, 2), dtype=bool)
    )
    assert grey_metrics["mean_hue_circular_saturation_weighted_0_to_1"] is None
    assert grey_metrics["mean_saturation"] == 0.0


def test_holder_and_glass_proxies_are_deterministic_and_explicit():
    rgb = np.full((8, 8, 3), 51, dtype=np.uint8)
    mask = np.full((8, 8), 3, dtype=np.uint8)
    rgb[0:2, 0:2] = 255
    holder = pilot.holder_proxy_metrics(rgb, mask, [0, 0, 2, 2])
    assert holder["proxy_name"] == "decoded_blocked_non_sticker"
    assert holder["sticker_bbox_excluded_blocked_pixel_count"] == 4
    assert holder["pixel_count"] == 60
    assert holder["mean_value"] == pytest.approx(0.2)
    assert holder["median_value"] == pytest.approx(0.2)

    boundary_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    boundary_mask = np.full((8, 8), 255, dtype=np.uint8)
    boundary_mask[1:7, 1:7] = 0
    boundary_rgb[1, 1:7] = 255
    boundary_rgb[6, 1:7] = 255
    boundary_rgb[1:7, 1] = 255
    boundary_rgb[1:7, 6] = 255
    glass = pilot.glass_boundary_proxy_metrics(
        boundary_rgb, boundary_mask, [1, 1, 7, 7]
    )
    assert glass["proxy_name"] == "semantic_vial_one_pixel_boundary"
    assert glass["pixel_count"] == 20
    assert glass["value_p95"] == 1.0
    assert glass["highlight_fraction_v_ge_0_80"] == 1.0
    assert glass["gradient_magnitude_mean"] > 0.0


def test_backdrop_target_and_darkest_tail_summary_have_exact_traceable_math():
    target = pilot.backdrop_neutral_target_comparison(
        {"backdrop_median_luma": 0.66, "backdrop_median_saturation": 0.04}
    )
    assert target["absolute_luma_error"] == 0.0
    assert target["absolute_saturation_error"] == 0.0

    values = [0.055, 0.080] + [0.12 + index * 0.01 for index in range(17)]
    frames = [
        {"index": index, "seed": 900 + index, "liquid_mean_v": value}
        for index, value in enumerate(values)
    ]
    tail = pilot._darkest_tail_summary(frames)
    assert tail["target_count_nearest_integer"] == 2
    assert tail["observed_count_in_target_range"] == 2
    assert tail["darkest_target_frames"] == [
        {"seed": 900, "mean_value": 0.055},
        {"seed": 901, "mean_value": 0.080},
    ]
    assert tail["gates"]["passed"]


def test_montage_selection_tie_breaks_and_metadata_are_deterministic():
    def frame(index, seed, value, family, material_v, surface_v=0.2):
        return {
            "index": index,
            "seed": seed,
            "liquid_mean_v": value,
            "material": {"hsv": [0.2, 0.5, material_v], "hue_family": family},
            "surface_bbox": [1, 1, 3, 3] if value is not None else None,
            "surface": {"mean_value": surface_v} if value is not None else None,
            "foam_bbox": None,
            "foam": {"visible_in_decoded_mask": False},
        }

    frames = [
        frame(0, 100, None, "olive", 0.0),
        frame(1, 101, 0.10, "olive", 0.10, 0.8),
        frame(2, 102, 0.40, "olive", 0.30, 0.8),
        frame(3, 103, 0.20, "blue", 0.20, 0.3),
    ]
    selected, metadata = pilot._select_scene_montage_frames(frames)
    assert [item["seed"] for item in selected] == [100, 101, 102, 103]
    olive = next(
        item
        for item in metadata
        if "mid_value_hue_family:olive" in item["reasons"]
    )
    assert olive["seed"] == 101
    assert olive["hue_family_material_value_median"] == pytest.approx(0.20)
    assert "earliest manifest index" in olive["hue_family_selection"]

    surface_frames = [
        frame(5, 205, 0.1, "olive", 0.1, 0.8),
        frame(4, 204, 0.1, "olive", 0.1, 0.8),
    ]
    surfaces, outlier = pilot._select_surface_montage_frames(surface_frames, [])
    assert [item["seed"] for item in surfaces] == [204]
    assert outlier["seed"] == 204

    foam_frames = []
    for index, seed in enumerate((202, 201)):
        item = frame(index, seed, 0.1, "olive", 0.1)
        item["foam_bbox"] = [1, 1, 3, 3]
        item["foam"] = {
            "visible_in_decoded_mask": True,
            "small_mask_component_count_2_to_64_px_proxy": 2,
            "authored_small_bubble_count": 4,
            "pixel_count": 20,
        }
        foam_frames.append(item)
    foam_pick, foam_metadata = pilot._select_foam_montage_frame(foam_frames)
    assert foam_pick["seed"] == 202
    assert foam_metadata["seed"] == 202
    assert "earliest manifest index" in foam_metadata["selection"]


def test_complete_report_gates_darkness_and_labels_evidence(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)

    report = pilot.summarize_pilot(
        pilot_root,
        real_path,
        expected_specs_sha256="3" * 64,
    )

    assert report["status"] == "complete"
    assert report["population"] == {
        "manifest_records": 20,
        "liquid_eligible_records": 19,
        "no_visible_liquid_records": 1,
        "variation_profile": "real-look-v2",
    }
    assert report["inputs"]["specs_match_expected"] is True
    checker = report["independent_capture_checker"]
    assert checker["status"] == "passed"
    assert checker["passed"] is True
    assert checker["record_count"] == 20
    assert checker["checker_sha256"] == pilot.file_sha256(
        ROOT / "scripts" / "check_fill_level_capture.py"
    )
    assert checker["report"]["passed"] is True
    assert len(checker["report"]["records"]) == 20
    assert report["pilot_identity"]["decoded_liquid_ineligible_seeds"] == [
        28001
    ]
    assert report["pilot_identity"]["seed_level_grid"][3] == {
        "seed": 28025,
        "level_ml": 39.5,
    }
    rendered = report["rendered_liquid_v"]
    assert rendered["low_count"] == 7
    assert rendered["gates"] == {
        "eligible_count_is_19": True,
        "minimum_covers_real_0_088": True,
        "maximum_covers_real_0_155": True,
        "passed": True,
    }
    assert rendered["legacy_observations"] == {
        "review_source": "2026-11-01 ~11:22 pilot r2-v4 look review",
        "low_count_is_6_to_8": True,
        "waived_for_v5": True,
    }
    rendered_hsv = report["rendered_liquid_hsv"]
    assert rendered_hsv["target_saturation_range_inclusive"] == [0.30, 0.85]
    assert rendered_hsv["target_saturated_count_in_19"] == [9, 10]
    assert rendered_hsv["observed_saturated_count"] == 10
    assert rendered_hsv["gates"] == {
        "eligible_count_is_19": True,
        "saturated_count_is_9_or_10": True,
        "passed": True,
    }
    seed_28002 = next(
        frame for frame in rendered_hsv["per_frame"] if frame["seed"] == 28002
    )
    assert seed_28002["rendered_hue_0_to_1"] == pytest.approx(0.40, abs=0.02)
    assert seed_28002["rendered_mean_saturation"] > 0.45
    assert seed_28002["circular_hue_error"] < 0.02
    assert seed_28002["saturation_retention_ratio"] > 0.75
    assert set(rendered_hsv["manifested_hue_families"]) == {
        "blue",
        "brown",
        "dark_green",
        "near_black",
        "olive",
    }
    tail = report["darkest_liquid_tail"]
    assert tail["target_count_nearest_integer"] == 2
    assert tail["observed_count_in_target_range"] == 2
    assert [frame["seed"] for frame in tail["darkest_target_frames"]] == [
        28002,
        28003,
    ]
    assert tail["gates"]["passed"]
    holder = report["holder_appearance"]
    assert holder["available_frame_count"] == 20
    assert holder["gates"]["passed"]
    sticker_holder = next(
        frame for frame in holder["per_frame"] if frame["seed"] == 28025
    )
    assert sticker_holder["metrics"][
        "sticker_bbox_excluded_blocked_pixel_count"
    ] == 1500
    assert sticker_holder["metrics"]["pixel_count"] == 16
    assert report["glass_edge_highlight_proxies"]["available_frame_count"] == 20
    neutral = report["scene_appearance"]["neutral_backdrop_target_comparison"]
    assert neutral["review_target"] == {
        "median_luma_about": 0.66,
        "median_saturation_about": 0.04,
    }
    assert neutral["automated_gate"] is None
    assert report["material_inputs"]["per_frame"][0][
        "variation_profile"
    ] == "real-look-v2"
    assert [
        frame["seed"] for frame in report["sticker_appearance"]["per_frame"]
    ] == [28001, 28025, 28005]
    assert report["liquid_top_surface"]["surface_records"] == 19
    assert report["liquid_top_surface"]["dark_liquid_surface_records"] == 7
    assert report["liquid_top_surface"]["automated_gate"] is None
    assert report["foam_appearance"]["frames_with_foam_contract"] == 20
    assert report["foam_appearance"]["visible_foam_records"] == 3
    assert report["foam_appearance"][
        "positive_authored_small_bubble_count_range"
    ] == [3, 5]
    visible_foam = next(
        frame
        for frame in report["foam_appearance"]["per_frame"]
        if frame["metrics"]["visible_in_decoded_mask"]
    )
    assert visible_foam["metrics"][
        "small_mask_component_count_2_to_64_px_proxy"
    ] == 1
    assert report["scene_appearance"]["comparisons"][
        "scene_robust_contrast"
    ]["available"]
    assert report["automated_gates"]["darkest_liquid_tail_passed"]
    assert report["automated_gates"]["rendered_liquid_saturation_passed"]
    assert report["automated_gates"]["extreme_liquid_holder_relation_passed"]


def test_waived_legacy_low_count_is_reporting_only(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path, low_count=5)

    report = pilot.summarize_pilot(pilot_root, real_path)

    assert report["status"] == "complete"
    assert report["rendered_liquid_v"]["low_count"] == 5
    assert not report["rendered_liquid_v"]["legacy_observations"][
        "low_count_is_6_to_8"
    ]
    assert report["automated_gates"]["integrity_passed"]
    assert report["automated_gates"]["passed"]


@pytest.mark.parametrize("saturated_count", [8, 11])
def test_about_half_saturation_gate_rejects_8_or_11_of_19(
    tmp_path, saturated_count
):
    pilot_root, real_path = _write_pilot(
        tmp_path, saturated_count=saturated_count
    )

    report = pilot.summarize_pilot(pilot_root, real_path)

    rendered_hsv = report["rendered_liquid_hsv"]
    assert rendered_hsv["observed_saturated_count"] == saturated_count
    assert not rendered_hsv["gates"]["saturated_count_is_9_or_10"]
    assert report["status"] == "failed_automated_gate"
    assert not report["automated_gates"]["rendered_liquid_saturation_passed"]
    assert not report["automated_gates"]["passed"]


@pytest.mark.parametrize(
    ("index", "field", "value"),
    [(0, "seed", 28000), (3, "level_ml", 40.0)],
)
def test_exact_planned_seed_level_identity_is_mandatory(
    tmp_path, index, field, value
):
    pilot_root, real_path = _write_pilot(tmp_path)
    records = _read_manifest(pilot_root)
    records[index][field] = value
    _rewrite_manifest_and_bind_summary(pilot_root, records)
    with pytest.raises(ValueError, match="exact planned pilot seed/level grid"):
        pilot.summarize_pilot(pilot_root, real_path)


def test_positive_frame_without_decoded_liquid_is_rejected(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)
    records = _read_manifest(pilot_root)

    def remove_liquid(mask):
        mask[mask == 1] = 0
        return mask

    _replace_mask(pilot_root, records, 1, remove_liquid)
    _rewrite_manifest_and_bind_summary(pilot_root, records)
    with pytest.raises(ValueError, match="positive-level planned seed 28002"):
        pilot.summarize_pilot(pilot_root, real_path)


def test_count_19_cannot_hide_wrong_eligibility_association(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)
    records = _read_manifest(pilot_root)

    def add_liquid(mask):
        mask[20:24, 20:24] = 1
        return mask

    def remove_liquid(mask):
        mask[mask == 1] = 0
        return mask

    _replace_mask(pilot_root, records, 0, add_liquid)
    _replace_mask(pilot_root, records, 1, remove_liquid)
    _rewrite_manifest_and_bind_summary(pilot_root, records)
    with pytest.raises(ValueError, match="zero-level seed 28001"):
        pilot.summarize_pilot(pilot_root, real_path)


def test_independent_capture_checker_can_veto_checker_invalid_capture(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)
    records = _read_manifest(pilot_root)

    def break_ignored_boundary(mask):
        mask[0, 0] = 0
        return mask

    _replace_mask(pilot_root, records, 1, break_ignored_boundary)
    _rewrite_manifest_and_bind_summary(pilot_root, records)
    with pytest.raises(
        ValueError,
        match="independent capture checker did not pass.*mask boundary",
    ):
        pilot.summarize_pilot(pilot_root, real_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("passed", False, "did not pass"),
        ("source_unchanged", False, "source changed"),
        ("integrity_passed", False, "integrity did not pass"),
        ("real_look_requested", False, "did not request the real-look profile"),
    ],
)
def test_bad_render_summary_flags_are_rejected(tmp_path, field, value, message):
    pilot_root, real_path = _write_pilot(tmp_path)
    summary_path = pilot_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        pilot.summarize_pilot(pilot_root, real_path)


def test_mirror_glass_summary_is_rejected_by_dielectric_f0_guard(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)
    summary_path = pilot_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["active_vial_glass_material_params"]["specular_rgb"] = [0.96] * 3
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="dielectric F0 guard"):
        pilot.summarize_pilot(pilot_root, real_path)


def test_expected_specs_hash_is_an_optional_strict_binding(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)
    with pytest.raises(ValueError, match="intended pilot"):
        pilot.summarize_pilot(
            pilot_root,
            real_path,
            expected_specs_sha256="9" * 64,
        )


def test_invalid_authored_bubbles_and_path_escape_are_rejected(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path / "bubble")
    records = _read_manifest(pilot_root)
    records[1]["foam"]["authored_small_bubble_count"] = 2
    _rewrite_manifest_and_bind_summary(pilot_root, records)
    with pytest.raises(ValueError, match="needs 3..10 authored bubbles"):
        pilot.summarize_pilot(pilot_root, real_path)

    pilot_root, real_path = _write_pilot(tmp_path / "escape")
    records = _read_manifest(pilot_root)
    records[0]["outputs"]["rgb"]["path"] = "../escape.png"
    _rewrite_manifest_and_bind_summary(pilot_root, records)
    with pytest.raises(ValueError, match="escapes its evidence root"):
        pilot.summarize_pilot(pilot_root, real_path)


def test_current_and_expanded_real_appearance_schemas_are_strict():
    current = pilot.validate_real_appearance(_real_appearance())
    expanded = pilot.validate_real_appearance(_expanded_real_appearance())
    assert current["schema_version"] == 1
    assert expanded["schema_version"] == 2

    invalid = _expanded_real_appearance()
    invalid["additional_video5_frames_opened"] = 1
    with pytest.raises(ValueError, match="additional video-5 frames"):
        pilot.validate_real_appearance(invalid)

    invalid = _expanded_real_appearance()
    invalid["other_real_frames_opened_definition"] = "ambiguous"
    with pytest.raises(ValueError, match="legacy video-5 counter"):
        pilot.validate_real_appearance(invalid)

    invalid = _expanded_real_appearance()
    invalid["provenance"]["frame_count"] = 99
    with pytest.raises(ValueError, match="only 98 vial-1/vial-13 frames"):
        pilot.validate_real_appearance(invalid)

    invalid = _expanded_real_appearance()
    invalid["external_reference_sample"]["provenance"][
        "selected_pair_count"
    ] = 33
    with pytest.raises(ValueError, match="selected_pair_count"):
        pilot.validate_real_appearance(invalid)


def test_manifest_profile_summary_binding_and_output_hash_are_strict(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)
    manifest_path = pilot_root / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    records[0]["variation_profile"] = "baseline"
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="appearance profile"):
        pilot.summarize_pilot(pilot_root, real_path)

    pilot_root, real_path = _write_pilot(tmp_path / "second")
    summary_path = pilot_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["manifest_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        pilot.summarize_pilot(pilot_root, real_path)

    pilot_root, real_path = _write_pilot(tmp_path / "third")
    with (pilot_root / "rgb" / "00001.png").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="RGB SHA-256 mismatch"):
        pilot.summarize_pilot(pilot_root, real_path)


def test_montage_opens_only_two_whitelisted_frames_and_adds_surface_zooms(
    tmp_path,
):
    pilot_root, real_path = _write_pilot(tmp_path)
    allowed_root = _write_allowed_real_root(tmp_path)
    real = json.loads(real_path.read_text())
    real["provenance"]["index_sha256"] = pilot.file_sha256(
        allowed_root / "index.json"
    )
    real_path.write_text(json.dumps(real, sort_keys=True) + "\n", encoding="utf-8")
    montage_path = tmp_path / "ignored" / "revision2-montage.png"

    report = pilot.summarize_pilot(
        pilot_root,
        real_path,
        allowed_frames_root=allowed_root,
        montage_out=montage_path,
    )

    assert montage_path.is_file()
    assert report["montage"]["status"] == "created_from_explicit_whitelist"
    assert [frame["frame_index"] for frame in report["montage"]["real_frames"]] == [
        830,
        19735,
    ]
    assert report["montage"]["synthetic_surface_zoom_seeds"]
    scene_selections = report["montage"]["synthetic_scene_selections"]
    family_reasons = {
        reason.split(":", 1)[1]
        for selection in scene_selections
        for reason in selection["reasons"]
        if reason.startswith("mid_value_hue_family:")
    }
    assert family_reasons == {
        "blue",
        "brown",
        "dark_green",
        "near_black",
        "olive",
    }
    assert report["montage"]["synthetic_surface_outlier"]["seed"] == 28002
    assert report["montage"]["synthetic_foam_crop"]["seed"] in {
        28002,
        28003,
        28025,
    }
    assert report["montage"]["real_sticker_crop"] == {
        "frame_index": 830,
        "bbox_xyxy_half_open": [285, 168, 341, 240],
    }
    assert report["montage"]["synthetic_sticker_crop"]["seed"] == 28025
    assert report["montage"]["synthetic_sticker_crop"][
        "bbox_xyxy_half_open"
    ] == [100, 100, 150, 130]
    assert report["montage"]["synthetic_sticker_crop"][
        "sticker_blocked_pixels"
    ] == 1500
    assert report["montage"]["layout"].startswith("frame-830 BRADY")
    with Image.open(montage_path) as montage_image:
        assert montage_image.size[0] == 960
        assert montage_image.size[1] % 268 == 0
        assert montage_image.size[1] >= 1340
    with pytest.raises(FileExistsError, match="refusing to overwrite montage"):
        pilot.summarize_pilot(
            pilot_root,
            real_path,
            allowed_frames_root=allowed_root,
            montage_out=montage_path,
        )

    index_path = allowed_root / "index.json"
    index = json.loads(index_path.read_text())
    selected = next(frame for frame in index["frames"] if frame["frame_index"] == 830)
    selected["sha256"] = "0" * 64
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="allowed real frame hash mismatch"):
        pilot.summarize_pilot(
            pilot_root,
            real_path,
            allowed_frames_root=allowed_root,
            montage_out=tmp_path / "ignored" / "second.png",
        )


def test_synthetic_montage_is_deterministic_and_never_reads_real_frames(
    tmp_path, monkeypatch
):
    pilot_root, real_path = _write_pilot(tmp_path)

    def forbid_real_index(_root):
        raise AssertionError("synthetic montage must not inspect real-frame index")

    monkeypatch.setattr(pilot, "_validate_real_index", forbid_real_index)
    first_path = tmp_path / "assets" / "synthetic-pilot.png"
    second_path = tmp_path / "assets" / "synthetic-pilot-copy.png"
    first = pilot.summarize_pilot(
        pilot_root,
        real_path,
        synthetic_montage_out=first_path,
    )
    second = pilot.summarize_pilot(
        pilot_root,
        real_path,
        synthetic_montage_out=second_path,
    )

    assert first["montage"]["status"] == "not_requested"
    metadata = first["synthetic_montage"]
    assert metadata["status"] == "created_from_synthetic_pilot_only"
    assert metadata["contains_real_frames"] is False
    assert metadata["raw_full_renders_included"] is False
    assert all(source.startswith("synthetic_") for source in metadata["panel_sources"])
    assert metadata["sha256"] == pilot.file_sha256(first_path)
    assert metadata["sha256"] == second["synthetic_montage"]["sha256"]
    assert metadata["layout_dimensions_px"][0] == 720
    with pytest.raises(FileExistsError, match="synthetic montage"):
        pilot.summarize_pilot(
            pilot_root,
            real_path,
            synthetic_montage_out=first_path,
        )


def test_synthetic_montage_path_cannot_alias_mixed_montage(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)
    destination = tmp_path / "same.png"
    with pytest.raises(
        ValueError,
        match="montage_out and synthetic_montage_out must be different paths",
    ):
        pilot.summarize_pilot(
            pilot_root,
            real_path,
            allowed_frames_root=tmp_path / "unused-real-root",
            montage_out=destination,
            synthetic_montage_out=destination,
        )
    assert not destination.exists()


def test_cli_writes_specs_bound_report_then_refuses_overwrite(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)
    report_path = tmp_path / "report.json"
    synthetic_montage_path = tmp_path / "synthetic-montage.png"
    arguments = [
        "--pilot-dir",
        str(pilot_root),
        "--real-appearance",
        str(real_path),
        "--expected-specs-sha256",
        "3" * 64,
        "--synthetic-montage-out",
        str(synthetic_montage_path),
        "--report-out",
        str(report_path),
    ]
    assert CLI.main(arguments) == 0
    report = json.loads(report_path.read_text())
    assert report["inputs"]["specs_match_expected"] is True
    assert synthetic_montage_path.is_file()
    assert report["synthetic_montage"]["contains_real_frames"] is False
    assert report["synthetic_montage"]["sha256"] == pilot.file_sha256(
        synthetic_montage_path
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite evidence"):
        CLI.main(arguments)


def test_cli_rejects_report_and_montage_path_alias_before_creation(tmp_path):
    pilot_root, real_path = _write_pilot(tmp_path)
    destination = tmp_path / "must-not-be-created.png"
    with pytest.raises(
        ValueError, match="report-out and montage-out must be different paths"
    ):
        CLI.main(
            [
                "--pilot-dir",
                str(pilot_root),
                "--real-appearance",
                str(real_path),
                "--allowed-frames-root",
                str(tmp_path / "unused-real-root"),
                "--montage-out",
                str(destination),
                "--report-out",
                str(destination),
            ]
        )
    assert not destination.exists()

    with pytest.raises(ValueError, match="synthetic-montage-out must differ"):
        CLI.main(
            [
                "--pilot-dir",
                str(pilot_root),
                "--real-appearance",
                str(real_path),
                "--synthetic-montage-out",
                str(destination),
                "--report-out",
                str(destination),
            ]
        )
    assert not destination.exists()


def test_writers_refuse_overwrite_and_scene_metrics_are_deterministic(tmp_path):
    destination = tmp_path / "report.json"
    pilot.write_json_no_overwrite(destination, {"schema_version": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pilot.write_json_no_overwrite(destination, {"schema_version": 2})
    assert json.loads(destination.read_text()) == {"schema_version": 1}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        CLI.main(
            [
                "--pilot-dir",
                str(tmp_path / "unused-pilot"),
                "--real-appearance",
                str(tmp_path / "unused-real.json"),
                "--report-out",
                str(destination),
            ]
        )

    image = np.full((48, 64, 3), [80, 90, 85], dtype=np.uint8)
    rois = {
        "scene_rows_half_open": [0, 40],
        "backdrop_xyxy": [2, 5, 20, 25],
    }
    assert pilot.scene_metrics(image, rois) == pilot.scene_metrics(image, rois)
