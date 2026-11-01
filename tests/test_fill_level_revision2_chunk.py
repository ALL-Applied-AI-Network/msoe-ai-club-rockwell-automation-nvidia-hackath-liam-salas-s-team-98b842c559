"""Contracts for the first Revision 2 train-chunk look evidence."""

import colorsys
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix.fill_level_revision2_chunk import (  # noqa: E402
    aggregate_rendered_hsv,
    file_sha256,
    summarize_train_chunk,
    write_montage,
    write_json_no_overwrite,
)


SCRIPT_PATH = ROOT / "scripts" / "summarize_fill_level_revision2_chunk.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "summarize_fill_level_revision2_chunk", SCRIPT_PATH
)
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


def _rgb8(hue, saturation, value):
    return np.asarray(
        [round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, value)],
        dtype=np.uint8,
    )


def _write_train_chunk(root):
    root.mkdir()
    (root / "rgb").mkdir()
    (root / "mask").mkdir()
    records = []
    for index, seed in enumerate(range(103_000, 103_100)):
        if index < 10:
            rendered_color = _rgb8(0.40, 0.50, 0.07)
        elif index < 50:
            rendered_color = _rgb8(0.40, 0.50, 0.15)
        else:
            rendered_color = _rgb8(0.20, 0.10, 0.15)
        rgb = np.full((480, 640, 3), [170, 174, 172], dtype=np.uint8)
        rgb[50:450, 270:370] = [210, 214, 212]
        mask = np.full((480, 640), 255, dtype=np.uint8)
        mask[50:450, 270:370] = 0
        mask[100:120, 280:300] = 3
        if index != 99:
            rgb[260:430, 280:360] = rendered_color
            mask[260:430, 280:360] = 1
        rgb_path = root / "rgb" / f"{index:05d}-train-{seed:07d}.png"
        mask_path = root / "mask" / f"{index:05d}-train-{seed:07d}.png"
        Image.fromarray(rgb, "RGB").save(rgb_path)
        Image.fromarray(mask, "L").save(mask_path)
        material_saturation = 0.50 if seed % 2 == 0 else 0.10
        extreme = seed % 9 == 0
        very_dark = seed % 3 == 0
        identifiers, counts = np.unique(mask, return_counts=True)
        measured = {
            int(identifier): int(count)
            for identifier, count in zip(identifiers, counts)
        }
        records.append(
            {
                "schema_version": 1,
                "frame_index": index,
                "seed": seed,
                "role": "train",
                "level_ml": 0.05 if index == 0 else 0.10 if index == 99 else 55.0,
                "variation_profile": "real-look-v2",
                "foam": {
                    "enabled": False,
                    "visible_in_camera": False,
                    "thickness_mm": 0.0,
                },
                "sticker_pose": {
                    "present": False,
                    "visible_in_camera": False,
                },
                "liquid": {
                    "hsv": [0.40, material_saturation, 0.03 if extreme else 0.10 if very_dark else 0.25],
                    "hue_family": "dark_green" if index % 2 == 0 else "olive",
                    "saturation_band": "saturated" if seed % 2 == 0 else "subdued",
                    "darkness_band": "extreme_dark" if extreme else "very_dark" if very_dark else "dark",
                },
                "label_stats": {
                    "vial_bbox_xyxy": [270, 50, 370, 450],
                    "class_pixels": {
                        "empty": measured.get(0, 0),
                        "liquid": measured.get(1, 0),
                        "foam": measured.get(2, 0),
                        "blocked": measured.get(3, 0),
                        "ignore": measured.get(255, 0),
                    },
                    "vial_pixels": int(np.count_nonzero(mask != 255)),
                    "sticker_blocked_pixels": 0,
                },
                "outputs": {
                    "rgb": {
                        "path": str(rgb_path.relative_to(root)).replace("\\", "/"),
                        "sha256": file_sha256(rgb_path),
                    },
                    "mask": {
                        "path": str(mask_path.relative_to(root)).replace("\\", "/"),
                        "sha256": file_sha256(mask_path),
                    },
                },
            }
        )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    session_layer = root / "session-layer.usda"
    session_layer.write_text("#usda 1.0\n", encoding="utf-8")
    hashes = {
        "renderer_script_sha256": "1" * 64,
        "variation_module_sha256": "2" * 64,
        "specs_sha256": "3" * 64,
        "session_layer_sha256": file_sha256(session_layer),
    }
    summary = {
        "schema_version": 1,
        "passed": True,
        "source_unchanged": True,
        "integrity_passed": True,
        "real_look_requested": True,
        "frame_count": 100,
        "renderer": "PathTracing",
        "subframes": 32,
        "resolution": [640, 480],
        "label_ids": {
            "empty": 0,
            "liquid": 1,
            "foam": 2,
            "blocked": 3,
            "ignore": 255,
        },
        "manifest_sha256": file_sha256(manifest),
        "active_vial_prim": "/Station/Vial",
        "active_vial_glass_material_params": {
            "diffuse_rgb": [0.92, 0.96, 0.98],
            "opacity": 0.14,
            "roughness": 0.08,
            "ior": 1.46,
            "specular_rgb": [0.034965959415691715] * 3,
            "use_specular_workflow": 1,
            "metallic": 0.0,
        },
        "active_vial_glass_material_prim_paths": ["/Station/Vial/Glass"],
        "source_station_tree_hashes_before": {"station.usd": "a" * 64},
        "source_station_tree_hashes_after": {"station.usd": "a" * 64},
        "source_camera_json_sha256_before": "b" * 64,
        "source_camera_json_sha256_after": "b" * 64,
        **hashes,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hashes


def _read_records(root):
    return [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_records(root, records):
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["manifest_sha256"] = file_sha256(manifest)
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_summary(root):
    return json.loads((root / "summary.json").read_text(encoding="utf-8"))


def _write_summary(root, summary):
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )


def _summarize(chunk_dir, hashes, montage):
    return summarize_train_chunk(
        chunk_dir,
        103_000,
        103_099,
        expected_role="train",
        montage_out=montage,
        expected_specs_sha256=hashes["specs_sha256"],
        expected_renderer_sha256=hashes["renderer_script_sha256"],
        expected_variation_sha256=hashes["variation_module_sha256"],
    )


def test_first_train_chunk_summary_verifies_provenance_and_writes_montage(tmp_path):
    chunk_dir = tmp_path / "chunk"
    hashes = _write_train_chunk(chunk_dir)
    montage = tmp_path / "montage.png"
    report = _summarize(chunk_dir, hashes, montage)
    assert report["scope"] == {
        "role": "train",
        "seed_start": 103_000,
        "seed_end": 103_099,
        "frame_count": 100,
        "sealed_test_frames_opened": 0,
        "real_frames_opened": 0,
    }
    assert report["rendered_liquid_hsv"]["eligible_frame_count"] == 99
    assert report["rendered_liquid_hsv"]["saturation_0_30_to_0_85"]["count"] == 50
    assert report["rendered_liquid_hsv"]["value_0_05_to_0_09"]["count"] == 10
    assert report["positive_level_zero_liquid_frames"][0]["seed"] == 103_099
    assert report["positive_level_zero_liquid_frames"][0]["explanation"]
    assert report["independent_capture_checker"]["passed"] is True
    assert report["independent_capture_checker"]["record_count"] == 100
    assert report["montage"]["synthetic_only"] is True
    assert report["montage"]["contains_real_frames"] is False
    assert report["montage"]["contains_test_frames"] is False
    assert len(report["montage"]["selected_seeds"]) == 20
    selection_by_seed = {
        item["seed"]: item for item in report["montage"]["selections"]
    }
    assert "positive_level_zero_liquid_glass_focus" in selection_by_seed[103_099][
        "reasons"
    ]
    assert "lowest_positive_level_glass_focus" in selection_by_seed[103_000][
        "reasons"
    ]
    assert selection_by_seed[103_099]["liquid_pixel_count"] == 0
    assert selection_by_seed[103_099]["empty_fraction_of_vial"] > 0
    assert montage.is_file()
    assert report["montage"]["sha256"] == file_sha256(montage)
    with pytest.raises(FileExistsError, match="overwrite montage"):
        summarize_train_chunk(
            chunk_dir,
            103_000,
            103_099,
            expected_role="train",
            montage_out=montage,
            expected_specs_sha256=hashes["specs_sha256"],
            expected_renderer_sha256=hashes["renderer_script_sha256"],
            expected_variation_sha256=hashes["variation_module_sha256"],
        )


def test_aggregate_rendered_hsv_handles_no_visible_liquid_and_report_is_no_overwrite(tmp_path):
    frames = [
        {
            "liquid_hsv": {
                "pixel_count": 0,
                "mean_hue_circular_saturation_weighted_0_to_1": None,
                "mean_saturation": None,
                "mean_value": None,
            }
        }
    ]
    aggregate = aggregate_rendered_hsv(frames)
    assert aggregate["eligible_frame_count"] == 0
    assert aggregate["saturation"]["count"] == 0
    assert aggregate["saturation_0_30_to_0_85"]["fraction_of_eligible"] is None
    report = tmp_path / "report.json"
    write_json_no_overwrite(report, {"ok": True})
    with pytest.raises(FileExistsError, match="overwrite evidence"):
        write_json_no_overwrite(report, {"ok": False})


def test_first_chunk_refuses_wrong_count_or_hash(tmp_path):
    chunk_dir = tmp_path / "chunk"
    hashes = _write_train_chunk(chunk_dir)
    with pytest.raises(ValueError, match="exactly 100"):
        summarize_train_chunk(
            chunk_dir,
            103_000,
            103_098,
            expected_role="train",
            montage_out=tmp_path / "short.png",
            expected_specs_sha256=hashes["specs_sha256"],
            expected_renderer_sha256=hashes["renderer_script_sha256"],
            expected_variation_sha256=hashes["variation_module_sha256"],
        )
    with pytest.raises(ValueError, match="does not match"):
        summarize_train_chunk(
            chunk_dir,
            103_000,
            103_099,
            expected_role="train",
            montage_out=tmp_path / "bad-hash.png",
            expected_specs_sha256=hashes["specs_sha256"],
            expected_renderer_sha256="f" * 64,
            expected_variation_sha256=hashes["variation_module_sha256"],
        )


def test_module_and_cli_require_train_role_and_all_external_hashes(tmp_path):
    with pytest.raises(TypeError, match="expected_specs_sha256"):
        summarize_train_chunk(
            tmp_path / "unused",
            103_000,
            103_099,
            expected_role="train",
            montage_out=tmp_path / "unused.png",
        )
    with pytest.raises(ValueError, match="expected_role='train'"):
        summarize_train_chunk(
            tmp_path / "unused",
            103_000,
            103_099,
            expected_role="test",
            montage_out=tmp_path / "unused.png",
            expected_specs_sha256="1" * 64,
            expected_renderer_sha256="2" * 64,
            expected_variation_sha256="3" * 64,
        )
    with pytest.raises(SystemExit):
        SCRIPT_MODULE._parser().parse_args(
            [
                "--chunk-dir",
                str(tmp_path),
                "--expected-role",
                "train",
                "--expected-seed-start",
                "103000",
                "--expected-seed-end",
                "103099",
                "--report-out",
                str(tmp_path / "report.json"),
                "--montage-out",
                str(tmp_path / "montage.png"),
            ]
        )


def test_independent_checker_vetoes_manifest_count_tampering(tmp_path):
    chunk_dir = tmp_path / "chunk"
    hashes = _write_train_chunk(chunk_dir)
    records = _read_records(chunk_dir)
    records[0]["label_stats"]["class_pixels"]["liquid"] += 1
    _write_records(chunk_dir, records)
    montage = tmp_path / "checker-veto.png"
    with pytest.raises(
        ValueError,
        match="independent capture checker did not pass.*class_pixels.liquid",
    ):
        _summarize(chunk_dir, hashes, montage)
    assert not montage.exists()


def test_preflight_rejects_nontrain_traversal_and_reused_outputs(tmp_path):
    chunk_dir = tmp_path / "chunk"
    hashes = _write_train_chunk(chunk_dir)
    original = _read_records(chunk_dir)

    records = copy.deepcopy(original)
    records[-1]["role"] = "test"
    _write_records(chunk_dir, records)
    with pytest.raises(ValueError, match="refuses non-train"):
        _summarize(chunk_dir, hashes, tmp_path / "nontrain.png")

    records = copy.deepcopy(original)
    records[0]["outputs"]["rgb"]["path"] = "../sealed-test.png"
    _write_records(chunk_dir, records)
    with pytest.raises(ValueError, match="escapes the chunk root"):
        _summarize(chunk_dir, hashes, tmp_path / "traversal.png")

    records = copy.deepcopy(original)
    records[0]["outputs"]["mask"]["sha256"] = "0" * 64
    _write_records(chunk_dir, records)
    with pytest.raises(ValueError, match="MASK SHA-256 mismatch"):
        _summarize(chunk_dir, hashes, tmp_path / "image-hash.png")

    records = copy.deepcopy(original)
    records[1]["outputs"]["rgb"] = copy.deepcopy(records[0]["outputs"]["rgb"])
    _write_records(chunk_dir, records)
    with pytest.raises(ValueError, match="output file is reused"):
        _summarize(chunk_dir, hashes, tmp_path / "reused.png")


def test_summary_requires_true_integrity_full_glass_and_descendant_paths(tmp_path):
    chunk_dir = tmp_path / "chunk"
    hashes = _write_train_chunk(chunk_dir)
    original = _read_summary(chunk_dir)

    summary = copy.deepcopy(original)
    summary["integrity_passed"] = False
    _write_summary(chunk_dir, summary)
    with pytest.raises(ValueError, match="integrity_passed is not true"):
        _summarize(chunk_dir, hashes, tmp_path / "false-integrity.png")

    summary = copy.deepcopy(original)
    del summary["active_vial_glass_material_params"]["specular_rgb"]
    _write_summary(chunk_dir, summary)
    with pytest.raises(ValueError, match="glass specular_rgb"):
        _summarize(chunk_dir, hashes, tmp_path / "missing-glass-field.png")

    summary = copy.deepcopy(original)
    summary["active_vial_glass_material_params"]["specular_rgb"] = [0.96] * 3
    _write_summary(chunk_dir, summary)
    with pytest.raises(ValueError, match="dielectric F0 guard"):
        _summarize(chunk_dir, hashes, tmp_path / "mirror-glass.png")

    summary = copy.deepcopy(original)
    summary["active_vial_glass_material_prim_paths"] = [
        "/Station/OtherVial/Glass"
    ]
    _write_summary(chunk_dir, summary)
    with pytest.raises(ValueError, match="strict descendant"):
        _summarize(chunk_dir, hashes, tmp_path / "sibling-glass.png")

    summary = copy.deepcopy(original)
    summary["source_station_tree_hashes_after"] = {"station.usd": "c" * 64}
    _write_summary(chunk_dir, summary)
    with pytest.raises(ValueError, match="source station hashes changed"):
        _summarize(chunk_dir, hashes, tmp_path / "changed-source.png")


def test_neutral_liquid_hue_writes_montage_without_formatting_error(tmp_path):
    rgb_path = tmp_path / "neutral.png"
    Image.fromarray(np.full((480, 640, 3), 96, dtype=np.uint8), "RGB").save(
        rgb_path
    )
    frame = {
        "seed": 103_000,
        "role": "train",
        "level_ml": 1.0,
        "material": {
            "hsv": [0.0, 0.0, 96 / 255],
            "hue_family": "neutral",
            "saturation_band": "subdued",
            "darkness_band": "dark",
        },
        "liquid_hsv": {
            "pixel_count": 100,
            "mean_hue_circular_saturation_weighted_0_to_1": None,
            "mean_saturation": 0.0,
            "mean_value": 96 / 255,
        },
        "class_pixel_counts": {
            "empty": 900,
            "liquid": 100,
            "foam": 0,
            "blocked": 10,
            "ignore": 640 * 480 - 1010,
        },
        "vial_pixel_count": 1010,
        "class_fractions_of_vial": {
            "empty": 900 / 1010,
            "liquid": 100 / 1010,
            "foam": 0.0,
            "blocked": 10 / 1010,
        },
        "vial_bbox_xyxy": [270, 50, 370, 450],
        "_rgb_path": rgb_path,
    }
    montage_path = tmp_path / "neutral-montage.png"
    metadata = write_montage(montage_path, [frame])
    assert montage_path.is_file()
    assert metadata["selected_seeds"] == [103_000]
    assert metadata["selections"][0]["rendered_liquid_hsv"][
        "mean_hue_circular_saturation_weighted_0_to_1"
    ] is None
