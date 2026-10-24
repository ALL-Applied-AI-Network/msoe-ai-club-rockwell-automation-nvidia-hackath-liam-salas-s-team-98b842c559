from pathlib import Path
import importlib.util
import os
import struct
import sys
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def batch_script_module():
    path = ROOT / "scripts/run_twin_render_batch.py"
    spec = importlib.util.spec_from_file_location("run_twin_render_batch_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest():
    from fluidmix.render_manifest import load_render_manifest

    return load_render_manifest(ROOT / "data/twin_render_feasibility_v1.json")


def test_direct_variant_command_is_argument_safe_and_exact(tmp_path):
    from fluidmix.render_runner import build_variant_command

    job = manifest().batch[3]
    command = build_variant_command(
        "/isaac/python.sh", "/repo/scripts/twin_liquid_variant.py",
        "/input/twin", tmp_path / "M03.usda", job,
    )

    assert command == [
        "/isaac/python.sh", "/repo/scripts/twin_liquid_variant.py",
        "--twin", "/input/twin",
        "--material-color", "0.117,0.18,0.2275",
        "--opacity", "0.6",
        "--variant-id", "M03",
        "--out", str(tmp_path / "M03.usda"),
    ]
    assert all("API_KEY" not in part for part in command)


def test_default_stage_job_has_no_variant_command(tmp_path):
    from fluidmix.render_runner import build_variant_command

    assert build_variant_command(
        "/isaac/python.sh", "/repo/scripts/twin_liquid_variant.py",
        "/input/twin", tmp_path / "S00.usda", manifest().smoke[0],
    ) is None


def test_render_command_uses_fixed_view_and_reviewed_privacy_args(tmp_path):
    from fluidmix.render_runner import build_render_command

    frozen = manifest()
    command = build_render_command(
        "/isaac/python.sh", "/input/twin/tools/render_example.py",
        "/input/twin/stations/fill_station/fill_station.usda",
        tmp_path / "S00.png", frozen, frozen.smoke[0],
    )

    assert command == [
        "/isaac/python.sh", "/input/twin/tools/render_example.py",
        "--camera", "FillCam", "--frame", "686", "--subframes", "32",
        "--stage", "/input/twin/stations/fill_station/fill_station.usda",
        "--out", str(tmp_path / "S00.png"),
        "--kit-arg=--/telemetry/enableAnonymousData=false",
        "--kit-arg=--/structuredLog/enable=false",
    ]


def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def valid_rgb_png(width=2, height=1):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )


def test_png_dimensions_validates_complete_image_without_image_dependency(tmp_path):
    from fluidmix.render_runner import png_dimensions

    path = tmp_path / "image.png"
    path.write_bytes(valid_rgb_png(2, 1))
    assert png_dimensions(path) == (2, 1)


def test_png_dimensions_rejects_non_png(tmp_path):
    from fluidmix.render_runner import png_dimensions

    path = tmp_path / "image.png"
    path.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="PNG"):
        png_dimensions(path)


def test_png_dimensions_rejects_truncated_header_only_png(tmp_path):
    from fluidmix.render_runner import png_dimensions

    path = tmp_path / "image.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
        + struct.pack(">II", 640, 480) + b"rest"
    )
    with pytest.raises(ValueError, match="PNG"):
        png_dimensions(path)


def test_child_environment_is_allowlisted_and_drops_secrets():
    from fluidmix.render_runner import build_child_environment

    source = {
        "PATH": "/usr/bin",
        "HOME": "/home/render",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "NVIDIA_API_KEY": "secret",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "CUSTOM_TOKEN": "secret",
    }
    result = build_child_environment(source)

    assert result == {
        "PATH": "/usr/bin",
        "HOME": "/home/render",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA": "1",
    }


def test_output_path_occupied_detects_dangling_symlink(tmp_path):
    from fluidmix.render_runner import output_path_occupied

    target = tmp_path / "missing-target"
    link = tmp_path / "render.png"
    link.symlink_to(target)
    assert output_path_occupied(link)


def test_run_command_refuses_dangling_log_symlink(tmp_path):
    module = batch_script_module()
    link = tmp_path / "render.log"
    link.symlink_to(tmp_path / "outside.log")

    with pytest.raises(FileExistsError):
        module.run_command(
            [sys.executable, "-c", "print('must not run')"],
            link,
            10,
            {"PATH": os.environ.get("PATH", "")},
        )


def test_save_summary_refuses_dangling_temporary_symlink(tmp_path):
    module = batch_script_module()
    summary = tmp_path / "summary.json"
    temporary = tmp_path / "summary.json.tmp"
    temporary.symlink_to(tmp_path / "outside.json")

    with pytest.raises(FileExistsError):
        module.save_summary(summary, {"status": "must-not-write"})
