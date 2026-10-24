import importlib.util
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def recovery_script_module():
    path = ROOT / "scripts/run_twin_render_recovery.py"
    spec = importlib.util.spec_from_file_location("run_twin_render_recovery_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recovery_manifest():
    from fluidmix.render_manifest import load_render_manifest

    return load_render_manifest(ROOT / "data/twin_render_recovery_v1.json")


def write_source_lock(directory, root=ROOT):
    from fluidmix.render_runner import RECOVERY_SOURCE_PATHS, sha256_file

    path = directory / "source-lock.json"
    payload = {
        "schema_version": 1,
        "files": {
            relative: sha256_file(root / relative)
            for relative in RECOVERY_SOURCE_PATHS
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_lock_pins_exact_recovery_files_and_detects_drift(tmp_path):
    from fluidmix.render_runner import RECOVERY_SOURCE_PATHS, verify_source_lock

    root = tmp_path / "root"
    for relative in RECOVERY_SOURCE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {relative}\n")
    lock, digest = write_source_lock(tmp_path, root)

    verified = verify_source_lock(lock, expected_lock_sha256=digest, root=root)
    assert verified["lock_sha256"] == digest
    assert set(verified["files"]) == set(RECOVERY_SOURCE_PATHS)

    (root / RECOVERY_SOURCE_PATHS[0]).write_text("drift\n")
    with pytest.raises(ValueError, match="source digest mismatch"):
        verify_source_lock(lock, expected_lock_sha256=digest, root=root)

    with pytest.raises(ValueError, match="lock digest"):
        verify_source_lock(lock, expected_lock_sha256="0" * 64, root=root)


def test_tracked_recovery_source_lock_matches_reviewed_tree():
    from fluidmix.render_runner import verify_source_lock

    lock = ROOT / "data/twin_render_recovery_source_lock_v1.json"
    verified = verify_source_lock(
        lock,
        expected_lock_sha256=(
            "2c9861f7e2e94985e550300ac48a361a3c7805d896b35498ea9c9e85fded8a89"
        ),
        root=ROOT,
    )
    assert len(verified["files"]) == 5


def test_recovery_entry_point_has_no_phase_or_batch_mode():
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_twin_render_recovery.py"), "--help"],
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0
    assert "--phase" not in run.stdout
    assert "batch" not in run.stdout.lower()


def test_attempt_output_root_is_private_exclusive_and_rejects_symlink(tmp_path):
    module = recovery_script_module()
    parent = tmp_path / "approved-private-parent"
    parent.mkdir(mode=0o700)

    root = module.create_attempt_output_root(parent, "attempt-a1")
    assert root.is_dir()
    assert root.stat().st_mode & 0o777 == 0o700

    with pytest.raises(FileExistsError):
        module.create_attempt_output_root(parent, "attempt-a1")

    link = tmp_path / "linked-parent"
    link.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.create_attempt_output_root(link, "attempt-a2")

    with pytest.raises(ValueError, match="one path component"):
        module.create_attempt_output_root(parent, "../outside")


def make_minimal_twin(tmp_path):
    twin = tmp_path / "twin"
    (twin / "tools").mkdir(parents=True)
    (twin / "tools/render_example.py").write_text("# canonical public renderer\n")
    station = twin / "stations/fill_station/fill_station.usda"
    station.parent.mkdir(parents=True)
    station.write_text("#usda 1.0\n")
    return twin


@pytest.mark.parametrize("linked_input", ["manifest", "twin"])
def test_recovery_runner_rejects_symlinked_input_roots_before_attempt(
    tmp_path, monkeypatch, linked_input
):
    module = recovery_script_module()
    twin = make_minimal_twin(tmp_path)
    manifest = ROOT / "data/twin_render_recovery_v1.json"
    isaac = tmp_path / "python.sh"
    isaac.write_text("#!/bin/sh\nexit 99\n")
    isaac.chmod(0o755)
    isaac_sha256 = hashlib.sha256(isaac.read_bytes()).hexdigest()
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    source_lock, source_lock_sha256 = write_source_lock(tmp_path)
    if linked_input == "manifest":
        linked_manifest = tmp_path / "manifest-link.json"
        linked_manifest.symlink_to(manifest)
        manifest = linked_manifest
    else:
        linked_twin = tmp_path / "twin-link"
        linked_twin.symlink_to(twin, target_is_directory=True)
        twin = linked_twin
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_twin_render_recovery.py",
            "--manifest", str(manifest),
            "--source-lock", str(source_lock),
            "--source-lock-sha256", source_lock_sha256,
            "--isaac-python", str(isaac),
            "--isaac-python-sha256", isaac_sha256,
            "--twin", str(twin),
            "--out-parent", str(parent),
        ],
    )

    with pytest.raises(SystemExit):
        module.main()
    assert list(parent.iterdir()) == []


def fake_render_result(command, log_path, *, returncode, attested, wrote):
    log_path.parent.mkdir(mode=0o700, exist_ok=True)
    log_path.write_text("preserved renderer evidence\n")
    return {
        "command": command,
        "started_utc": "start",
        "ended_utc": "end",
        "elapsed_seconds": 1.0,
        "returncode": returncode,
        "timed_out": False,
        "log": str(log_path),
        "log_sha256": "loghash",
        "wrote_marker": wrote,
        "live_setting_attestation": (
            {
                "path": "/exts/isaacsim.core.throttling/enable_async",
                "type": "bool",
                "value": False,
            }
            if attested
            else None
        ),
    }


def invoke_fake_recovery(
    tmp_path, monkeypatch, *, returncode=0, attested=True, wrote=True
):
    module = recovery_script_module()
    twin = make_minimal_twin(tmp_path)
    isaac = tmp_path / "python.sh"
    isaac.write_text("#!/bin/sh\nexit 99\n")
    isaac.chmod(0o755)
    isaac_sha256 = hashlib.sha256(isaac.read_bytes()).hexdigest()
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    source_lock, source_lock_sha256 = write_source_lock(tmp_path)
    calls = []
    inspected = []

    def fake_run(command, log_path, timeout_seconds, env):
        calls.append(command)
        return fake_render_result(
            command,
            log_path,
            returncode=returncode,
            attested=attested,
            wrote=wrote,
        )

    def fake_inspect(path):
        inspected.append(path)
        return {
            "exists": True,
            "is_symlink": False,
            "is_regular": True,
            "decoder_passed": True,
            "content_gate_passed": True,
        }

    monkeypatch.setattr(module, "run_command", fake_run)
    monkeypatch.setattr(module, "inspect_render_output", fake_inspect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_twin_render_recovery.py",
            "--manifest", str(ROOT / "data/twin_render_recovery_v1.json"),
            "--source-lock", str(source_lock),
            "--source-lock-sha256", source_lock_sha256,
            "--isaac-python", str(isaac),
            "--isaac-python-sha256", isaac_sha256,
            "--twin", str(twin),
            "--out-parent", str(parent),
        ],
    )
    result = module.main()
    summary = json.loads(
        (parent / "fillcam-recovery-s00-20261024-a1/summary.json").read_text()
    )
    return result, calls, inspected, summary


def test_recovery_runner_spawns_once_and_preserves_output_after_nonzero(
    tmp_path, monkeypatch
):
    result, calls, inspected, summary = invoke_fake_recovery(
        tmp_path, monkeypatch, returncode=7
    )

    assert result == 1
    assert len(calls) == 1
    assert len(inspected) == 1
    assert summary["render"]["returncode"] == 7
    assert summary["output"]["content_gate_passed"] is True
    assert summary["status"] == "failed"
    assert "renderer_returncode" in summary["failure_reasons"]


def test_recovery_runner_can_only_report_mechanical_checks_passed(
    tmp_path, monkeypatch
):
    result, calls, inspected, summary = invoke_fake_recovery(tmp_path, monkeypatch)

    assert result == 0
    assert len(calls) == len(inspected) == 1
    assert summary["failure_reasons"] == []
    assert summary["status"] == "mechanical_checks_passed"
    assert summary["visual_check_status"] == "pending_local_review"
    assert summary["source_twin_unchanged"] is True
    assert summary["recovery_source_unchanged"] is True
    assert summary["isaac_python_unchanged"] is True
    assert summary["isaac_python_sha256_before"] == summary["isaac_python_sha256_after"]
    assert summary["source_verification_before"] == summary["source_verification_after"]
    assert summary["child_environment"]["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] == "1"
    assert "PYTHONPATH" not in summary["child_environment"]
    assert "LD_LIBRARY_PATH" not in summary["child_environment"]
    assert set(summary["recovery_inputs"]) == {
        "runner_sha256",
        "recovery_renderer_sha256",
        "manifest_loader_sha256",
        "render_runner_sha256",
        "canonical_renderer_sha256",
        "station_sha256",
    }


def test_recovery_runner_fails_if_renderer_never_attests_its_write(tmp_path, monkeypatch):
    result, calls, inspected, summary = invoke_fake_recovery(
        tmp_path, monkeypatch, wrote=False
    )

    assert result == 1
    assert len(calls) == len(inspected) == 1
    assert "renderer_write_marker_missing" in summary["failure_reasons"]


def test_render_command_is_exact_and_contains_only_reviewed_intervention(tmp_path):
    from fluidmix.render_runner import build_render_command

    frozen = recovery_manifest()
    command = build_render_command(
        "/isaac/python.sh",
        "/repo/scripts/render_fillcam_recovery.py",
        "/input/twin/stations/fill_station/fill_station.usda",
        tmp_path / "S00.png",
        frozen,
        frozen.smoke,
        twin="/input/twin",
    )

    assert command == [
        "/isaac/python.sh", "/repo/scripts/render_fillcam_recovery.py",
        "--twin", "/input/twin",
        "--camera", "FillCam", "--frame", "686", "--subframes", "32",
        "--stage", "/input/twin/stations/fill_station/fill_station.usda",
        "--out", str(tmp_path / "S00.png"),
        "--kit-arg=--/telemetry/enableAnonymousData=false",
        "--kit-arg=--/structuredLog/enable=false",
        "--kit-arg=--/exts/isaacsim.core.throttling/enable_async=false",
    ]
    assert not any("maxAssetLoadingTime" in part for part in command)


def test_recovery_render_command_cannot_omit_external_twin(tmp_path):
    from fluidmix.render_runner import build_render_command

    frozen = recovery_manifest()
    with pytest.raises(TypeError, match="twin"):
        build_render_command(
            "/isaac/python.sh",
            "/repo/scripts/render_fillcam_recovery.py",
            "/input/twin/stations/fill_station/fill_station.usda",
            tmp_path / "S00.png",
            frozen,
            frozen.smoke,
        )


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["--/exts/isaacsim.core.throttling/enable_async=false"],
        [
            "--/telemetry/enableAnonymousData=false",
            "--/structuredLog/enable=false",
            "--/exts/isaacsim.core.throttling/enable_async=false",
            "--/unreviewed=true",
        ],
    ],
)
def test_renderer_rejects_any_kit_argument_drift(values):
    from fluidmix.render_runner import validate_recovery_kit_args

    with pytest.raises(ValueError, match="exactly match"):
        validate_recovery_kit_args(values)


@pytest.mark.parametrize("value", [None, True, "false", 0])
def test_live_async_setting_must_be_typed_boolean_false(value):
    from fluidmix.render_runner import require_async_throttling_disabled

    with pytest.raises(RuntimeError, match="Boolean false"):
        require_async_throttling_disabled(value)


def test_live_async_attestation_reads_exact_setting_and_records_type():
    from fluidmix.render_runner import attest_live_async_setting

    class Settings:
        def __init__(self):
            self.paths = []

        def get(self, path):
            self.paths.append(path)
            return False

    settings = Settings()
    assert attest_live_async_setting(settings) == {
        "path": "/exts/isaacsim.core.throttling/enable_async",
        "type": "bool",
        "value": False,
    }
    assert settings.paths == ["/exts/isaacsim.core.throttling/enable_async"]


def test_live_setting_attestation_is_parsed_from_exactly_one_log_line():
    from fluidmix.render_runner import parse_live_setting_attestation

    line = (
        "live_setting_attestation "
        '{"path":"/exts/isaacsim.core.throttling/enable_async",'
        '"type":"bool","value":false}'
    )
    assert parse_live_setting_attestation(f"startup\n{line}\nshutdown\n")["value"] is False
    with pytest.raises(ValueError, match="multiple"):
        parse_live_setting_attestation(f"{line}\n{line}\n")


def test_recovery_renderer_enforces_contract_before_startup_and_capture():
    source = (ROOT / "scripts/render_fillcam_recovery.py").read_text()

    kit_validation = source.index("validate_recovery_kit_args(a.kit_arg)")
    request_validation = source.index("validate_recovery_render_request(")
    startup = source.index("SimulationApp(")
    attestation = source.index("attest_live_async_setting(")
    capture = source.index("rep.orchestrator.step(")
    assert kit_validation < startup
    assert request_validation < startup
    assert startup < attestation < capture
    assert 'choices=["FillCam"]' in source


def test_recovery_renderer_requires_external_twin_and_safe_final_publish():
    source = (ROOT / "scripts/render_fillcam_recovery.py").read_text()

    assert 'ap.add_argument("--twin", required=True)' in source
    assert 'twin = Path(a.twin).expanduser().resolve(strict=True)' in source
    assert 'twin / "cameras" / (a.camera + ".json")' in source
    assert "tempfile.NamedTemporaryFile" in source
    assert "os.link(" in source
    assert "cv2.imwrite(a.out" not in source


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera", "CapSealCam"),
        ("frame", 685),
        ("subframes", 31),
        ("resolution", (641, 480)),
    ],
)
def test_recovery_request_rejects_any_view_or_capture_drift(tmp_path, field, value):
    from fluidmix.render_runner import validate_recovery_render_request

    twin = make_minimal_twin(tmp_path)
    station = twin / "stations/fill_station/fill_station.usda"
    request = {
        "camera": "FillCam",
        "frame": 686,
        "subframes": 32,
        "resolution": (640, 480),
    }
    request[field] = value

    with pytest.raises(ValueError, match=field):
        validate_recovery_render_request(twin, station, **request)


def test_recovery_request_accepts_only_canonical_fill_station(tmp_path):
    from fluidmix.render_runner import validate_recovery_render_request

    twin = make_minimal_twin(tmp_path)
    station = twin / "stations/fill_station/fill_station.usda"
    assert validate_recovery_render_request(
        twin, station, "FillCam", 686, 32, (640, 480)
    ) == station.resolve()

    other = twin / "other.usda"
    other.write_text("#usda 1.0\n")
    with pytest.raises(ValueError, match="stage"):
        validate_recovery_render_request(
            twin, other, "FillCam", 686, 32, (640, 480)
        )


def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def filtered_rgb_png(width, height, rows, filter_type=0):
    prior = bytes(width * 3)
    scanlines = []
    for values in rows:
        source = bytes(values)
        encoded = bytearray(len(source))
        for index, value in enumerate(source):
            left = source[index - 3] if index >= 3 else 0
            above = prior[index]
            upper_left = prior[index - 3] if index >= 3 else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            else:
                estimate = left + above - upper_left
                distances = (
                    abs(estimate - left),
                    abs(estimate - above),
                    abs(estimate - upper_left),
                )
                predictor = (left, above, upper_left)[distances.index(min(distances))]
            encoded[index] = (value - predictor) & 0xFF
        scanlines.append(bytes([filter_type]) + encoded)
        prior = source
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(b"".join(scanlines)))
        + png_chunk(b"IEND", b"")
    )


def raw_rgb_png(width, height, raw, *, color_type=2, middle_chunks=()):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    compressed = zlib.compress(raw)
    midpoint = len(compressed) // 2
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", compressed[:midpoint])
        + b"".join(png_chunk(kind, data) for kind, data in middle_chunks)
        + png_chunk(b"IDAT", compressed[midpoint:])
        + png_chunk(b"IEND", b"")
    )


def test_png_rgb_statistics_use_reconstructed_rgb_pixel_denominators(tmp_path):
    from fluidmix.render_runner import png_rgb_statistics

    path = tmp_path / "image.png"
    path.write_bytes(filtered_rgb_png(2, 1, [[0, 10, 20, 30, 40, 50]]))

    assert png_rgb_statistics(path, expected_resolution=(2, 1)) == {
        "width": 2,
        "height": 1,
        "pixel_count": 2,
        "scalar_rgb_samples": 6,
        "channel_min": [0, 10, 20],
        "channel_max": [30, 40, 50],
        "channel_mean": [15.0, 25.0, 35.0],
        "channel_stddev": [15.0, 15.0, 15.0],
        "spatial_rgb_stddev": 15.0,
        "nonzero_rgb_pixels": 2,
        "nonzero_rgb_fraction": 1.0,
        "distinct_rgb_count": 2,
    }


@pytest.mark.parametrize("filter_type", [1, 2, 3, 4])
def test_png_rgb_statistics_reverse_all_row_filters(tmp_path, filter_type):
    from fluidmix.render_runner import png_rgb_statistics

    rows = [
        [0, 10, 20, 30, 40, 50, 60, 70, 80],
        [90, 100, 110, 120, 130, 140, 150, 160, 170],
    ]
    plain = tmp_path / "plain.png"
    filtered = tmp_path / f"filter-{filter_type}.png"
    plain.write_bytes(filtered_rgb_png(3, 2, rows))
    filtered.write_bytes(filtered_rgb_png(3, 2, rows, filter_type))

    assert png_rgb_statistics(filtered, expected_resolution=(3, 2)) == png_rgb_statistics(
        plain, expected_resolution=(3, 2)
    )


@pytest.mark.parametrize(
    "row,reason",
    [
        ([0, 0, 0, 0, 0, 0], "black"),
        ([10, 20, 30, 10, 20, 30], "constant"),
    ],
)
def test_content_gate_rejects_black_and_uniform_nonblack(tmp_path, row, reason):
    from fluidmix.render_runner import png_rgb_statistics, require_nonblank_rgb

    path = tmp_path / "image.png"
    path.write_bytes(filtered_rgb_png(2, 1, [row]))
    with pytest.raises(ValueError, match=reason):
        require_nonblank_rgb(png_rgb_statistics(path, expected_resolution=(2, 1)))


def test_content_gate_records_but_does_not_hide_weak_variation(tmp_path):
    from fluidmix.render_runner import png_rgb_statistics, require_nonblank_rgb

    path = tmp_path / "image.png"
    path.write_bytes(filtered_rgb_png(2, 1, [[0, 0, 0, 1, 0, 0]]))
    assert require_nonblank_rgb(
        png_rgb_statistics(path, expected_resolution=(2, 1))
    ) is None


def test_output_inspection_preserves_missing_black_and_varying_evidence(tmp_path):
    from fluidmix.render_runner import inspect_render_output

    assert inspect_render_output(tmp_path / "missing.png", expected_resolution=(2, 1)) == {
        "exists": False,
        "is_symlink": False,
        "is_regular": False,
    }
    black = tmp_path / "black.png"
    black.write_bytes(filtered_rgb_png(2, 1, [[0, 0, 0, 0, 0, 0]]))
    black_result = inspect_render_output(black, expected_resolution=(2, 1))
    assert black_result["decoder_passed"] is True
    assert black_result["content_gate_passed"] is False
    assert black_result["statistics"]["pixel_count"] == 2
    assert black_result["evidence_snapshot"] == "single_read"

    varying = tmp_path / "varying.png"
    varying.write_bytes(filtered_rgb_png(2, 1, [[0, 0, 0, 1, 2, 3]]))
    varying_result = inspect_render_output(varying, expected_resolution=(2, 1))
    assert varying_result["decoder_passed"] is True
    assert varying_result["content_gate_passed"] is True


@pytest.mark.parametrize(
    "payload,match",
    [
        (raw_rgb_png(2, 1, b"\x05" + b"\x00" * 6), "filter"),
        (raw_rgb_png(2, 1, b"\x00" + b"\x00" * 7), "exceeds|wrong size"),
        (raw_rgb_png(2, 1, b"\x00" + b"\x00" * 6, color_type=6), "8-bit RGB"),
        (
            raw_rgb_png(2, 1, b"\x00" + b"\x00" * 6, middle_chunks=((b"tEXt", b"x"),)),
            "out of order",
        ),
    ],
)
def test_png_decoder_rejects_malformed_unsupported_or_disordered_streams(
    tmp_path, payload, match
):
    from fluidmix.render_runner import png_rgb_statistics

    path = tmp_path / "image.png"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=match):
        png_rgb_statistics(path, expected_resolution=(2, 1))


def test_png_decoder_rejects_crc_corruption_and_symlink(tmp_path):
    from fluidmix.render_runner import png_rgb_statistics

    path = tmp_path / "image.png"
    payload = bytearray(filtered_rgb_png(2, 1, [[0, 0, 0, 1, 2, 3]]))
    payload[-5] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="CRC"):
        png_rgb_statistics(path, expected_resolution=(2, 1))

    target = tmp_path / "target.png"
    target.write_bytes(filtered_rgb_png(2, 1, [[0, 0, 0, 1, 2, 3]]))
    link = tmp_path / "link.png"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        png_rgb_statistics(link, expected_resolution=(2, 1))


def test_child_environment_is_allowlisted_and_drops_secrets():
    from fluidmix.render_runner import build_child_environment

    source = {
        "PATH": "/usr/bin",
        "HOME": "/home/render",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "NVIDIA_API_KEY": "secret",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "CUSTOM_TOKEN": "secret",
        "PYTHONPATH": "/unreviewed/python",
        "LD_LIBRARY_PATH": "/unreviewed/libraries",
        "EXP_PATH": "/unreviewed/kit",
        "OV_CACHE_ROOT": "/unreviewed/cache",
    }
    assert build_child_environment(source) == {
        "PATH": "/usr/bin",
        "HOME": "/home/render",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA": "1",
    }


def test_output_path_occupied_detects_dangling_symlink(tmp_path):
    from fluidmix.render_runner import output_path_occupied

    link = tmp_path / "render.png"
    link.symlink_to(tmp_path / "missing-target")
    assert output_path_occupied(link)


def test_private_output_path_rejects_relative_traversal_and_intermediate_symlink(tmp_path):
    from fluidmix.render_runner import resolve_private_new_output

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    expected = private / "S00.png"
    assert resolve_private_new_output(expected, expected_name="S00.png") == expected

    with pytest.raises(ValueError, match="absolute"):
        resolve_private_new_output(Path("private/S00.png"), expected_name="S00.png")
    with pytest.raises(ValueError, match="name"):
        resolve_private_new_output(private / "other.png", expected_name="S00.png")

    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    link = private / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|traversal"):
        resolve_private_new_output(link / "S00.png", expected_name="S00.png")
    with pytest.raises(ValueError, match="symlink|traversal"):
        resolve_private_new_output(
            link / ".." / "S00.png", expected_name="S00.png"
        )


def test_hash_regular_tree_is_complete_sorted_and_rejects_symlinks(tmp_path):
    from fluidmix.render_runner import hash_regular_tree, sha256_file

    root = tmp_path / "tree"
    (root / "nested").mkdir(parents=True)
    (root / "b.bin").write_bytes(b"b")
    (root / "nested/a.bin").write_bytes(b"a")
    assert hash_regular_tree(root) == {
        "b.bin": sha256_file(root / "b.bin"),
        "nested/a.bin": sha256_file(root / "nested/a.bin"),
    }
    (root / "link.bin").symlink_to(root / "b.bin")
    with pytest.raises(ValueError, match="symlink"):
        hash_regular_tree(root)
    with pytest.raises(ValueError, match="non-symlink"):
        sha256_file(root / "link.bin")


def test_run_command_refuses_dangling_log_symlink(tmp_path):
    module = recovery_script_module()
    link = tmp_path / "render.log"
    link.symlink_to(tmp_path / "outside.log")

    with pytest.raises(ValueError, match="overwrite"):
        module.run_command(
            [sys.executable, "-c", "print('must not run')"],
            link,
            10,
            {"PATH": os.environ.get("PATH", "")},
        )


def test_run_command_records_spawn_failure_without_retry(tmp_path):
    module = recovery_script_module()
    result = module.run_command(
        [str(tmp_path / "missing-executable")],
        tmp_path / "spawn.log",
        10,
        {"PATH": os.environ.get("PATH", "")},
    )

    assert result["returncode"] is None
    assert result["timed_out"] is False
    assert result["spawn_error"].startswith("FileNotFoundError:")
    assert result["log_sha256"]


def test_run_command_records_live_setting_and_wrote_markers(tmp_path):
    module = recovery_script_module()
    code = (
        "print('live_setting_attestation "
        '{\"path\":\"/exts/isaacsim.core.throttling/enable_async\",'
        '\"type\":\"bool\",\"value\":false}\')\n'
        "print('wrote /private/S00.png')"
    )
    result = module.run_command(
        [sys.executable, "-c", code],
        tmp_path / "render.log",
        10,
        {"PATH": os.environ.get("PATH", "")},
    )

    assert result["returncode"] == 0
    assert result["wrote_marker"] is True
    assert result["live_setting_attestation"]["value"] is False


def test_save_summary_refuses_dangling_temporary_symlink(tmp_path):
    module = recovery_script_module()
    summary = tmp_path / "summary.json"
    (tmp_path / "summary.json.tmp").symlink_to(tmp_path / "outside.json")

    with pytest.raises(FileExistsError):
        module.save_summary(summary, {"status": "must-not-write"})
