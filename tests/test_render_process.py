"""CPU tests for the bounded external render process boundary."""

import os
from pathlib import Path
import json
import hashlib
import stat
import subprocess
import sys
import time

import pytest
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fluidmix.render_process as render_process


def test_child_environment_allowlists_runtime_identity_and_drops_unsafe_values():
    source = {
        "PATH": "/usr/bin",
        "HOME": "/home/render",
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "OMNI_KIT_ACCEPT_EULA": "NO",
        "OMNI_KIT_CACHE_PATH": "/attacker/cache",
        "XDG_CACHE_HOME": "/attacker/xdg",
        "PYTHONPATH": "/attacker/python",
        "LD_LIBRARY_PATH": "/attacker/lib",
        "LD_PRELOAD": "/attacker/preload.so",
        "HF_TOKEN": "secret",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "GOOGLE_APPLICATION_CREDENTIALS": "/secret/cloud.json",
    }

    child = render_process.build_child_environment(source, "generic")

    assert child == {
        "PATH": "/usr/bin",
        "HOME": "/home/render",
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA": "1",
    }


def test_child_environment_injects_only_exact_reviewed_gb10_lexical_preload():
    child = render_process.build_child_environment(
        {"LD_PRELOAD": "/unreviewed.so", "PATH": "/usr/bin"},
        "gb10-aarch64",
    )

    assert child["LD_PRELOAD"] == "/lib/aarch64-linux-gnu/libgomp.so.1"
    assert "/unreviewed.so" not in child.values()


def _configure_preload_fixture(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "libgomp.so.1.0.0"
    target.write_bytes(b"reviewed-target")
    lexical = tmp_path / "libgomp.so.1"
    lexical.symlink_to(target)
    monkeypatch.setattr(render_process, "GB10_PRELOAD", lexical)
    monkeypatch.setattr(render_process, "GB10_PRELOAD_ALLOWED_ROOTS", (allowed,))
    return lexical, target, allowed


def test_gb10_preload_snapshots_fixed_link_and_resolved_regular_target(
    tmp_path, monkeypatch
):
    lexical, target, _ = _configure_preload_fixture(tmp_path, monkeypatch)

    snapshot = render_process.snapshot_gb10_preload()

    assert snapshot["lexical_path"] == str(lexical)
    assert snapshot["lexical_identity"]["kind"] == "symlink"
    assert snapshot["lexical_identity"]["link_target"] == str(target)
    assert snapshot["resolved_target"]["path"] == str(target.resolve())
    assert snapshot["resolved_target"]["sha256"] == hashlib.sha256(
        b"reviewed-target"
    ).hexdigest()


@pytest.mark.parametrize("change", ["link", "target_bytes"])
def test_gb10_preload_reverification_detects_link_or_target_change(
    tmp_path, monkeypatch, change
):
    lexical, target, allowed = _configure_preload_fixture(tmp_path, monkeypatch)
    before = render_process.snapshot_gb10_preload()
    if change == "link":
        replacement = allowed / "replacement.so"
        replacement.write_bytes(b"replacement")
        lexical.unlink()
        lexical.symlink_to(replacement)
    else:
        target.write_bytes(b"drifted-target")

    verification = render_process.reverify_gb10_preload(before)

    assert verification["unchanged"] is False
    assert verification["before"] == before
    assert verification["after"] != before


def test_gb10_preload_rejects_resolved_target_outside_allowed_roots(
    tmp_path, monkeypatch
):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.so"
    outside.write_bytes(b"outside")
    lexical = tmp_path / "libgomp.so.1"
    lexical.symlink_to(outside)
    monkeypatch.setattr(render_process, "GB10_PRELOAD", lexical)
    monkeypatch.setattr(render_process, "GB10_PRELOAD_ALLOWED_ROOTS", (allowed,))

    with pytest.raises(ValueError, match="allowed GB10 library roots"):
        render_process.snapshot_gb10_preload()


def test_child_environment_rejects_unknown_profile():
    with pytest.raises(ValueError, match="profile"):
        render_process.build_child_environment({}, "automatic")


def test_bounded_runner_records_normal_exit_and_private_log(tmp_path):
    log = tmp_path / "normal.log"
    result = render_process.run_bounded_command(
        [sys.executable, "-c", "print('complete', flush=True)"],
        log,
        timeout_seconds=2,
        environment={"PATH": os.environ.get("PATH", "")},
    )

    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert result["spawn_error"] is None
    assert result["termination_escalation"] == "none"
    assert result["command"] == [sys.executable, "-c", "print('complete', flush=True)"]
    assert result["log_path"] == str(log)
    assert len(result["log_sha256"]) == 64
    assert log.read_text() == "complete\n"
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert result["started_utc"].endswith("+00:00")
    assert result["ended_utc"].endswith("+00:00")
    assert result["elapsed_seconds"] >= 0


def test_bounded_runner_records_spawn_error_and_preserves_log(tmp_path):
    log = tmp_path / "spawn-error.log"
    result = render_process.run_bounded_command(
        [str(tmp_path / "missing-executable")],
        log,
        timeout_seconds=1,
        environment={},
    )

    assert result["returncode"] is None
    assert result["timed_out"] is False
    assert result["spawn_error"].startswith("FileNotFoundError:")
    assert "FileNotFoundError:" in log.read_text()
    assert len(result["log_sha256"]) == 64


def test_bounded_runner_times_out_and_terminates_process_group(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(render_process, "TERMINATION_GRACE_SECONDS", 0.2)
    descendant_pid = tmp_path / "descendant.pid"
    parent_source = (
        "import pathlib,signal,subprocess,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);time.sleep(30)']);"
        f"pathlib.Path({str(descendant_pid)!r}).write_text(str(child.pid));"
        "print('spawned', child.pid, flush=True);time.sleep(30)"
    )
    result = render_process.run_bounded_command(
        [sys.executable, "-c", parent_source],
        tmp_path / "timeout.log",
        timeout_seconds=0.2,
        environment=dict(os.environ),
    )

    assert result["timed_out"] is True
    assert result["termination_escalation"] == "sigkill"
    assert result["returncode"] == -9
    pid = int(descendant_pid.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        if state == "Z":
            break
        time.sleep(0.01)
    if Path(f"/proc/{pid}/stat").exists():
        assert Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"


def test_bounded_runner_quiesces_normal_exit_descendant_before_hashing_log(
    tmp_path
):
    descendant_pid = tmp_path / "normal-descendant.pid"
    child_source = (
        "import time;"
        "[(print('descendant-write', flush=True), time.sleep(0.01)) "
        "for _ in range(1000)]"
    )
    parent_source = (
        "import pathlib,subprocess,sys;"
        f"child=subprocess.Popen([sys.executable,'-u','-c',{child_source!r}]);"
        f"pathlib.Path({str(descendant_pid)!r}).write_text(str(child.pid));"
        "print('leader-exit', flush=True)"
    )
    log = tmp_path / "normal-descendant.log"

    result = render_process.run_bounded_command(
        [sys.executable, "-c", parent_source],
        log,
        timeout_seconds=2,
        environment=dict(os.environ),
    )

    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert result["process_group_cleanup_required"] is True
    assert result["process_group_quiescent"] is True
    assert result["termination_escalation"] == "sigterm"
    first_bytes = log.read_bytes()
    assert hashlib.sha256(first_bytes).hexdigest() == result["log_sha256"]
    time.sleep(0.1)
    assert log.read_bytes() == first_bytes


def test_bounded_runner_refuses_existing_or_dangling_log(tmp_path):
    existing = tmp_path / "existing.log"
    existing.write_text("keep")
    dangling = tmp_path / "dangling.log"
    dangling.symlink_to(tmp_path / "missing")

    for path in (existing, dangling):
        with pytest.raises(FileExistsError):
            render_process.run_bounded_command(
                [sys.executable, "-c", "pass"],
                path,
                timeout_seconds=1,
                environment={},
            )
    assert existing.read_text() == "keep"


@pytest.mark.parametrize("command", [("not", "a", "list"), [], ["ok", 1]])
def test_bounded_runner_accepts_only_nonempty_string_list_argv(tmp_path, command):
    with pytest.raises((TypeError, ValueError), match="command"):
        render_process.run_bounded_command(
            command,
            tmp_path / "invalid.log",
            timeout_seconds=1,
            environment={},
        )


def test_regular_tree_hashes_sorted_paths_with_hashes_and_byte_counts(tmp_path):
    root = tmp_path / "tree"
    (root / "nested").mkdir(parents=True)
    (root / "z.txt").write_bytes(b"z")
    (root / "nested" / "a.txt").write_bytes(b"alpha")

    snapshot = render_process.hash_regular_tree(root)

    assert list(snapshot) == ["nested/a.txt", "z.txt"]
    assert snapshot["nested/a.txt"] == {
        "sha256": "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8",
        "bytes": 5,
    }
    assert snapshot["z.txt"]["bytes"] == 1


def test_regular_tree_rejects_symlink_root_leaf_and_nonregular_entry(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "target"
    target.write_text("data")
    linked_leaf = root / "linked"
    linked_leaf.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        render_process.hash_regular_tree(root)

    linked_leaf.unlink()
    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="non-regular"):
        render_process.hash_regular_tree(root)

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        render_process.hash_regular_tree(linked_root)


def test_regular_file_hash_rejects_byte_drift_while_reading(tmp_path, monkeypatch):
    source = tmp_path / "drift.bin"
    source.write_bytes(b"a" * (1024 * 1024 + 1))
    original_read = render_process.os.read
    changed = False

    def drift_after_first_read(descriptor, size):
        nonlocal changed
        data = original_read(descriptor, size)
        if data and not changed:
            changed = True
            source.write_bytes(b"changed")
        return data

    monkeypatch.setattr(render_process.os, "read", drift_after_first_read)
    with pytest.raises(ValueError, match="changed while hashing"):
        render_process.hash_regular_file(source)


def test_attempt_root_is_private_exclusive_and_rejects_unsafe_targets(tmp_path):
    parent = tmp_path / "attempts"
    parent.mkdir()

    root = render_process.create_attempt_root(parent, "run-20261024.a1")
    assert root == parent.resolve() / "run-20261024.a1"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    with pytest.raises(FileExistsError):
        render_process.create_attempt_root(parent, "run-20261024.a1")

    dangling = parent / "dangling"
    dangling.symlink_to(parent / "missing")
    with pytest.raises(FileExistsError):
        render_process.create_attempt_root(parent, "dangling")

    for unsafe in ("", ".", "..", "../outside", "nested/run", " white"):
        with pytest.raises(ValueError, match="run ID"):
            render_process.create_attempt_root(parent, unsafe)

    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        render_process.create_attempt_root(linked_parent, "another-run")


def test_atomic_json_publication_is_complete_exclusive_and_contained(tmp_path):
    parent = tmp_path / "attempts"
    parent.mkdir()
    root = render_process.create_attempt_root(parent, "run-a1")
    payload = {"schema_version": 1, "status": "failed"}

    path = render_process.atomic_write_json(root, "attempt.json", payload)

    assert path == root / "attempt.json"
    assert json.loads(path.read_text()) == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(root.glob(".*.tmp-*"))
    with pytest.raises(FileExistsError):
        render_process.atomic_write_json(root, "attempt.json", payload)
    with pytest.raises(ValueError, match="file name"):
        render_process.atomic_write_json(root, "../escape.json", payload)

    linked_root = tmp_path / "linked-attempt"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        render_process.atomic_write_json(linked_root, "linked.json", payload)


REQUESTED_SETTINGS = {
    "camera": "FillCam",
    "stage": "stations/fill_station/fill_station.usda",
    "frames": "1:2:1",
    "subframes": 32,
    "warmup": 60,
    "max_tries": 20,
    "seed": 20261024,
    "require_liquid_pixels": True,
    "renderer": "PathTracing",
}


def _sha256(path):
    return render_process.hash_regular_file(path)["sha256"]


def write_complete_capture(root: Path) -> dict:
    for name in ("rgb", "semantic", "instance", "liquid"):
        (root / name).mkdir(parents=True, exist_ok=True)
    rgb = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[15, 25, 35], [45, 55, 65], [75, 85, 95]],
        ],
        dtype=np.uint8,
    )
    liquid = np.array([[0, 255, 0], [0, 255, 0]], dtype=np.uint8)
    semantic = np.array([[1, 2, 1], [1, 2, 0]], dtype=np.uint32)
    instance = np.array([[4, 7, 4], [4, 7, 0]], dtype=np.uint64)
    Image.fromarray(rgb, mode="RGB").save(root / "rgb/0001.png")
    Image.fromarray(liquid, mode="L").save(root / "liquid/0001.png")
    np.save(root / "semantic/0001.npy", semantic, allow_pickle=False)
    np.save(root / "instance/0001.npy", instance, allow_pickle=False)
    artifacts = {
        name: {
            "path": f"{name}/0001.{('npy' if name in {'semantic', 'instance'} else 'png')}",
        }
        for name in ("rgb", "semantic", "instance", "liquid")
    }
    for record in artifacts.values():
        record["sha256"] = _sha256(root / record["path"])
    summary = {
        "schema_version": 1,
        "mechanical_gates_passed": True,
        "visual_check_status": "pending",
        "inputs": {
            "stage": {"requested": REQUESTED_SETTINGS["stage"]},
            "camera": {"name": "FillCam"},
        },
        "render_settings": {
            "renderer": "PathTracing",
            "resolution": [3, 2],
            "frame_spec": "1:2:1",
            "time_codes": [1],
            "warmup_updates": 60,
            "rt_subframes": 32,
            "max_tries_per_frame": 20,
            "require_liquid_pixels": True,
            "replicator_global_seed": 20261024,
        },
        "frame_count": 1,
        "frames": [
            {
                "capture_ordinal": 1,
                "time_code": 1,
                "rgb": {"shape": [2, 3, 3], "dtype": "uint8"},
                "class_pixel_counts": {"liquid": 2, "vial": 3},
                "instance_agreement": {"semantic_liquid_pixels": 2},
                "artifacts": artifacts,
            }
        ],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def test_capture_summary_verifier_accepts_complete_schema_one_fixture(tmp_path):
    capture = tmp_path / "capture"
    write_complete_capture(capture)

    verified = render_process.verify_capture_summary(capture, REQUESTED_SETTINGS)

    assert verified["summary"]["frame_count"] == 1
    assert verified["summary_path"] == str(capture / "summary.json")
    assert len(verified["summary_sha256"]) == 64
    assert list(verified["artifacts"]) == [
        "instance/0001.npy",
        "liquid/0001.png",
        "rgb/0001.png",
        "semantic/0001.npy",
    ]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "schema"),
        ("mechanical_gates_passed", False, "mechanical"),
        ("visual_check_status", "accepted", "visual"),
    ],
)
def test_capture_summary_verifier_rejects_wrong_schema_or_status(
    tmp_path, field, value, match
):
    capture = tmp_path / "capture"
    summary = write_complete_capture(capture)
    summary[field] = value
    (capture / "summary.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match=match):
        render_process.verify_capture_summary(capture, REQUESTED_SETTINGS)


def test_capture_summary_verifier_rejects_requested_setting_mismatch(tmp_path):
    capture = tmp_path / "capture"
    write_complete_capture(capture)
    expected = dict(REQUESTED_SETTINGS, renderer="RaytracedLighting")

    with pytest.raises(ValueError, match="requested setting"):
        render_process.verify_capture_summary(capture, expected)


@pytest.mark.parametrize("damage", ["missing", "extra", "hash", "unreadable"])
def test_capture_summary_verifier_rejects_incomplete_or_drifted_artifacts(
    tmp_path, damage
):
    capture = tmp_path / "capture"
    write_complete_capture(capture)
    if damage == "missing":
        (capture / "liquid/0001.png").unlink()
    elif damage == "extra":
        (capture / "rgb/extra.png").write_bytes(b"extra")
    elif damage == "unreadable":
        (capture / "rgb/0001.png").chmod(0)
    else:
        (capture / "rgb/0001.png").write_bytes(b"drift")

    with pytest.raises(ValueError):
        render_process.verify_capture_summary(capture, REQUESTED_SETTINGS)


def test_capture_summary_verifier_rejects_drift_between_tree_scans(
    tmp_path, monkeypatch
):
    capture = tmp_path / "capture"
    write_complete_capture(capture)
    real_hash_tree = render_process.hash_regular_tree
    calls = 0

    def drift_after_first_tree(root):
        nonlocal calls
        snapshot = real_hash_tree(root)
        calls += 1
        if calls == 1:
            (capture / "rgb/0001.png").write_bytes(b"late drift")
        return snapshot

    monkeypatch.setattr(render_process, "hash_regular_tree", drift_after_first_tree)
    with pytest.raises(ValueError, match="changed during verification"):
        render_process.verify_capture_summary(capture, REQUESTED_SETTINGS)


def test_capture_summary_verifier_rejects_artifact_path_escape(tmp_path):
    capture = tmp_path / "capture"
    summary = write_complete_capture(capture)
    outside = tmp_path / "outside.png"
    outside.write_bytes((capture / "rgb/0001.png").read_bytes())
    summary["frames"][0]["artifacts"]["rgb"] = {
        "path": "../outside.png",
        "sha256": _sha256(outside),
    }
    (capture / "summary.json").write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="relative|escape"):
        render_process.verify_capture_summary(capture, REQUESTED_SETTINGS)


def write_experiment_manifest(path: Path, run_id: str = "run-a1") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "hypothesis": "The reviewed renderer emits a mechanically valid capture.",
                "falsifier": "Any lifecycle or capture gate fails.",
                "command_plan": {
                    "entry_point": "render_twin_dataset.py",
                    "controls": "fixed FillCam",
                },
                "seed": 20261024,
                "denominators": {"requested_frames": 1},
                "stop_conditions": ["one process attempt", "timeout"],
            },
            sort_keys=True,
        )
        + "\n"
    )


def test_experiment_manifest_snapshot_accepts_bound_preregistration(tmp_path):
    path = tmp_path / "experiment.json"
    write_experiment_manifest(path)

    snapshot = render_process.load_experiment_manifest(path, "run-a1")

    assert snapshot["path"] == str(path.resolve())
    assert len(snapshot["sha256"]) == 64
    assert snapshot["payload"]["run_id"] == "run-a1"


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "run_id": "run-a1"},
        {"schema_version": 1, "run_id": "other"},
        {
            "schema_version": 1,
            "run_id": "run-a1",
            "hypothesis": "",
            "falsifier": "f",
            "command_plan": ["c"],
            "seed": 0,
            "denominators": {"frames": 1},
            "stop_conditions": ["stop"],
        },
    ],
)
def test_experiment_manifest_rejects_schema_run_or_empty_required_field(
    tmp_path, payload
):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        render_process.load_experiment_manifest(path, "run-a1")


def test_experiment_manifest_rejects_duplicate_keys_and_symlink(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":1,"run_id":"run-a1","run_id":"run-a1",'
        '"hypothesis":"h","falsifier":"f","command_plan":["c"],'
        '"seed":0,"denominators":{"frames":1},"stop_conditions":["s"]}'
    )
    with pytest.raises(ValueError, match="duplicate"):
        render_process.load_experiment_manifest(path, "run-a1")

    valid = tmp_path / "valid.json"
    write_experiment_manifest(valid)
    linked = tmp_path / "linked.json"
    linked.symlink_to(valid)
    with pytest.raises(ValueError, match="non-symlink"):
        render_process.load_experiment_manifest(linked, "run-a1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hypothesis", "   "),
        ("hypothesis", ["not", "a", "string"]),
        ("falsifier", ""),
        ("command_plan", []),
        ("command_plan", {}),
        ("command_plan", ["fixed FillCam"]),
        ("seed", True),
        ("seed", -1),
        ("denominators", {}),
        ("denominators", [1]),
        ("denominators", {"frames": 0}),
        ("denominators", {"frames": True}),
        ("stop_conditions", []),
        ("stop_conditions", [" "]),
        ("stop_conditions", "one attempt"),
    ],
)
def test_experiment_manifest_requires_concrete_field_types_and_content(
    tmp_path, field, value
):
    path = tmp_path / "experiment.json"
    write_experiment_manifest(path)
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=field):
        render_process.load_experiment_manifest(path, "run-a1")
