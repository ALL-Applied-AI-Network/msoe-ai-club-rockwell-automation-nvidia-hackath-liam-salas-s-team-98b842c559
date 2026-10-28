"""Public CLI contract for Varied V3 triad decisions."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "decide_f2_varied_triads.py"
CONFIG = ROOT / "data" / "f2_varied_triad_config_v1.json"
EXTERNAL_MEASURED = Path(
    "/home/hackathon5/liamreporoot/"
    "f2-varied-v3-measurements-20261027/measured-evidence.json"
)
EXTERNAL_DECISIONS = Path(
    "/home/hackathon5/liamreporoot/"
    "f2-varied-v3-decisions-20261027/triad-decisions.json"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run(measured: Path, output: Path, *, config: Path = CONFIG) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--measured-evidence",
            str(measured),
            "--config",
            str(config),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_publishes_frozen_relative_decisions(tmp_path):
    from test_f2_varied_triage import _measurement_artifact

    measured = tmp_path / "measured-evidence.json"
    output = tmp_path / "triad-decisions.json"
    _write_json(measured, _measurement_artifact())

    completed = _run(measured, output)

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["config"] == {
        "config_id": "v3-relative-diagnostic-inherited-stage-v1",
        "tight_peer_pair_max_delta_e76": 2.0,
        "target_outlier_margin_delta_e76": 1.0,
    }
    assert artifact["denominators"] == {
        "runs": 1,
        "vial_decisions": 3,
        "keep": 0,
        "discard": 0,
        "insufficient_evidence": 3,
        "relative_outlier_vials": 1,
    }
    assert "assessed 3 vials" in completed.stdout


@pytest.mark.parametrize(
    "config_payload",
    [
        {
            "config_id": "v3-relative-diagnostic-inherited-stage-v1",
            "tight_peer_pair_max_delta_e76": 2.0,
            "target_outlier_margin_delta_e76": 1.0,
            "unexpected": True,
        },
        {
            "config_id": "v3-relative-diagnostic-inherited-stage-v1",
            "tight_peer_pair_max_delta_e76": 2.0,
        },
    ],
)
def test_cli_rejects_missing_or_extra_config_fields(tmp_path, config_payload):
    from test_f2_varied_triage import _measurement_artifact

    measured = tmp_path / "measured-evidence.json"
    config = tmp_path / "config.json"
    _write_json(measured, _measurement_artifact())
    _write_json(config, config_payload)

    completed = _run(measured, tmp_path / "decisions.json", config=config)

    assert completed.returncode != 0
    assert "triad config fields are invalid" in completed.stderr
    assert not (tmp_path / "decisions.json").exists()


def test_cli_rejects_malformed_config_json(tmp_path):
    from test_f2_varied_triage import _measurement_artifact

    measured = tmp_path / "measured-evidence.json"
    config = tmp_path / "config.json"
    _write_json(measured, _measurement_artifact())
    config.write_text('{"config_id":', encoding="utf-8")

    completed = _run(measured, tmp_path / "decisions.json", config=config)

    assert completed.returncode != 0
    assert "could not load triad config" in completed.stderr
    assert not (tmp_path / "decisions.json").exists()


def test_cli_rejects_duplicate_config_keys(tmp_path):
    from test_f2_varied_triage import _measurement_artifact

    measured = tmp_path / "measured-evidence.json"
    config = tmp_path / "config.json"
    _write_json(measured, _measurement_artifact())
    config.write_text(
        '{"config_id":"v3-relative-diagnostic-inherited-stage-v1",'
        '"config_id":"shadowed",'
        '"tight_peer_pair_max_delta_e76":2.0,'
        '"target_outlier_margin_delta_e76":1.0}\n',
        encoding="utf-8",
    )

    completed = _run(measured, tmp_path / "decisions.json", config=config)

    assert completed.returncode != 0
    assert "duplicate JSON key" in completed.stderr
    assert not (tmp_path / "decisions.json").exists()


def test_cli_inherits_oracle_rejection_from_decision_boundary(tmp_path):
    from test_f2_varied_triage import _measurement_artifact

    measured = tmp_path / "measured-evidence.json"
    payload = _measurement_artifact()
    payload["stage_measurements"][0]["expected_disposition"] = "discard"
    _write_json(measured, payload)

    completed = _run(measured, tmp_path / "decisions.json")

    assert completed.returncode != 0
    assert "forbidden_deploy_input_field" in completed.stderr
    assert not (tmp_path / "decisions.json").exists()


def test_cli_refuses_to_clobber_a_published_decision_artifact(tmp_path):
    from test_f2_varied_triage import _measurement_artifact

    measured = tmp_path / "measured-evidence.json"
    output = tmp_path / "decisions.json"
    _write_json(measured, _measurement_artifact())
    first = _run(measured, output)
    assert first.returncode == 0, first.stderr
    published = output.read_bytes()

    repeated = _run(measured, output)

    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr
    assert output.read_bytes() == published
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))


@pytest.mark.skipif(
    not (EXTERNAL_MEASURED.is_file() and EXTERNAL_DECISIONS.is_file()),
    reason="accepted external Varied V3 evidence is unavailable",
)
def test_cli_byte_reproduces_the_accepted_v3_decision_artifact(tmp_path):
    output = tmp_path / "triad-decisions.json"

    completed = _run(EXTERNAL_MEASURED, output)

    assert completed.returncode == 0, completed.stderr
    assert output.read_bytes() == EXTERNAL_DECISIONS.read_bytes()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        "a3eb98df03c76ac158c44d1efe297f26b3904f71e6ac85287abc0e26179ace81"
    )
