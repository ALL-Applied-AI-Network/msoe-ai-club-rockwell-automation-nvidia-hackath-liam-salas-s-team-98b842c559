from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.inn67_bank_materializer import (  # noqa: E402
    ACCEPTED_OCCLUSION_VALIDATOR_SHA256,
    _validate_render_inventory,
)
from fluidmix.inn67_postrender_producer import (  # noqa: E402
    PostrenderProductionError,
    produce_postrender_inputs,
)
from fluidmix.inn67_reference_bank import build_reference_bank_bundle  # noqa: E402


def _body(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = value if isinstance(value, bytes) else _body(value)
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    bundle = build_reference_bank_bundle(ROOT)
    plan_root = tmp_path / "plan"
    hashes: dict[str, str] = {}
    keep = {
        "reference-bank-plan.json",
        "known-good-profile.json",
        "delta-e-freeze-config.json",
        "leakage-audit.json",
        "no-access-attestation.json",
    }
    for relative, value in bundle.items():
        if relative in keep or relative.endswith("/render-schedule-plan.json"):
            hashes[relative] = _write(plan_root / relative, value)
    hashes["PLAN.md"] = "7bcaa8d7a59cc23b5fb9ab930e7bbcc8ba1dfa65af5d4762709593215e99e06f"
    hashes["resource-preflight.json"] = "fe72d39f7933bc1afcfafcc8d50cc0a79bde3e4b26fa15775d70f4ef73462902"
    for relative, value in bundle.items():
        if relative.endswith(("/label-provenance.json", "/model-input-plan.json")):
            hashes[relative] = hashlib.sha256(_body(value)).hexdigest()
    inventory = {
        "schema_version": 1,
        "artifact_type": "inn67_reference_bank_plan_hash_inventory",
        "files": hashes,
    }
    assert _write(plan_root / "artifact-hashes.json", inventory) == (
        "0b373388c040d0d01d6b8eb45f353b8cbfdeba5b5608f02a9ec5eda09f5e7f95"
    )

    plan = bundle["reference-bank-plan.json"]
    render_root = tmp_path / "renders"
    for subrun in plan["subruns"]:
        subrun_root = render_root / subrun["subrun_id"]
        cases = [row for row in plan["cases"] if row["subrun_id"] == subrun["subrun_id"]]
        results = []
        plan_hash = hashes[subrun["render_plan_artifact"]]
        for case in cases:
            receipt = {
                "schema_version": 14,
                "artifact_type": "physx_three_pour_receipt",
                "status": "pass",
                "case_id": case["renderer_case_id"],
                "source_plan_sha256": plan_hash,
                "known_good_profile_sha256": hashes["known-good-profile.json"],
            }
            _write(
                subrun_root / "cases" / case["renderer_case_id"] / "app-output" / "receipt.json",
                receipt,
            )
            results.append(
                {
                    "case_id": case["renderer_case_id"],
                    "outcome": "passed",
                    "evidence": {"artifact_validation": {"status": "pass", "failures": []}},
                }
            )
        _write(
            subrun_root / "supervisor-attempts" / "attempt-0001.json",
            {
                "schema_version": 1,
                "artifact_type": "physx_train_residual_serial_supervisor_attempt",
                "status": "pass",
                "plan_sha256": plan_hash,
                "planned_case_count": 3,
                "validated_completed_case_count": 3,
                "case_results": results,
            },
        )
    return plan_root, render_root, plan


def _passing_validator(_app_output: Path) -> dict[str, object]:
    return {
        "status": "pass",
        "verified_artifact_count": 7,
        "failures": [],
        "warnings": [],
        "offline_revalidations": [],
    }


def test_produces_exact_wrappers_and_inventory_only_after_all_eighteen_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan_root, render_root, plan = _fixture(tmp_path)
    calls: list[Path] = []

    def validator(path: Path) -> dict[str, object]:
        calls.append(path)
        return _passing_validator(path)

    monkeypatch.setattr(
        "fluidmix.inn67_postrender_producer._load_canonical_validator",
        lambda _path: validator,
    )
    output = render_root / "accepted"
    result = produce_postrender_inputs(
        plan_root=plan_root, render_root=render_root, output_dir=output
    )

    assert result == {
        "status": "published",
        "subrun_count": 6,
        "case_count": 18,
        "render_inventory_path": str(output / "render-inventory.json"),
    }
    assert len(calls) == 18
    assert set(path.parent.name for path in calls) == {
        row["renderer_case_id"] for row in plan["cases"]
    }
    assert {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()} == {
        "render-inventory.json",
        *{f"validation/{row['renderer_case_id']}.json" for row in plan["cases"]},
    }
    inventory = json.loads((output / "render-inventory.json").read_text())
    assert inventory["status"] == "accepted"
    assert len(inventory["subruns"]) == 6
    assert sum(len(row["cases"]) for row in inventory["subruns"]) == 18
    wrapper_path = Path(inventory["subruns"][0]["cases"][0]["validation_receipt_path"])
    wrapper = json.loads(wrapper_path.read_text())
    assert wrapper["artifact_type"] == "inn67_occlusion_aware_artifact_validation"
    assert wrapper["validator_implementation_sha256"] == ACCEPTED_OCCLUSION_VALIDATOR_SHA256
    assert wrapper["known_good_profile_sha256"] == (
        "47bc0b6f483a121a28871b3eb74ec27efb79996b83d03d8c45d6f5b0b9cdeb7c"
    )
    frozen = {
        "plan": plan,
        "inventory": json.loads((plan_root / "artifact-hashes.json").read_text()),
    }
    accepted = _validate_render_inventory(output / "render-inventory.json", frozen)
    assert set(accepted) == {row["case_id"] for row in plan["cases"]} | {
        "__inventory__"
    }


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nonpass", "partial", "hash"])
def test_fails_closed_before_publication_for_incomplete_or_unbound_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
):
    plan_root, render_root, plan = _fixture(tmp_path)
    first_subrun = render_root / plan["subruns"][0]["subrun_id"]
    first_case = first_subrun / "cases" / plan["cases"][0]["renderer_case_id"] / "app-output"
    supervisor = first_subrun / "supervisor-attempts" / "attempt-0001.json"
    if failure == "missing":
        (first_case / "receipt.json").unlink()
    elif failure == "duplicate":
        _write(first_subrun / "supervisor-attempts" / "attempt-0002.json", json.loads(supervisor.read_text()))
    elif failure == "nonpass":
        monkeypatch.setattr(
            "fluidmix.inn67_postrender_producer._load_canonical_validator",
            lambda _path: lambda _output: {
                **_passing_validator(_output), "status": "insufficient_evidence", "failures": ["bad"]
            },
        )
    elif failure == "partial":
        value = json.loads(supervisor.read_text())
        value["validated_completed_case_count"] = 2
        _write(supervisor, value)
    else:
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["known_good_profile_sha256"] = "0" * 64
        _write(first_case / "receipt.json", receipt)
    if failure != "nonpass":
        monkeypatch.setattr(
            "fluidmix.inn67_postrender_producer._load_canonical_validator",
            lambda _path: _passing_validator,
        )

    output = render_root / "accepted"
    with pytest.raises(PostrenderProductionError):
        produce_postrender_inputs(plan_root=plan_root, render_root=render_root, output_dir=output)
    assert not output.exists()


def test_rejects_symlink_out_of_root_and_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "fluidmix.inn67_postrender_producer._load_canonical_validator",
        lambda _path: _passing_validator,
    )
    plan_root, render_root, plan = _fixture(tmp_path / "symlink")
    receipt = (
        render_root / plan["subruns"][0]["subrun_id"] / "cases"
        / plan["cases"][0]["renderer_case_id"] / "app-output" / "receipt.json"
    )
    external = tmp_path / "external-receipt.json"
    external.write_bytes(receipt.read_bytes())
    receipt.unlink()
    receipt.symlink_to(external)
    with pytest.raises(PostrenderProductionError, match="symlink"):
        produce_postrender_inputs(
            plan_root=plan_root, render_root=render_root, output_dir=render_root / "symlink-output"
        )

    plan_root, render_root, _ = _fixture(tmp_path / "existing")
    output = render_root / "existing-output"
    output.mkdir()
    with pytest.raises(PostrenderProductionError, match="output_exists"):
        produce_postrender_inputs(plan_root=plan_root, render_root=render_root, output_dir=output)

    plan_root, render_root, _ = _fixture(tmp_path / "outside")
    with pytest.raises(PostrenderProductionError, match="output_must_be_new_render_root_child"):
        produce_postrender_inputs(
            plan_root=plan_root,
            render_root=render_root,
            output_dir=tmp_path / "outside-output",
        )


def test_cli_exposes_cpu_only_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plan_root, render_root, _ = _fixture(tmp_path)
    # A subprocess cannot use the fake validator, so this only proves argument and refusal wiring.
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "produce_inn67_postrender_inputs.py"),
            "--plan-root", str(plan_root),
            "--render-root", str(render_root / "missing"),
            "--output-dir", str(render_root / "output"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "INN-67 post-render production refused" in completed.stderr
