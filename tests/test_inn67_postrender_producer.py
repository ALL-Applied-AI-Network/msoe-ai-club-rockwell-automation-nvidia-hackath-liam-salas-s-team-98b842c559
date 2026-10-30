from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image


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


def _write_rgb(
    path: Path,
    color: tuple[int, int, int],
    marker: tuple[int, int, int] | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 480), color=color)
    if marker is not None:
        image.putpixel((0, 0), marker)
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


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
        render_plan = bundle[subrun["render_plan_artifact"]]
        schedules = {
            row["case_binding"]["case_id"]: row
            for row in render_plan["execution_schedules"]
        }
        results = []
        plan_hash = _canonical_sha(render_plan)
        for case in cases:
            renderer_id = case["renderer_case_id"]
            schedule = schedules[renderer_id]
            app_output = subrun_root / "cases" / renderer_id / "app-output"
            prefill_path = "pre-pour-parity/f2-prefill-time-0001.png"
            prefill_sha = _write_rgb(app_output / prefill_path, (5, 7, 11))
            pairs = []
            for stage in case["stages"]:
                rgb_path = f"frames/stage-{stage['ordinal']:02d}-{stage['tank_id']}.png"
                rgb_sha = _write_rgb(
                    app_output / rgb_path,
                    (
                        30 + int(case["recipe_id"][-2:]),
                        40 + stage["ordinal"],
                        50,
                    ),
                    marker=(int(renderer_id[-2:], 16), stage["ordinal"], 1),
                )
                pairs.append(
                    {
                        "phase": f"{stage['tank_id']}-stage",
                        "rgb_path": rgb_path,
                        "rgb_sha256": rgb_sha,
                    }
                )
            receipt = {
                "schema_version": 20,
                "status": "pass",
                "arguments": {
                    "train_case_id": renderer_id,
                    "train_render_plan_sha256": plan_hash,
                },
                "execution_schedule": {
                    "case_id": renderer_id,
                    "cohort_plan_sha256": plan_hash,
                    "model_input_record": {
                        "batch_id": schedule["case_binding"]["batch_id"],
                        "case_id": renderer_id,
                        "current_programmed_ml": case["programmed_ml"],
                        "current_recipe_id": case["recipe_id"],
                        "data_origin": "twin",
                        "programmed_dispense_order": case["dispense_order"],
                        "previous_programmed_ml": schedule["model_input_record"][
                            "previous_programmed_ml"
                        ],
                        "previous_recipe_id": schedule["model_input_record"][
                            "previous_recipe_id"
                        ],
                        "run_id": case["run_id"],
                        "split_role": case["split_role"],
                        "split_version": "inn67-development-reference-bank-v1",
                        "vial_position": schedule["case_binding"]["vial_ordinal"],
                    },
                    "renderer_case_binding": schedule["case_binding"],
                },
                "pre_pour_parity": {
                    "f2_prefill_path": prefill_path,
                    "f2_prefill_sha256": prefill_sha,
                },
                "semantics": {"pairs": pairs},
            }
            _write(app_output / "receipt.json", receipt)
            results.append(
                {
                    "case_id": renderer_id,
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
    first_case = inventory["subruns"][0]["cases"][0]
    assert set(first_case) == {
        "renderer_case_id", "receipt_path", "receipt_sha256",
        "validation_receipt_path", "validation_receipt_sha256",
        "prefill_rgb_path", "prefill_rgb_sha256", "stage_rgbs",
    }
    assert len(first_case["stage_rgbs"]) in {2, 3}
    assert all(set(row) == {"stage_ordinal", "tank_id", "rgb_path", "rgb_sha256"}
               for row in first_case["stage_rgbs"])
    frozen = {
        "plan": plan,
        "inventory": json.loads((plan_root / "artifact-hashes.json").read_text()),
        "render_plan_hashes": {
            subrun["subrun_id"]: _canonical_sha(
                build_reference_bank_bundle(ROOT)[subrun["render_plan_artifact"]]
            )
            for subrun in plan["subruns"]
        },
        "render_schedules": {
            subrun["subrun_id"]: {
                row["case_binding"]["case_id"]: row
                for row in build_reference_bank_bundle(ROOT)[
                    subrun["render_plan_artifact"]
                ]["execution_schedules"]
            }
            for subrun in plan["subruns"]
        },
    }
    accepted = _validate_render_inventory(output / "render-inventory.json", frozen)
    assert set(accepted) == {row["case_id"] for row in plan["cases"]} | {
        "__inventory__"
    }


def test_failed_publication_removes_owned_partial_output_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan_root, render_root, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        "fluidmix.inn67_postrender_producer._load_canonical_validator",
        lambda _path: _passing_validator,
    )
    output = render_root / "accepted"
    original_open = Path.open
    wrapper_writes = 0

    def fail_after_first_wrapper(path: Path, mode: str = "r", *args, **kwargs):
        nonlocal wrapper_writes
        if path.parent == output / "validation" and mode == "xb":
            wrapper_writes += 1
            if wrapper_writes == 2:
                raise OSError("synthetic publication failure")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as publication_failure:
        publication_failure.setattr(Path, "open", fail_after_first_wrapper)
        with pytest.raises(PostrenderProductionError, match="publication_failed"):
            produce_postrender_inputs(
                plan_root=plan_root,
                render_root=render_root,
                output_dir=output,
            )

    assert wrapper_writes == 2
    assert not output.exists()
    retried = produce_postrender_inputs(
        plan_root=plan_root,
        render_root=render_root,
        output_dir=output,
    )
    assert retried["status"] == "published"
    assert (output / "render-inventory.json").is_file()


@pytest.mark.parametrize(
    "failure",
    [
        "missing", "duplicate", "nonpass", "partial",
        "schema", "artifact_type", "argument_case", "argument_plan",
        "schedule_case", "schedule_plan",
        "model_case", "renderer_case", "run", "batch", "recipe",
        "prefill_hash", "stage_hash", "stage_phase",
    ],
)
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
    elif failure == "schema":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["schema_version"] = 14
        _write(first_case / "receipt.json", receipt)
    elif failure == "artifact_type":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["artifact_type"] = None
        _write(first_case / "receipt.json", receipt)
    elif failure == "argument_case":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["arguments"]["train_case_id"] = "wrong"
        _write(first_case / "receipt.json", receipt)
    elif failure == "argument_plan":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["arguments"]["train_render_plan_sha256"] = "0" * 64
        _write(first_case / "receipt.json", receipt)
    elif failure == "schedule_case":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["execution_schedule"]["case_id"] = "wrong"
        _write(first_case / "receipt.json", receipt)
    elif failure == "schedule_plan":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["execution_schedule"]["cohort_plan_sha256"] = "0" * 64
        _write(first_case / "receipt.json", receipt)
    elif failure == "model_case":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["execution_schedule"]["model_input_record"]["case_id"] = "wrong"
        _write(first_case / "receipt.json", receipt)
    elif failure == "renderer_case":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["execution_schedule"]["renderer_case_binding"]["case_id"] = "wrong"
        _write(first_case / "receipt.json", receipt)
    elif failure == "run":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["execution_schedule"]["model_input_record"]["run_id"] = "wrong"
        _write(first_case / "receipt.json", receipt)
    elif failure == "batch":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["execution_schedule"]["model_input_record"]["batch_id"] = "wrong"
        _write(first_case / "receipt.json", receipt)
    elif failure == "recipe":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["execution_schedule"]["renderer_case_binding"]["current_recipe_id"] = "wrong"
        _write(first_case / "receipt.json", receipt)
    elif failure == "prefill_hash":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["pre_pour_parity"]["f2_prefill_sha256"] = "0" * 64
        _write(first_case / "receipt.json", receipt)
    elif failure == "stage_hash":
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["semantics"]["pairs"][0]["rgb_sha256"] = "0" * 64
        _write(first_case / "receipt.json", receipt)
    else:
        receipt = json.loads((first_case / "receipt.json").read_text())
        receipt["semantics"]["pairs"][0]["phase"] = "wrong-stage"
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
    sentinel = output / "caller-owned.txt"
    sentinel.write_text("preserve me\n")
    with pytest.raises(PostrenderProductionError, match="output_exists"):
        produce_postrender_inputs(plan_root=plan_root, render_root=render_root, output_dir=output)
    assert sentinel.read_text() == "preserve me\n"

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
