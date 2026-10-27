"""Execution boundary for the 72-row Varied V3 Cosmos matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


CONDITIONS = (
    "numbers_only",
    "ordered_stage_visual_pairs",
    "irrelevant_shuffled_images",
    "deterministic_template",
)


def _plan() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for decision_index in range(18):
        decision_id = f"DEC-BATCH-{decision_index:02d}-V1"
        for condition_index, condition in enumerate(CONDITIONS):
            control = condition == "deterministic_template"
            prompt = None if control else f"Prompt {decision_index} {condition}"
            template = (
                "Evidence: UNKNOWN | evidence is not directional\n"
                "Uncertainty: ABSTAIN | clean reference is unverified\n"
                "Action: COLLECT_EVIDENCE | collect independent reference evidence"
                if control
                else None
            )
            images = [] if condition in {"numbers_only", "deterministic_template"} else [
                f"IMG-{decision_index:02d}-STAGE-1-TARGET",
                f"IMG-{decision_index:02d}-STAGE-1-REFERENCE",
                f"IMG-{decision_index:02d}-STAGE-2-TARGET",
                f"IMG-{decision_index:02d}-STAGE-2-REFERENCE",
            ]
            captions = [] if not images else [
                "Stage 1 target image.",
                "Stage 1 reference image.",
                "Stage 2 target image.",
                "Stage 2 reference image.",
            ]
            rows.append(
                {
                    "attempt_index": decision_index * 4 + condition_index,
                    "attempt_id": f"{condition}-{decision_id}",
                    "decision_id": decision_id,
                    "run_id": f"RUN-{decision_index // 3:02d}",
                    "split_role": "train",
                    "status": "not_run",
                    "condition_id": condition,
                    "kind": "control" if control else "model",
                    "prompt": prompt,
                    "prompt_sha256": (
                        None
                        if prompt is None
                        else hashlib.sha256(prompt.encode()).hexdigest()
                    ),
                    "evidence_record_sha256": f"{decision_index + 1:064x}",
                    "image_ids": images,
                    "image_captions": captions,
                    "image_count": len(images),
                    "irrelevant_source_decision_ids": (
                        [f"DEC-BATCH-{(decision_index + 3) % 18:02d}-V1"]
                        if condition == "irrelevant_shuffled_images"
                        else []
                    ),
                    "irrelevant_source_run_ids": (
                        [f"RUN-{((decision_index + 3) % 18) // 3:02d}"]
                        if condition == "irrelevant_shuffled_images"
                        else []
                    ),
                    "requires_multi_image_smoke": bool(images),
                    "template_response": template,
                }
            )
    return rows


def _completion(raw: str) -> dict[str, object]:
    return {
        "response_raw": raw,
        "response_utf8_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "generated_suffix_token_ids": [1, 2, 3],
        "generated_suffix_sha256": hashlib.sha256(b"[1,2,3]").hexdigest(),
        "generated_token_count": 3,
        "terminal_token_id": 3,
        "terminal_token_is_eos": False,
        "length_cap_reached": False,
        "truncated_without_eos": False,
    }


def test_executes_54_model_attempts_in_condition_batches_and_retains_18_controls():
    from fluidmix.v3_cosmos_runtime import execute_v3_attempt_plan

    calls: list[tuple[str, list[int]]] = []

    def infer_batch(specs):
        calls.append(
            (
                specs[0]["condition_id"],
                [spec["attempt_index"] for spec in specs],
            )
        )
        return [_completion(f"messy raw {spec['attempt_id']}") for spec in specs]

    rows = execute_v3_attempt_plan(_plan(), infer_batch=infer_batch, batch_size=6)

    assert len(rows) == 72
    assert [row["attempt_index"] for row in rows] == list(range(72))
    assert sum(row["kind"] == "model" for row in rows) == 54
    assert sum(row["kind"] == "control" for row in rows) == 18
    assert sum(row["attempt_status"] == "completed" for row in rows) == 72
    assert calls == [
        (condition, [decision * 4 + offset for decision in range(start, start + 6)])
        for condition, offset in zip(CONDITIONS[:3], range(3), strict=True)
        for start in (0, 6, 12)
    ]
    controls = [row for row in rows if row["kind"] == "control"]
    assert all(row["generated_suffix_token_ids"] is None for row in controls)
    assert all(row["response_raw"].startswith("Evidence: UNKNOWN") for row in controls)
    # Raw responses are retained without parsing or repair; scoring is separate.
    assert rows[0]["response_raw"].startswith("messy raw")


def test_batch_exception_and_model_load_error_remain_in_fixed_denominator():
    from fluidmix.v3_cosmos_runtime import execute_v3_attempt_plan

    calls = 0

    def partly_failing(specs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("one bounded batch failed")
        return [_completion("unstructured answer") for _ in specs]

    rows = execute_v3_attempt_plan(_plan(), infer_batch=partly_failing, batch_size=6)
    assert len(rows) == 72
    assert sum(row["attempt_status"] == "error" for row in rows) == 6
    assert all(
        row["response_raw"] is None
        for row in rows
        if row["attempt_status"] == "error"
    )

    load_error_rows = execute_v3_attempt_plan(
        _plan(), model_load_error={"type": "LoadError", "message": "no model"}
    )
    assert sum(row["attempt_status"] == "error" for row in load_error_rows) == 54
    assert sum(row["attempt_status"] == "completed" for row in load_error_rows) == 18


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda rows: rows.pop(), "exactly 72"),
        (
            lambda rows: rows[0].update(condition_id="ordered_stage_visual_pairs"),
            "decision-major",
        ),
        (
            lambda rows: rows[4].update(decision_id=rows[0]["decision_id"]),
            "decision identities",
        ),
        (
            lambda rows: rows[1].update(image_count=99),
            "metadata",
        ),
    ],
)
def test_plan_integrity_fails_closed_before_inference(mutate, match):
    from fluidmix.v3_cosmos_runtime import execute_v3_attempt_plan

    rows = _plan()
    mutate(rows)
    called = False

    def infer(_specs):
        nonlocal called
        called = True
        return []

    with pytest.raises(ValueError, match=match):
        execute_v3_attempt_plan(rows, infer_batch=infer, batch_size=6)
    assert called is False


def test_batch_size_must_divide_eighteen():
    from fluidmix.v3_cosmos_runtime import execute_v3_attempt_plan

    with pytest.raises(ValueError, match="positive divisor of 18"):
        execute_v3_attempt_plan(_plan(), infer_batch=lambda specs: [], batch_size=4)
