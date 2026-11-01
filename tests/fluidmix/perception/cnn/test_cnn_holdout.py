"""Partition and acceptance contracts for the parallel WP5 CNN candidate."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def _source_plan() -> tuple[dict[str, object], str]:
    from twin.cohorts.varied_cohort_manifest_v4 import build_varied_cohort_artifacts

    plan = build_varied_cohort_artifacts(repo_root=ROOT)["render-schedule-plan.json"]
    digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return plan, digest


def test_preregister_selects_three_whole_v4_evaluation_runs_metadata_only():
    from fluidmix.perception.cnn.cnn_holdout import build_preregistered_holdout

    plan, digest = _source_plan()
    artifact = build_preregistered_holdout(
        plan,
        source_plan_sha256=digest,
        consumed_case_ids={
            "CASE_9f879a382a0141bd",
            "CASE_f1cf37594eb6c51e",
            "CASE_b04ce36f000d794d",
        },
        forbidden_run_ids={
            "RUN-EVALUATION-01",
            "RUN-VALIDATION-01",
            "RUN-V4-TRAIN-01",
            "RUN-V4-TRAIN-02",
            "RUN-V4-TRAIN-03",
            "RUN-V4-TRAIN-04",
            "RUN-V4-VALIDATION-01",
            "RUN-V4-VALIDATION-02",
            "RUN-V4-VALIDATION-03",
            "WP1-TEST-RUN-01",
            "WP1-TEST-RUN-02",
            "video-5",
        },
    )

    assert artifact["run_ids"] == [
        "RUN-V4-EVALUATION-01",
        "RUN-V4-EVALUATION-02",
        "RUN-V4-EVALUATION-03",
    ]
    assert artifact["case_ids"] == [
        "CASE_7e6cd06afd76f00c",
        "CASE_6b596ca7925365fd",
        "CASE_596eb298b2a2a27c",
        "CASE_ddb9b9a7e73d469e",
        "CASE_dd322018dfe9e896",
        "CASE_e6c1137b333626c8",
        "CASE_ceb7daefd84116f3",
        "CASE_ce1fbc40bb68407f",
        "CASE_ff720aaa5b347eda",
    ]
    assert artifact["render_seeds"] == [
        520801,
        520802,
        520803,
        520901,
        520902,
        520903,
        521001,
        521002,
        521003,
    ]
    assert artifact["pixel_access_before_candidate_freeze"] is False
    assert artifact["media_open_count_at_preregistration"] == 0
    assert artifact["source_plan_sha256"] == digest
    encoded = json.dumps(artifact, sort_keys=True).lower()
    assert not any(
        forbidden in encoded
        for forbidden in ("rgb_path", "mask_path", "video_path", "semantic_path")
    )


def test_preregister_rejects_overlap_and_non_whole_run_selection():
    from fluidmix.perception.cnn.cnn_holdout import HoldoutRegistrationError, build_preregistered_holdout

    plan, digest = _source_plan()
    with pytest.raises(HoldoutRegistrationError, match="forbidden_run_overlap"):
        build_preregistered_holdout(
            plan,
            source_plan_sha256=digest,
            consumed_case_ids=set(),
            forbidden_run_ids={"RUN-V4-EVALUATION-02"},
        )

    broken = deepcopy(plan)
    broken["batches"][-1]["case_ids"] = broken["batches"][-1]["case_ids"][:2]
    with pytest.raises(HoldoutRegistrationError, match="whole_run_case_mismatch"):
        build_preregistered_holdout(
            broken,
            source_plan_sha256=digest,
            consumed_case_ids=set(),
            forbidden_run_ids=set(),
        )


def test_preregister_rejects_media_fields_even_when_nested():
    from fluidmix.perception.cnn.cnn_holdout import HoldoutRegistrationError, build_preregistered_holdout

    plan, digest = _source_plan()
    plan["runs"][-1]["unopened"] = {"rgb_path": "/sealed/do-not-open.png"}

    with pytest.raises(HoldoutRegistrationError, match="media_field_present"):
        build_preregistered_holdout(
            plan,
            source_plan_sha256=digest,
            consumed_case_ids=set(),
            forbidden_run_ids=set(),
        )
