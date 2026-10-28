"""Deterministic decisions from already measured synthetic stage evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.stage_decision import (  # noqa: E402
    FrozenStageDecisionConfig,
    StageReference,
    StageStatistic,
    decide_vial_stages,
)


CONFIG = FrozenStageDecisionConfig(
    config_id="synthetic-stage-thresholds-v1",
    minimum_peer_count=2,
    minimum_distinct_reference_batches=2,
    tight_cluster_max_diameter_delta_e76=2.0,
    threshold_floor_delta_e76=2.0,
    threshold_margin_delta_e76=1.0,
)


def _stat(
    token: str,
    lab: tuple[float, float, float] | None,
    *,
    stage_ordinal: int = 1,
    batch_id: str = "TARGET-BATCH",
    vial_id: str = "V1",
    recipe_id: str = "S0117",
    split_role: str = "train",
    split_version: str = "inn-7-splits-v2",
    validity: str = "valid",
    limit_codes: tuple[str, ...] = (),
) -> StageStatistic:
    return StageStatistic(
        evidence_id=f"EVID-{token}",
        measurement_id=f"MEAS-{token}",
        image_id=f"IMG-{token}",
        batch_id=batch_id,
        vial_id=vial_id,
        recipe_id=recipe_id,
        data_origin="twin",
        split_role=split_role,
        split_version=split_version,
        stage_id=f"after-stage-{stage_ordinal}",
        stage_ordinal=stage_ordinal,
        lab=lab,
        validity=validity,
        limit_codes=limit_codes,
    )


def _targets(value: float, *, vial_id: str = "V1") -> list[StageStatistic]:
    return [
        _stat(f"TARGET-{vial_id}-S{ordinal}", (value, 0.0, 0.0), stage_ordinal=ordinal, vial_id=vial_id)
        for ordinal in (1, 2, 3)
    ]


def _references(targets: list[StageStatistic]) -> dict[str, list[StageReference]]:
    return {
        target.stage_id: [
            StageReference(
                _stat(
                    f"REF-A-S{target.stage_ordinal}",
                    (0.0, 0.0, 0.0),
                    stage_ordinal=target.stage_ordinal,
                    batch_id="REFERENCE-A",
                    vial_id="V2",
                ),
                "reference_bank",
            ),
            StageReference(
                _stat(
                    f"REF-B-S{target.stage_ordinal}",
                    (0.5, 0.0, 0.0),
                    stage_ordinal=target.stage_ordinal,
                    batch_id="REFERENCE-B",
                    vial_id="V3",
                ),
                "reference_bank",
            ),
        ]
        for target in targets
    }


@pytest.mark.parametrize(
    ("target_value", "expected"),
    [(0.25, "keep"), (3.0, "discard")],
)
def test_cross_batch_stage_numbers_drive_grounded_keep_or_discard(
    target_value: float,
    expected: str,
) -> None:
    targets = _targets(target_value)

    result = decide_vial_stages(
        "DEC-TARGET-V1", targets, _references(targets), CONFIG
    )

    assert result["decision_id"] == "DEC-TARGET-V1"
    assert result["config_id"] == CONFIG.config_id
    assert result["disposition"] == expected
    assert result["disposition_source"] == (
        "deterministic_pipeline_stage_colour_evidence"
    )
    assert result["target"] == {
        "batch_id": "TARGET-BATCH",
        "vial_id": "V1",
        "recipe_id": "S0117",
        "data_origin": "twin",
        "split_role": "train",
        "split_version": "inn-7-splits-v2",
    }
    assert result["limit_codes"] == []
    assert result["decision_timing"] == (
        "retrospective_after_complete_three_vial_batch_no_latency_claim"
    )
    assert result["real_data_used"] is False
    assert "Synthetic" in result["claim_boundary"]
    assert len(result["stages"]) == 3
    for target, stage in zip(targets, result["stages"], strict=True):
        assert stage["evidence_id"] == target.evidence_id
        assert stage["measurement_id"] == target.measurement_id
        assert stage["target_image_id"] == target.image_id
        assert stage["unit"] == "delta_e76"
        assert stage["validity"] == "valid"
        assert stage["target_delta_e76"] == pytest.approx(
            abs(target_value - 0.25)
        )
        assert stage["threshold_delta_e76"] == pytest.approx(2.0)
        assert stage["reference"]["peer_evidence_ids"] == [
            f"EVID-REF-A-S{target.stage_ordinal}",
            f"EVID-REF-B-S{target.stage_ordinal}",
        ]
        assert stage["reference"]["reference_batch_ids"] == [
            "REFERENCE-A",
            "REFERENCE-B",
        ]
        assert stage["reference"]["pairwise_diameter_delta_e76"] == pytest.approx(
            0.5
        )
        assert stage["stage_outlier"] is (expected == "discard")
    assert "action" not in result
    assert "operator" not in result


def test_same_batch_only_context_abstains() -> None:
    targets = _targets(0.25)
    references = {
        target.stage_id: [
            StageReference(
                _stat(
                    f"SAME-V2-S{target.stage_ordinal}",
                    (0.0, 0.0, 0.0),
                    stage_ordinal=target.stage_ordinal,
                    batch_id=target.batch_id,
                    vial_id="V2",
                ),
                "later_same_batch",
            ),
            StageReference(
                _stat(
                    f"SAME-V3-S{target.stage_ordinal}",
                    (0.5, 0.0, 0.0),
                    stage_ordinal=target.stage_ordinal,
                    batch_id=target.batch_id,
                    vial_id="V3",
                ),
                "later_same_batch",
            ),
        ]
        for target in targets
    }

    result = decide_vial_stages("DEC-SAME-BATCH", targets, references, CONFIG)

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == [
        "independent_reference_batches_below_minimum"
    ]
    assert all(
        stage["validity"] == "insufficient_evidence"
        and stage["target_delta_e76"] is None
        for stage in result["stages"]
    )


def test_one_reference_batch_plus_same_batch_context_still_abstains() -> None:
    targets = _targets(0.25)
    references = _references(targets)
    for target in targets:
        references[target.stage_id] = [
            references[target.stage_id][0],
            StageReference(
                _stat(
                    f"SAME-V2-S{target.stage_ordinal}",
                    (0.5, 0.0, 0.0),
                    stage_ordinal=target.stage_ordinal,
                    batch_id=target.batch_id,
                    vial_id="V2",
                ),
                "later_same_batch",
            ),
        ]

    result = decide_vial_stages("DEC-ONE-BANK", targets, references, CONFIG)

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == [
        "independent_reference_batches_below_minimum"
    ]


def test_same_batch_vials_are_optional_context_not_reference_independence() -> None:
    targets = _targets(0.25)
    references = _references(targets)
    for target in targets:
        references[target.stage_id].append(
            StageReference(
                _stat(
                    f"SAME-V3-S{target.stage_ordinal}",
                    (0.25, 0.0, 0.0),
                    stage_ordinal=target.stage_ordinal,
                    batch_id=target.batch_id,
                    vial_id="V3",
                ),
                "later_same_batch",
            )
        )

    result = decide_vial_stages("DEC-WITH-CONTEXT", targets, references, CONFIG)

    assert result["disposition"] == "keep"
    for stage in result["stages"]:
        assert stage["reference"]["reference_batch_count"] == 2
        assert stage["reference"]["same_batch_context_count"] == 1


def test_conflicting_reference_modes_abstain() -> None:
    targets = _targets(0.1)
    references: dict[str, list[StageReference]] = {}
    for target in targets:
        references[target.stage_id] = [
            StageReference(
                _stat(
                    f"MODE-{batch}-S{target.stage_ordinal}",
                    (value, 0.0, 0.0),
                    stage_ordinal=target.stage_ordinal,
                    batch_id=f"REFERENCE-{batch}",
                ),
                "reference_bank",
            )
            for batch, value in (("A", 0.0), ("B", 0.2), ("C", 10.0), ("D", 10.2))
        ]

    result = decide_vial_stages("DEC-CONFLICT", targets, references, CONFIG)

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == ["reference_mode_not_unique_or_too_small"]


def test_invalid_reference_is_named_and_blocks_disposition() -> None:
    targets = _targets(0.25)
    references = _references(targets)
    first = targets[0]
    references[first.stage_id].append(
        StageReference(
            _stat(
                "REF-INVALID-S1",
                None,
                batch_id="REFERENCE-C",
                validity="invalid",
                limit_codes=("liquid_mask_empty",),
            ),
            "reference_bank",
        )
    )

    result = decide_vial_stages("DEC-INVALID-REFERENCE", targets, references, CONFIG)

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == [
        "reference_measurement_invalid:EVID-REF-INVALID-S1",
        "reference:EVID-REF-INVALID-S1:liquid_mask_empty",
    ]
    assert result["stages"][0]["reference"] is None


def test_invalid_target_stage_abstains_with_measurement_reason() -> None:
    targets = _targets(0.25)
    bad = targets[1]
    targets[1] = replace(
        bad,
        lab=None,
        validity="invalid",
        limit_codes=("liquid_mask_misaligned",),
    )

    result = decide_vial_stages(
        "DEC-INVALID-TARGET", targets, _references(targets), CONFIG
    )

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == [
        "target_measurement_invalid",
        "liquid_mask_misaligned",
    ]
    assert result["stages"][1]["evidence_id"] == bad.evidence_id
    assert result["stages"][1]["target_delta_e76"] is None


@pytest.mark.parametrize("vial_id", ["V1", "V2", "V3"])
def test_any_vial_position_can_be_discarded_from_the_same_evidence(
    vial_id: str,
) -> None:
    targets = _targets(3.0, vial_id=vial_id)

    result = decide_vial_stages(
        f"DEC-{vial_id}", targets, _references(targets), CONFIG
    )

    assert result["disposition"] == "discard"
    assert all(stage["stage_outlier"] is True for stage in result["stages"])
    assert "scheduled_clean" not in result
    assert "vial_position" not in result


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_id", "OTHER-BATCH"),
        ("vial_id", "OTHER-VIAL"),
        ("recipe_id", "OTHER-RECIPE"),
        ("data_origin", "real"),
        ("split_role", "test"),
        ("split_version", "other-split-v1"),
    ],
)
def test_target_stages_cannot_cross_run_or_split_boundaries(
    field: str,
    value: str,
) -> None:
    targets = _targets(0.25)
    targets[1] = replace(targets[1], **{field: value})

    with pytest.raises(ValueError, match="single vial and partition"):
        decide_vial_stages(
            "DEC-MIXED-CONTEXT", targets, _references(targets), CONFIG
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recipe_id", "OTHER-RECIPE"),
        ("data_origin", "real"),
        ("split_role", "validation"),
        ("split_version", "other-split-v1"),
        ("stage_ordinal", 2),
    ],
)
def test_cross_partition_or_stage_reference_is_never_used(
    field: str,
    value: object,
) -> None:
    targets = _targets(0.25)
    references = _references(targets)
    first_reference = references[targets[0].stage_id][0]
    references[targets[0].stage_id][0] = replace(
        first_reference,
        statistic=replace(first_reference.statistic, **{field: value}),
    )

    result = decide_vial_stages(
        "DEC-REFERENCE-LEAK", targets, references, CONFIG
    )

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == [
        "reference_context_mismatch:EVID-REF-A-S1"
    ]


def test_reference_bank_cannot_reuse_target_batch() -> None:
    targets = _targets(0.25)
    references = _references(targets)
    first_reference = references[targets[0].stage_id][0]
    references[targets[0].stage_id][0] = replace(
        first_reference,
        statistic=replace(first_reference.statistic, batch_id="TARGET-BATCH"),
    )

    result = decide_vial_stages(
        "DEC-BATCH-LEAK", targets, references, CONFIG
    )

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == [
        "reference_bank_reuses_target_batch:EVID-REF-A-S1"
    ]


def test_same_batch_context_cannot_imply_more_than_three_vials() -> None:
    targets = _targets(0.25)
    references = _references(targets)
    first = targets[0]
    for vial_id, value in (("V2", 0.1), ("V3", 0.2), ("V4", 0.3)):
        references[first.stage_id].append(
            StageReference(
                _stat(
                    f"SAME-{vial_id}-S1",
                    (value, 0.0, 0.0),
                    batch_id=first.batch_id,
                    vial_id=vial_id,
                ),
                "later_same_batch",
            )
        )

    result = decide_vial_stages(
        "DEC-OVERSIZED-BATCH", targets, references, CONFIG
    )

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == ["same_batch_context_exceeds_two_vials"]


def test_two_active_fill_stages_can_drive_a_grounded_decision() -> None:
    targets = [
        _stat(f"TARGET-S{ordinal}", (0.25, 0.0, 0.0), stage_ordinal=ordinal)
        for ordinal in (1, 2)
    ]

    result = decide_vial_stages(
        "DEC-TWO-ACTIVE-STAGES", targets, _references(targets), CONFIG
    )

    assert result["disposition"] == "keep"
    assert [stage["stage_ordinal"] for stage in result["stages"]] == [1, 2]


@pytest.mark.parametrize("kept_ordinals", [(1,), (1, 3), (1, 2, 4), (1, 2, 3, 4)])
def test_vial_decision_requires_two_or_three_contiguous_fill_stages(
    kept_ordinals: tuple[int, ...],
) -> None:
    targets = [
        _stat(f"TARGET-S{ordinal}", (0.25, 0.0, 0.0), stage_ordinal=ordinal)
        for ordinal in kept_ordinals
    ]

    with pytest.raises(ValueError, match="two or three contiguous"):
        decide_vial_stages(
            "DEC-MISSING-STAGE", targets, _references(targets), CONFIG
        )


@pytest.mark.parametrize("field", ["measurement_id", "image_id"])
def test_reused_reference_measurement_or_image_cannot_inflate_peer_count(
    field: str,
) -> None:
    targets = _targets(0.25)
    references = _references(targets)
    stage_references = references[targets[0].stage_id]
    stage_references[1] = replace(
        stage_references[1],
        statistic=replace(
            stage_references[1].statistic,
            **{field: getattr(stage_references[0].statistic, field)},
        ),
    )

    result = decide_vial_stages("DEC-DUPLICATE", targets, references, CONFIG)

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == ["duplicate_reference_identity"]


@pytest.mark.parametrize("field", ["evidence_id", "measurement_id", "image_id"])
def test_target_stage_evidence_identities_are_unique(field: str) -> None:
    targets = _targets(0.25)
    targets[1] = replace(targets[1], **{field: getattr(targets[0], field)})

    with pytest.raises(ValueError, match="target evidence identities must be unique"):
        decide_vial_stages(
            "DEC-TARGET-DUPLICATE", targets, _references(targets), CONFIG
        )


@pytest.mark.parametrize("field", ["evidence_id", "measurement_id", "image_id"])
def test_reference_cannot_reuse_any_target_stage_identity(field: str) -> None:
    targets = _targets(0.25)
    references = _references(targets)
    first_reference = references[targets[0].stage_id][0]
    references[targets[0].stage_id][0] = replace(
        first_reference,
        statistic=replace(
            first_reference.statistic,
            **{field: getattr(targets[1], field)},
        ),
    )

    result = decide_vial_stages(
        "DEC-TARGET-REFERENCE-REUSE", targets, references, CONFIG
    )

    assert result["disposition"] == "insufficient_evidence"
    assert result["limit_codes"] == ["target_reused_as_reference"]


def test_single_departing_fill_stage_is_preserved_in_discard_evidence() -> None:
    targets = [
        _stat(
            f"TARGET-V1-S{ordinal}",
            (value, 0.0, 0.0),
            stage_ordinal=ordinal,
        )
        for ordinal, value in ((1, 0.25), (2, 3.0), (3, 0.25))
    ]

    result = decide_vial_stages(
        "DEC-STAGE2-ONLY", targets, _references(targets), CONFIG
    )

    assert result["disposition"] == "discard"
    assert [stage["stage_outlier"] for stage in result["stages"]] == [
        False,
        True,
        False,
    ]
