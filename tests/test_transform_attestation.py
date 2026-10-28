from __future__ import annotations

import copy
import math

import pytest

from fluidmix.transform_attestation import (
    MATRIX_COMPARATOR_VERSION,
    attest_effective_state_matrix,
    attest_finite_f64_matrix4x4,
    verify_effective_state_matrix_attestation,
)


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def test_exact_and_one_ulp_matrix_attestations_report_all_distances() -> None:
    exact = attest_finite_f64_matrix4x4(IDENTITY, copy.deepcopy(IDENTITY))
    one_ulp = copy.deepcopy(IDENTITY)
    one_ulp[2][0] = math.nextafter(0.0, 1.0)
    adjacent = attest_finite_f64_matrix4x4(IDENTITY, one_ulp)

    assert exact["passed"] is True
    assert exact["comparator_version"] == MATRIX_COMPARATOR_VERSION
    assert exact["elementwise_ulp_distances"] == [[0] * 4 for _ in range(4)]
    assert exact["maximum_ulp_distance"] == 0
    assert adjacent["passed"] is True
    assert adjacent["elementwise_ulp_distances"][2][0] == 1
    assert adjacent["maximum_ulp_distance"] == 1
    assert adjacent["expected_matrix_f64_sha256"] != adjacent[
        "observed_matrix_f64_sha256"
    ]


def test_signed_zero_encodings_are_numerically_equal_but_hash_visible() -> None:
    observed = copy.deepcopy(IDENTITY)
    observed[0][1] = -0.0

    proof = attest_finite_f64_matrix4x4(IDENTITY, observed)

    assert proof["passed"] is True
    assert proof["elementwise_ulp_distances"][0][1] == 0
    assert proof["expected_matrix_f64_sha256"] != proof[
        "observed_matrix_f64_sha256"
    ]


@pytest.mark.parametrize(
    "observed",
    [
        [],
        [[1.0]],
        [[0.0] * 4 for _ in range(3)],
        [[0.0] * 3 for _ in range(4)],
        [[0.0] * 5 for _ in range(4)],
        [0.0] * 16,
        tuple(tuple(row) for row in IDENTITY),
    ],
)
def test_every_wrong_matrix_shape_or_container_fails(observed: object) -> None:
    proof = attest_finite_f64_matrix4x4(IDENTITY, observed)

    assert proof["passed"] is False
    assert "observed_matrix_shape_or_type_invalid" in proof["failure_reasons"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_nonfinite_or_boolean_matrix_elements_fail(value: object) -> None:
    observed = copy.deepcopy(IDENTITY)
    observed[1][2] = value

    proof = attest_finite_f64_matrix4x4(IDENTITY, observed)

    assert proof["passed"] is False
    assert "observed_matrix_element_invalid" in proof["failure_reasons"]


def test_two_ulp_and_meaningful_translation_or_rotation_fail() -> None:
    two_ulp = copy.deepcopy(IDENTITY)
    two_ulp[1][1] = math.nextafter(math.nextafter(1.0, 2.0), 2.0)
    translation = copy.deepcopy(IDENTITY)
    translation[3][0] = 0.001
    rotation = copy.deepcopy(IDENTITY)
    rotation[0][0] = 0.999

    for observed in (two_ulp, translation, rotation):
        proof = attest_finite_f64_matrix4x4(IDENTITY, observed)
        assert proof["passed"] is False
        assert proof["maximum_ulp_distance"] >= 2
        assert "matrix_element_exceeds_one_ulp" in proof["failure_reasons"]


def _camera_state() -> dict[str, object]:
    return {
        "path": "/FillStation/FillStation_LocalRoot/FillCam/FillCam",
        "projection": "perspective",
        "focal_length": 0.40993183851242065,
        "clipping_range": [0.0007619999814778566, 25.399999618530273],
        "world_transform_matrix": copy.deepcopy(IDENTITY),
    }


def test_effective_state_allows_only_the_derived_matrix_one_ulp() -> None:
    expected = _camera_state()
    observed = _camera_state()
    observed["world_transform_matrix"][2][0] = math.nextafter(0.0, 1.0)

    proof = attest_effective_state_matrix(
        expected,
        observed,
        matrix_field="world_transform_matrix",
    )

    assert proof["passed"] is True
    assert proof["exact_nonmatrix_state_match"] is True
    assert verify_effective_state_matrix_attestation(
        proof,
        expected,
        matrix_field="world_transform_matrix",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "/OtherCamera"),
        ("projection", "orthographic"),
        ("focal_length", 0.5),
        ("clipping_range", [0.1, 25.399999618530273]),
    ],
)
def test_exact_authored_or_scalar_state_drift_fails(
    field: str, value: object
) -> None:
    expected = _camera_state()
    observed = _camera_state()
    observed[field] = value

    proof = attest_effective_state_matrix(
        expected,
        observed,
        matrix_field="world_transform_matrix",
    )

    assert proof["passed"] is False
    assert proof["exact_nonmatrix_state_match"] is False
    assert "nonmatrix_state_mismatch" in proof["failure_reasons"]


def test_occluder_raw_matrix_hash_is_recomputed_while_other_fields_stay_exact() -> None:
    expected = {
        "mesh_points_f64_sha256": "a" * 64,
        "mesh_face_indices_i64_sha256": "b" * 64,
        "effective_visibility": "inherited",
        "mesh_world_transform_matrix": copy.deepcopy(IDENTITY),
        "mesh_world_transform_f64_sha256": "placeholder",
    }
    expected_matrix = attest_finite_f64_matrix4x4(IDENTITY, IDENTITY)
    expected["mesh_world_transform_f64_sha256"] = expected_matrix[
        "expected_matrix_f64_sha256"
    ]
    observed = copy.deepcopy(expected)
    observed["mesh_world_transform_matrix"][2][0] = math.nextafter(0.0, 1.0)
    observed_matrix = attest_finite_f64_matrix4x4(
        expected["mesh_world_transform_matrix"],
        observed["mesh_world_transform_matrix"],
    )
    observed["mesh_world_transform_f64_sha256"] = observed_matrix[
        "observed_matrix_f64_sha256"
    ]

    proof = attest_effective_state_matrix(
        expected,
        observed,
        matrix_field="mesh_world_transform_matrix",
        matrix_raw_hash_field="mesh_world_transform_f64_sha256",
    )
    assert proof["passed"] is True

    observed["effective_visibility"] = "invisible"
    drift = attest_effective_state_matrix(
        expected,
        observed,
        matrix_field="mesh_world_transform_matrix",
        matrix_raw_hash_field="mesh_world_transform_f64_sha256",
    )
    assert drift["passed"] is False
    assert "nonmatrix_state_mismatch" in drift["failure_reasons"]

    topology_observed = copy.deepcopy(expected)
    topology_observed["mesh_face_indices_i64_sha256"] = "c" * 64
    topology_drift = attest_effective_state_matrix(
        expected,
        topology_observed,
        matrix_field="mesh_world_transform_matrix",
        matrix_raw_hash_field="mesh_world_transform_f64_sha256",
    )
    assert topology_drift["passed"] is False
    assert "nonmatrix_state_mismatch" in topology_drift["failure_reasons"]
