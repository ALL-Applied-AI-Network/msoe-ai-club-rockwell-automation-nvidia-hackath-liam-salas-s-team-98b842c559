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

GB10_CAMERA_MATRIX = [
    [0.1711176749943712, -0.9852415969440802, 0.004233418314836246, 0.0],
    [0.5613216156591883, 0.10102031723522989, 0.8214092306362771, 0.0],
    [-0.8097141285498902, -0.13818131578401316, 0.570323714104472, 0.0],
    [4.938855458564632, -2.0360563206718894, 1.2983193888480855, 1.0],
]

L40S_CAMERA_MATRIX = copy.deepcopy(GB10_CAMERA_MATRIX)
L40S_CAMERA_MATRIX[2][0] = -0.8097141285498903

GB10_OCCLUDER_MATRIX = [
    [
        -2.663161050772638e-10,
        0.0009999999841894733,
        1.1920928112069272e-10,
        0.0,
    ],
    [
        -0.0009999999841894805,
        -2.663161050772633e-10,
        -2.009718047161327e-17,
        0.0,
    ],
    [
        1.1650171467452014e-17,
        -1.1920928112069383e-10,
        0.0009999999841895089,
        0.0,
    ],
    [5.23240032806396, -1.9869149409557139, 1.0943508335738272, 1.0],
]

L40S_OCCLUDER_MATRIX = copy.deepcopy(GB10_OCCLUDER_MATRIX)
L40S_OCCLUDER_MATRIX[2][0] = 1.1650171467452011e-17


def test_exact_and_cross_platform_matrix_attestations_report_all_errors() -> None:
    exact = attest_finite_f64_matrix4x4(IDENTITY, copy.deepcopy(IDENTITY))
    cross_platform = copy.deepcopy(IDENTITY)
    cross_platform[2][0] = 5.0e-13
    cross_platform[3][0] = 5.0
    expected_cross_platform = copy.deepcopy(IDENTITY)
    expected_cross_platform[3][0] = 5.0 + 4.0e-12
    adjacent = attest_finite_f64_matrix4x4(
        expected_cross_platform, cross_platform
    )

    assert exact["passed"] is True
    assert exact["comparator_version"] == MATRIX_COMPARATOR_VERSION
    assert exact["relative_tolerance"] == 1.0e-12
    assert exact["absolute_tolerance"] == 1.0e-12
    assert exact["elementwise_absolute_errors"] == [[0.0] * 4 for _ in range(4)]
    assert exact["maximum_absolute_error"] == 0.0
    assert adjacent["passed"] is True
    assert adjacent["elementwise_absolute_errors"][2][0] == 5.0e-13
    assert adjacent["elementwise_absolute_errors"][3][0] > 1.0e-12
    assert adjacent["elementwise_absolute_errors"][3][0] < adjacent[
        "elementwise_tolerance_limits"
    ][3][0]
    assert adjacent["expected_matrix_f64_sha256"] != adjacent[
        "observed_matrix_f64_sha256"
    ]


def test_preserved_gb10_and_observed_l40s_matrices_pass() -> None:
    camera = attest_finite_f64_matrix4x4(
        GB10_CAMERA_MATRIX, L40S_CAMERA_MATRIX
    )
    occluder = attest_finite_f64_matrix4x4(
        GB10_OCCLUDER_MATRIX, L40S_OCCLUDER_MATRIX
    )

    assert camera["passed"] is True
    assert occluder["passed"] is True
    assert camera["expected_matrix_f64_sha256"] == (
        "de76c91502cf0525138d1fb4154b3c3bd04b6c35a382935936868bff6086dcc8"
    )
    assert camera["observed_matrix_f64_sha256"] == (
        "907db63c7a6cb62e6facb3e5ac9e921c161cff7c94ab321834b06b534dbf61ff"
    )
    assert occluder["expected_matrix_f64_sha256"] == (
        "2df92340d8b7d9726f2bac8d79686c3ce1680ae0bc43ed326dca64e0319d22d0"
    )
    assert occluder["observed_matrix_f64_sha256"] == (
        "9938cbcff174a3cdbe573d729427a08855c0deeeca9514c8483f2360c3356e87"
    )


def test_signed_zero_encodings_are_numerically_equal_but_hash_visible() -> None:
    observed = copy.deepcopy(IDENTITY)
    observed[0][1] = -0.0

    proof = attest_finite_f64_matrix4x4(IDENTITY, observed)

    assert proof["passed"] is True
    assert proof["elementwise_absolute_errors"][0][1] == 0.0
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


def test_outside_tolerance_and_meaningful_translation_or_rotation_fail() -> None:
    outside_tolerance = copy.deepcopy(IDENTITY)
    outside_tolerance[1][1] = 1.0 + 3.0e-12
    translation = copy.deepcopy(IDENTITY)
    translation[3][0] = 0.001
    rotation = copy.deepcopy(IDENTITY)
    rotation[0][0] = 0.999

    for observed in (outside_tolerance, translation, rotation):
        proof = attest_finite_f64_matrix4x4(IDENTITY, observed)
        assert proof["passed"] is False
        assert proof["maximum_absolute_error"] > 1.0e-12
        assert "matrix_element_outside_tolerance" in proof["failure_reasons"]


def _camera_state() -> dict[str, object]:
    return {
        "path": "/FillStation/FillStation_LocalRoot/FillCam/FillCam",
        "projection": "perspective",
        "focal_length": 0.40993183851242065,
        "clipping_range": [0.0007619999814778566, 25.399999618530273],
        "world_transform_matrix": copy.deepcopy(IDENTITY),
    }


def test_effective_state_allows_only_the_derived_matrix_tolerance() -> None:
    expected = _camera_state()
    observed = _camera_state()
    observed["world_transform_matrix"][2][0] = 5.0e-13

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
