"""Fail-closed attestation for platform-derived binary64 transform matrices."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
import struct
from typing import Any


MATRIX_COMPARATOR_VERSION = "ieee754-binary64-4x4-elementwise-one-ulp-v1"
_UINT64_MASK = (1 << 64) - 1
_SIGN_BIT = 1 << 63


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copy_finite_f64_matrix4x4(
    value: object,
    *,
    label: str,
) -> tuple[list[list[float]] | None, list[str]]:
    shape_reason = f"{label}_matrix_shape_or_type_invalid"
    element_reason = f"{label}_matrix_element_invalid"
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        return None, [shape_reason]
    result: list[list[float]] = []
    for row in value:
        converted_row: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, Real):
                return None, [element_reason]
            try:
                converted = float(item)
            except (OverflowError, TypeError, ValueError):
                return None, [element_reason]
            if not math.isfinite(converted):
                return None, [element_reason]
            converted_row.append(converted)
        result.append(converted_row)
    return result, []


def _matrix_f64_sha256(matrix: list[list[float]]) -> str:
    payload = b"".join(
        struct.pack("<d", value) for row in matrix for value in row
    )
    return hashlib.sha256(payload).hexdigest()


def _ordered_binary64_bits(value: float) -> int:
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return (~bits & _UINT64_MASK) if bits & _SIGN_BIT else bits | _SIGN_BIT


def attest_finite_f64_matrix4x4(
    expected: object,
    observed: object,
) -> dict[str, object]:
    """Compare two finite list-backed 4x4 matrices elementwise within one ULP.

    This deliberately has no configurable tolerance. Equal numeric values, including
    opposite signed-zero encodings, have distance zero. Raw f64 hashes retain the
    encoding distinction for audit evidence.
    """

    expected_matrix, expected_failures = _copy_finite_f64_matrix4x4(
        expected, label="expected"
    )
    observed_matrix, observed_failures = _copy_finite_f64_matrix4x4(
        observed, label="observed"
    )
    failures = expected_failures + observed_failures
    result: dict[str, object] = {
        "comparator_version": MATRIX_COMPARATOR_VERSION,
        "maximum_allowed_ulp_distance": 1,
        "expected_matrix": deepcopy(expected),
        "observed_matrix": deepcopy(observed),
        "expected_matrix_f64_sha256": None,
        "observed_matrix_f64_sha256": None,
        "elementwise_ulp_distances": None,
        "maximum_ulp_distance": None,
        "passed": False,
        "failure_reasons": failures,
    }
    if expected_matrix is None or observed_matrix is None:
        return result

    distances: list[list[int]] = []
    for expected_row, observed_row in zip(expected_matrix, observed_matrix):
        distance_row: list[int] = []
        for expected_value, observed_value in zip(expected_row, observed_row):
            if expected_value == observed_value:
                distance = 0
            else:
                distance = abs(
                    _ordered_binary64_bits(expected_value)
                    - _ordered_binary64_bits(observed_value)
                )
            distance_row.append(distance)
        distances.append(distance_row)
    maximum = max(distance for row in distances for distance in row)
    if maximum > 1:
        failures.append("matrix_element_exceeds_one_ulp")
    result.update(
        {
            "expected_matrix_f64_sha256": _matrix_f64_sha256(expected_matrix),
            "observed_matrix_f64_sha256": _matrix_f64_sha256(observed_matrix),
            "elementwise_ulp_distances": distances,
            "maximum_ulp_distance": maximum,
            "passed": not failures,
            "failure_reasons": failures,
        }
    )
    return result


def attest_effective_state_matrix(
    expected_state: object,
    observed_state: object,
    *,
    matrix_field: str,
    matrix_raw_hash_field: str | None = None,
) -> dict[str, object]:
    """Attest one derived matrix while requiring every other state field exactly."""

    failures: list[str] = []
    if not isinstance(expected_state, Mapping):
        failures.append("expected_state_invalid")
        expected: dict[str, Any] = {}
    else:
        expected = deepcopy(dict(expected_state))
    if not isinstance(observed_state, Mapping):
        failures.append("observed_state_invalid")
        observed: dict[str, Any] = {}
    else:
        observed = deepcopy(dict(observed_state))

    matrix_proof = attest_finite_f64_matrix4x4(
        expected.get(matrix_field), observed.get(matrix_field)
    )
    failures.extend(str(reason) for reason in matrix_proof["failure_reasons"])

    excluded_fields = {matrix_field}
    if matrix_raw_hash_field is not None:
        excluded_fields.add(matrix_raw_hash_field)
    expected_exact = {
        key: value for key, value in expected.items() if key not in excluded_fields
    }
    observed_exact = {
        key: value for key, value in observed.items() if key not in excluded_fields
    }
    try:
        exact_nonmatrix_match = (
            json.dumps(
                expected_exact,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            == json.dumps(
                observed_exact,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        exact_nonmatrix_match = False
    if not exact_nonmatrix_match:
        failures.append("nonmatrix_state_mismatch")

    if matrix_raw_hash_field is not None:
        if (
            expected.get(matrix_raw_hash_field)
            != matrix_proof.get("expected_matrix_f64_sha256")
        ):
            failures.append("expected_matrix_raw_hash_mismatch")
        if (
            observed.get(matrix_raw_hash_field)
            != matrix_proof.get("observed_matrix_f64_sha256")
        ):
            failures.append("observed_matrix_raw_hash_mismatch")

    try:
        expected_sha256 = _canonical_sha256(expected)
    except (TypeError, ValueError):
        expected_sha256 = None
        failures.append("expected_state_not_canonical_json")
    try:
        observed_sha256 = _canonical_sha256(observed)
    except (TypeError, ValueError):
        observed_sha256 = None
        failures.append("observed_state_not_canonical_json")

    failures = list(dict.fromkeys(failures))
    return {
        "attestation_version": "effective-state-derived-matrix-attestation-v1",
        "matrix_field": matrix_field,
        "matrix_raw_hash_field": matrix_raw_hash_field,
        "expected_state": expected,
        "observed_state": observed,
        "expected_state_sha256": expected_sha256,
        "observed_state_sha256": observed_sha256,
        "exact_nonmatrix_state_match": exact_nonmatrix_match,
        "matrix_comparison": matrix_proof,
        "passed": not failures,
        "failure_reasons": failures,
    }


def verify_effective_state_matrix_attestation(
    attestation: object,
    expected_state: object,
    *,
    matrix_field: str,
    matrix_raw_hash_field: str | None = None,
) -> bool:
    """Recompute an attestation and require the supplied evidence to be exact."""

    if not isinstance(attestation, Mapping):
        return False
    observed_state = attestation.get("observed_state")
    recomputed = attest_effective_state_matrix(
        expected_state,
        observed_state,
        matrix_field=matrix_field,
        matrix_raw_hash_field=matrix_raw_hash_field,
    )
    return recomputed.get("passed") is True and dict(attestation) == recomputed
