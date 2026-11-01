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


MATRIX_COMPARATOR_VERSION = (
    "binary64-4x4-elementwise-rtol-1e-12-atol-1e-12-v2"
)
MATRIX_RELATIVE_TOLERANCE = 1.0e-12
MATRIX_ABSOLUTE_TOLERANCE = 1.0e-12


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


def attest_finite_f64_matrix4x4(
    expected: object,
    observed: object,
) -> dict[str, object]:
    """Compare finite list-backed 4x4 matrices with a frozen tolerance.

    This deliberately has no configurable tolerance. Equal numeric values, including
    opposite signed-zero encodings, have zero error. Raw f64 hashes retain the
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
        "relative_tolerance": MATRIX_RELATIVE_TOLERANCE,
        "absolute_tolerance": MATRIX_ABSOLUTE_TOLERANCE,
        "expected_matrix": deepcopy(expected),
        "observed_matrix": deepcopy(observed),
        "expected_matrix_f64_sha256": None,
        "observed_matrix_f64_sha256": None,
        "elementwise_absolute_errors": None,
        "elementwise_tolerance_limits": None,
        "elementwise_within_tolerance": None,
        "maximum_absolute_error": None,
        "passed": False,
        "failure_reasons": failures,
    }
    if expected_matrix is None or observed_matrix is None:
        return result

    errors: list[list[float]] = []
    limits: list[list[float]] = []
    comparisons: list[list[bool]] = []
    for expected_row, observed_row in zip(expected_matrix, observed_matrix):
        error_row: list[float] = []
        limit_row: list[float] = []
        comparison_row: list[bool] = []
        for expected_value, observed_value in zip(expected_row, observed_row):
            error_row.append(abs(expected_value - observed_value))
            limit_row.append(
                max(
                    MATRIX_ABSOLUTE_TOLERANCE,
                    MATRIX_RELATIVE_TOLERANCE
                    * max(abs(expected_value), abs(observed_value)),
                )
            )
            comparison_row.append(
                math.isclose(
                    expected_value,
                    observed_value,
                    rel_tol=MATRIX_RELATIVE_TOLERANCE,
                    abs_tol=MATRIX_ABSOLUTE_TOLERANCE,
                )
            )
        errors.append(error_row)
        limits.append(limit_row)
        comparisons.append(comparison_row)
    maximum = max(error for row in errors for error in row)
    if not all(passed for row in comparisons for passed in row):
        failures.append("matrix_element_outside_tolerance")
    result.update(
        {
            "expected_matrix_f64_sha256": _matrix_f64_sha256(expected_matrix),
            "observed_matrix_f64_sha256": _matrix_f64_sha256(observed_matrix),
            "elementwise_absolute_errors": errors,
            "elementwise_tolerance_limits": limits,
            "elementwise_within_tolerance": comparisons,
            "maximum_absolute_error": maximum,
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
        "attestation_version": "effective-state-derived-matrix-attestation-v2",
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


def attest_effective_state_matrix_sequence(
    expected_states: object,
    observed_states: object,
    *,
    matrix_field: str,
) -> dict[str, object]:
    """Attest an ordered state sequence with one derived matrix per row."""

    failures: list[str] = []
    if (
        not isinstance(expected_states, list)
        or not isinstance(observed_states, list)
        or not expected_states
        or len(expected_states) != len(observed_states)
    ):
        failures.append("state_sequence_membership_invalid")
        expected = deepcopy(expected_states)
        observed = deepcopy(observed_states)
        row_attestations: list[dict[str, object]] = []
    else:
        expected = deepcopy(expected_states)
        observed = deepcopy(observed_states)
        full_row_attestations = [
            attest_effective_state_matrix(
                expected_row,
                observed_row,
                matrix_field=matrix_field,
            )
            for expected_row, observed_row in zip(
                expected_states, observed_states, strict=True
            )
        ]
        row_attestations = [
            {
                "expected_state_sha256": row["expected_state_sha256"],
                "observed_state_sha256": row["observed_state_sha256"],
                "exact_nonmatrix_state_match": row["exact_nonmatrix_state_match"],
                "expected_matrix_f64_sha256": row["matrix_comparison"][
                    "expected_matrix_f64_sha256"
                ],
                "observed_matrix_f64_sha256": row["matrix_comparison"][
                    "observed_matrix_f64_sha256"
                ],
                "maximum_absolute_error": row["matrix_comparison"][
                    "maximum_absolute_error"
                ],
                "passed": row["passed"],
                "failure_reasons": list(row["failure_reasons"]),
            }
            for row in full_row_attestations
        ]
        if any(row.get("passed") is not True for row in full_row_attestations):
            failures.append("state_sequence_row_attestation_failed")
    try:
        expected_sha256 = _canonical_sha256(expected)
    except (TypeError, ValueError):
        expected_sha256 = None
        failures.append("expected_state_sequence_not_canonical_json")
    try:
        observed_sha256 = _canonical_sha256(observed)
    except (TypeError, ValueError):
        observed_sha256 = None
        failures.append("observed_state_sequence_not_canonical_json")
    failures = list(dict.fromkeys(failures))
    return {
        "attestation_version": "effective-state-derived-matrix-sequence-attestation-v1",
        "matrix_field": matrix_field,
        "matrix_comparator_version": MATRIX_COMPARATOR_VERSION,
        "relative_tolerance": MATRIX_RELATIVE_TOLERANCE,
        "absolute_tolerance": MATRIX_ABSOLUTE_TOLERANCE,
        "expected_states": expected,
        "observed_states": observed,
        "expected_states_sha256": expected_sha256,
        "observed_states_sha256": observed_sha256,
        "row_attestations": row_attestations,
        "passed": not failures,
        "failure_reasons": failures,
    }


def verify_effective_state_matrix_sequence_attestation(
    attestation: object,
    *,
    matrix_field: str,
) -> bool:
    """Recompute an ordered state-sequence attestation exactly."""

    if not isinstance(attestation, Mapping):
        return False
    recomputed = attest_effective_state_matrix_sequence(
        attestation.get("expected_states"),
        attestation.get("observed_states"),
        matrix_field=matrix_field,
    )
    return recomputed.get("passed") is True and dict(attestation) == recomputed
