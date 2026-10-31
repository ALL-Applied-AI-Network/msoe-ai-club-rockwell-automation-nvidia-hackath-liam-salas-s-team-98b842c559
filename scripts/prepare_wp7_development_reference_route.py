#!/usr/bin/env python3
"""Materialize or preflight the source-only WP7 option-3 route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.carryover_pipeline import (  # noqa: E402
    _validate_execution_preflight,
    preflight_explanation_route,
)
from fluidmix.wp7_development_reference import (  # noqa: E402
    DevelopmentReferenceStore,
    materialize_development_reference_store,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _expected_sha256(value: str, role: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{role} SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _read_bound_file(
    path_value: Path, expected_sha256: str, role: str
) -> tuple[Path, bytes]:
    """Read one exact regular file and verify its raw bytes before parsing."""

    path = path_value.expanduser().absolute()
    expected = _expected_sha256(expected_sha256, role)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{role} is not an exact regular file") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role} is not an exact regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{role} is not an exact regular file") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{role} is not an exact regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"{role} raw-file SHA-256 mismatch")
    return path, payload


def _parse_json(payload: bytes, role: str) -> dict[str, object]:
    try:
        value = json.loads(payload, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{role} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain one JSON object")
    return value


def _ensure_fresh_output(path_value: Path, role: str) -> Path:
    path = path_value.expanduser().absolute()
    if os.path.lexists(path):
        raise ValueError(f"{role} already exists")
    return path


def _write_two_exclusive_json(
    first_path: Path,
    first_value: object,
    second_path: Path,
    second_value: object,
) -> None:
    descriptors: list[tuple[Path, int]] = []
    created_paths: list[Path] = []
    try:
        for path in (first_path, second_path):
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640
            )
            descriptors.append((path, descriptor))
            created_paths.append(path)
        values = (first_value, second_value)
        for value in values:
            path, descriptor = descriptors.pop(0)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_canonical_bytes(value) + b"\n")
    except BaseException:
        for _, descriptor in descriptors:
            os.close(descriptor)
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise


def _ensure_distinct_paths(paths: dict[str, Path]) -> None:
    identities: dict[Path, str] = {}
    for role, path in paths.items():
        identity = path.resolve(strict=False)
        if identity in identities:
            raise ValueError(f"{role} must be distinct from {identities[identity]}")
        identities[identity] = role


def materialize_store(arguments: argparse.Namespace) -> dict[str, object]:
    result = materialize_development_reference_store(
        predicted_store_export_path=arguments.predicted_store_export,
        development_reference_bank_path=arguments.development_reference_bank,
        delta_e_frozen_config_path=arguments.delta_e_frozen_config,
        materialization_receipt_path=arguments.materialization_receipt,
        output_dir=arguments.output_dir,
    )
    queries = result["coverage_queries"]
    return {
        key: result[key]
        for key in (
            "status", "output_dir", "database_path", "database_sha256",
            "receipt_path", "receipt_sha256",
        )
    } | {
        "coverage_query_sha256s": [query.query_sha256 for query in queries],
        "coverage_query_count": len(queries),
        "coverage_member_count": sum(len(query.members) for query in queries),
    }


def publish_preflight(arguments: argparse.Namespace) -> dict[str, object]:
    preflight_output = arguments.preflight_output.expanduser().absolute()
    publication_output = arguments.publication_receipt_output.expanduser().absolute()
    source_arguments = {
        "development-reference store": arguments.development_reference_store,
        "store provenance receipt": arguments.store_provenance_receipt,
        "ordered-video provenance": arguments.ordered_video_provenance,
        "route decision": arguments.route_decision,
    }
    _ensure_distinct_paths(
        {
            "preflight output": preflight_output,
            "publication receipt output": publication_output,
            **{
                role: path.expanduser().absolute()
                for role, path in source_arguments.items()
            },
        }
    )
    _ensure_fresh_output(preflight_output, "preflight output")
    _ensure_fresh_output(publication_output, "publication receipt output")

    store_path, _ = _read_bound_file(
        arguments.development_reference_store,
        arguments.development_reference_store_sha256,
        "development-reference store",
    )
    receipt_path, receipt_bytes = _read_bound_file(
        arguments.store_provenance_receipt,
        arguments.store_provenance_receipt_sha256,
        "store provenance receipt",
    )
    ordered_path, ordered_bytes = _read_bound_file(
        arguments.ordered_video_provenance,
        arguments.ordered_video_provenance_sha256,
        "ordered-video provenance",
    )
    decision_path, decision_bytes = _read_bound_file(
        arguments.route_decision,
        arguments.route_decision_sha256,
        "route decision",
    )
    receipt = _parse_json(receipt_bytes, "store provenance receipt")
    ordered = _parse_json(ordered_bytes, "ordered-video provenance")
    decision = _parse_json(decision_bytes, "route decision")
    if (
        receipt.get("store_snapshot_sha256")
        != arguments.development_reference_store_sha256
    ):
        raise ValueError("development-reference store snapshot/receipt mismatch")

    store = DevelopmentReferenceStore(
        store_path,
        expected_snapshot_sha256=arguments.development_reference_store_sha256,
    )
    queries = store.coverage_queries()
    mask = {
        "schema_version": "wp7-development-predicted-mask-provenance-v1",
        "artifact_type": "development_reference_predicted_mask_provenance",
        "source_class": "accepted_rgb_only_cnn_predicted_masks",
        "predictor": receipt.get("predictor"),
        "predictor_sha256": receipt.get("predictor_sha256"),
        "accepted_store_tuple": receipt.get("accepted_store_tuple"),
        "predicted_store_export_sha256": receipt.get("predicted_store_export_sha256"),
    }
    canonical_decision_sha256 = _canonical_hash(decision)
    preflight = preflight_explanation_route(
        requested_mode="development_reference_backed",
        expected_test_vials=18,
        expected_real_check_vials=30,
        development_reference_store_provenance=receipt,
        predicted_mask_provenance=mask,
        ordered_video_provenance=ordered,
        history_queries=queries,
        route_decision=decision,
        expected_route_decision_sha256=canonical_decision_sha256,
    )
    _validate_execution_preflight(
        preflight,
        expected_sha256=str(preflight["preflight_sha256"]),
        selected_mode="development_reference_backed",
    )
    preflight_raw_sha256 = hashlib.sha256(_canonical_bytes(preflight) + b"\n").hexdigest()
    publication = {
        "schema_version": "wp7-option3-preflight-publication-v1",
        "artifact_type": "wp7_option3_preflight_publication",
        "status": "published",
        "one_shot_inputs_opened": False,
        "preflight_binding": {
            "path": str(preflight_output),
            "raw_file_sha256": preflight_raw_sha256,
            "canonical_preflight_sha256": preflight["preflight_sha256"],
        },
        "source_bindings": {
            "development_reference_store": {
                "path": str(store_path),
                "raw_file_sha256": arguments.development_reference_store_sha256,
            },
            "store_provenance_receipt": {
                "path": str(receipt_path),
                "raw_file_sha256": arguments.store_provenance_receipt_sha256,
            },
            "ordered_video_provenance": {
                "path": str(ordered_path),
                "raw_file_sha256": arguments.ordered_video_provenance_sha256,
            },
            "route_decision": {
                "path": str(decision_path),
                "raw_file_sha256": arguments.route_decision_sha256,
                "canonical_decision_sha256": canonical_decision_sha256,
            },
        },
    }
    _write_two_exclusive_json(
        preflight_output, preflight, publication_output, publication
    )
    return publication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    materialize = commands.add_parser("materialize-store")
    materialize.add_argument("--predicted-store-export", required=True, type=Path)
    materialize.add_argument("--development-reference-bank", required=True, type=Path)
    materialize.add_argument("--delta-e-frozen-config", required=True, type=Path)
    materialize.add_argument("--materialization-receipt", required=True, type=Path)
    materialize.add_argument("--output-dir", required=True, type=Path)

    publish = commands.add_parser("publish-preflight")
    publish.add_argument("--development-reference-store", required=True, type=Path)
    publish.add_argument("--development-reference-store-sha256", required=True)
    publish.add_argument("--store-provenance-receipt", required=True, type=Path)
    publish.add_argument("--store-provenance-receipt-sha256", required=True)
    publish.add_argument("--ordered-video-provenance", required=True, type=Path)
    publish.add_argument("--ordered-video-provenance-sha256", required=True)
    publish.add_argument("--route-decision", required=True, type=Path)
    publish.add_argument("--route-decision-sha256", required=True)
    publish.add_argument("--preflight-output", required=True, type=Path)
    publish.add_argument("--publication-receipt-output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = (
        materialize_store(arguments)
        if arguments.command == "materialize-store"
        else publish_preflight(arguments)
    )
    print(_canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
