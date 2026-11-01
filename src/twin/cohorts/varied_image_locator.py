"""Hash-bound image locators for the corrected Varied V3 cohort."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path

from fluidmix.storage.artifacts.capture_manifest_adapter import (
    CaptureManifestError,
    bind_stage_capture_manifest,
)
from twin.render.render_process import hash_regular_file
from fluidmix.measurement.vision_measurement import _publish_json_no_clobber


class VariedImageLocatorError(ValueError):
    """The corrected-cohort locator boundary failed closed."""


def build_varied_image_locators(
    manifests: object,
    validation_receipt: str | Path,
) -> dict[str, object]:
    """Build the label-free locator artifact for exactly six V3 runs."""

    if not isinstance(manifests, (list, tuple)) or len(manifests) != 6:
        raise VariedImageLocatorError("manifest_denominator_invalid")
    if any(not isinstance(path, (str, Path)) for path in manifests):
        raise VariedImageLocatorError("manifest_path_invalid")
    lexical_manifests = [Path(path).expanduser().absolute() for path in manifests]
    if len(set(lexical_manifests)) != 6:
        raise VariedImageLocatorError("duplicate_manifest_path")

    receipt_path = Path(validation_receipt).expanduser().absolute()
    try:
        receipt_resolved = receipt_path.resolve(strict=True)
        receipt_snapshot = hash_regular_file(receipt_path)
        receipt = json.loads(
            receipt_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise VariedImageLocatorError("validation_receipt_invalid") from error
    if receipt_resolved != receipt_path or not isinstance(receipt, Mapping):
        raise VariedImageLocatorError("validation_receipt_invalid")
    required_receipt = {
        "schema_version": 2,
        "artifact_type": "f2_varied_v3_downstream_validation_receipt",
        "status": "pass",
        "batch_count": 6,
        "vial_count": 18,
        "stage_pair_count": 42,
        "source_artifact_validation_count": 18,
    }
    if any(receipt.get(key) != value for key, value in required_receipt.items()):
        raise VariedImageLocatorError("validation_receipt_invalid")
    if receipt.get("split_batch_counts") != {
        "train": 4,
        "validation": 1,
        "evaluation": 1,
    }:
        raise VariedImageLocatorError("validation_receipt_invalid")
    receipt_hashes = receipt.get("capture_manifest_sha256_by_batch")
    if not isinstance(receipt_hashes, Mapping) or len(receipt_hashes) != 6:
        raise VariedImageLocatorError("validation_receipt_invalid")

    publication_root = receipt_path.parent / "publications"
    images: dict[str, dict[str, object]] = {}
    sources: list[dict[str, object]] = []
    run_ids: set[str] = set()
    vial_keys: set[tuple[str, str]] = set()
    active_stage_events = 0
    stage_counts: Counter[int] = Counter()
    split_images: Counter[str] = Counter()
    observed_hashes: dict[str, str] = {}
    for lexical in sorted(lexical_manifests):
        try:
            resolved = lexical.resolve(strict=True)
            manifest_snapshot = hash_regular_file(lexical)
        except (OSError, ValueError) as error:
            raise VariedImageLocatorError("manifest_path_invalid") from error
        if resolved != lexical or lexical.name != "f2-stage-capture-manifest.json":
            raise VariedImageLocatorError("manifest_path_invalid")
        try:
            bound = bind_stage_capture_manifest(lexical)
        except CaptureManifestError as error:
            raise VariedImageLocatorError(str(error)) from error
        if bound.get("capture_contract_version") != (
            "f2-brev-isaac62-varied-event-v3"
        ):
            raise VariedImageLocatorError("capture_contract_version_invalid")
        batch_id = str(bound["batch_id"])
        expected_path = (
            publication_root / batch_id / "f2-stage-capture-manifest.json"
        ).absolute()
        if lexical != expected_path:
            raise VariedImageLocatorError("manifest_not_in_validation_root")
        manifest_sha = str(manifest_snapshot["sha256"])
        if bound.get("source_manifest_sha256") != manifest_sha:
            raise VariedImageLocatorError("manifest_hash_binding_mismatch")
        if receipt_hashes.get(batch_id) != manifest_sha:
            raise VariedImageLocatorError("validation_manifest_hash_mismatch")
        if batch_id in observed_hashes:
            raise VariedImageLocatorError("duplicate_batch_id")
        observed_hashes[batch_id] = manifest_sha

        run_id = str(bound["run_id"])
        if run_id in run_ids:
            raise VariedImageLocatorError("duplicate_run_id")
        run_ids.add(run_id)
        events = bound.get("stage_events")
        measurements = bound.get("image_measurements")
        locators = bound.get("image_locators")
        if (
            not isinstance(events, list)
            or len(events) not in {2, 3}
            or not isinstance(measurements, list)
            or not isinstance(locators, Mapping)
            or not isinstance(locators.get("images"), Mapping)
        ):
            raise VariedImageLocatorError("bound_manifest_invalid")
        stage_counts[len(events)] += 1
        active_stage_events += len(events)
        event_by_id = {str(event["event_id"]): event for event in events}
        locator_rows = locators["images"]
        assert isinstance(locator_rows, Mapping)
        if len(measurements) != len(events) * 3 or len(locator_rows) != len(measurements):
            raise VariedImageLocatorError("manifest_image_denominator_invalid")
        seen_vial_stages: set[tuple[str, str]] = set()
        for measured in measurements:
            if not isinstance(measured, Mapping):
                raise VariedImageLocatorError("measurement_invalid")
            image_id = str(measured.get("image_id"))
            vial_id = str(measured.get("vial_id"))
            stage_id = str(measured.get("stage_id"))
            event = event_by_id.get(stage_id)
            locator = locator_rows.get(image_id)
            if (
                image_id in images
                or vial_id not in {"V1", "V2", "V3"}
                or event is None
                or not isinstance(locator, Mapping)
                or set(locator) != {
                    "rgb_path",
                    "rgb_sha256",
                    "mask_path",
                    "mask_sha256",
                }
            ):
                raise VariedImageLocatorError("image_binding_invalid")
            locator_copy: dict[str, str] = {}
            for role in ("rgb", "mask"):
                path = Path(str(locator[f"{role}_path"])).absolute()
                expected_sha = str(locator[f"{role}_sha256"])
                try:
                    artifact_snapshot = hash_regular_file(path)
                    resolved_artifact = path.resolve(strict=True)
                except (OSError, ValueError) as error:
                    raise VariedImageLocatorError(
                        f"{role}_artifact_invalid"
                    ) from error
                if (
                    resolved_artifact != path
                    or artifact_snapshot["sha256"] != expected_sha
                ):
                    raise VariedImageLocatorError(f"{role}_artifact_hash_mismatch")
                locator_copy[f"{role}_path"] = str(path)
                locator_copy[f"{role}_sha256"] = expected_sha
            stage_event = measured.get("stage_event")
            if not isinstance(stage_event, Mapping):
                raise VariedImageLocatorError("stage_event_invalid")
            split_role = str(bound["split_role"])
            images[image_id] = {
                "batch_id": batch_id,
                "run_id": run_id,
                "recipe_id": str(bound["recipe_id"]),
                "split_role": split_role,
                "split_version": str(bound["split_version"]),
                "vial_id": vial_id,
                "stage_id": stage_id,
                "stage_ordinal": int(measured["stage_ordinal"]),
                "tank_id": str(stage_event["tank_id"]),
                "source_manifest_sha256": manifest_sha,
                **locator_copy,
            }
            split_images[split_role] += 1
            seen_vial_stages.add((vial_id, stage_id))
            vial_keys.add((batch_id, vial_id))
        expected = {
            (vial_id, str(event["event_id"]))
            for vial_id in ("V1", "V2", "V3")
            for event in events
        }
        if seen_vial_stages != expected:
            raise VariedImageLocatorError("vial_stage_set_mismatch")
        sources.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "recipe_id": str(bound["recipe_id"]),
                "split_role": str(bound["split_role"]),
                "split_version": str(bound["split_version"]),
                "path": str(lexical),
                "sha256": manifest_sha,
                "active_stage_count": len(events),
                "image_count": len(measurements),
            }
        )

    if observed_hashes != dict(receipt_hashes):
        raise VariedImageLocatorError("validation_manifest_set_mismatch")
    expected_denominators = {
        "source_manifests": 6,
        "runs": 6,
        "vials": 18,
        "active_stage_events": 14,
        "images": 42,
        "two_stage_runs": 4,
        "three_stage_runs": 2,
        "split_images": {"train": 30, "validation": 6, "evaluation": 6},
    }
    observed_denominators = {
        "source_manifests": len(sources),
        "runs": len(run_ids),
        "vials": len(vial_keys),
        "active_stage_events": active_stage_events,
        "images": len(images),
        "two_stage_runs": stage_counts[2],
        "three_stage_runs": stage_counts[3],
        "split_images": dict(split_images),
    }
    if observed_denominators != expected_denominators:
        raise VariedImageLocatorError("cohort_denominator_invalid")
    result = {
        "schema_version": 1,
        "artifact_type": "f2_varied_image_locators",
        "status": "published",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": "whole-run-varied-v3",
        "source_validation_receipt": {
            "path": str(receipt_path),
            "sha256": str(receipt_snapshot["sha256"]),
        },
        "source_manifests": sorted(sources, key=lambda row: str(row["batch_id"])),
        "images": {key: images[key] for key in sorted(images)},
        "denominators": expected_denominators,
        "claim_boundary": (
            "Synthetic-twin image locations and hashes only; no label, decision, "
            "physical-colour, cause, or sim-to-real claim."
        ),
    }
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise VariedImageLocatorError("output_invalid") from error
    return result


def write_varied_image_locators(
    manifests: object,
    validation_receipt: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Publish one immutable locator artifact."""

    result = build_varied_image_locators(manifests, validation_receipt)
    _publish_json_no_clobber(Path(output_path), result)
    return result
