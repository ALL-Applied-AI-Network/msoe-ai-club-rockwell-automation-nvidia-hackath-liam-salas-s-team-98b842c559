"""Production execution route for the frozen camera fill-step tracker.

The route deliberately does not call the legacy endpoint selector.  It decodes one
capture once, creates a distinct incremental tracker for every receipt-bound vial
window, closes each tracker at that window's boundary, and publishes only the
tracker-selected early/mid/end images.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .wp7_fill_step_production import (
    IncrementalFillStepTracker,
    ProductionFillStepError,
    TopologyFreeze,
    persist_completed_observations,
)
from .wp7_fill_step_tracker import RecipePour, TrackerFreeze
from .wp7_rgb_runtime import (
    LoadedFrozenCNN,
    bind_same_run_empty_references,
    build_trusted_real_rgb_mask_registry_v1,
    compose_fill_windows,
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


@dataclass(frozen=True)
class TrackerLineage:
    topology_file_sha256: str
    topology_receipt_sha256: str
    train_freeze_file_sha256: str
    materialization_file_sha256: str
    calibration_file_sha256: str
    checkpoint_sha256: str
    replay_sha256: str
    review_sha256: str


class FrozenFillStepTrackerRuntime:
    """PredictedMaskRuntime implementation backed only by the frozen tracker."""

    runtime_id = "wp7-frozen-fill-step-tracker-v1"
    accepts_real_rgb_only = True

    def __init__(
        self,
        *,
        loaded_cnn: LoadedFrozenCNN,
        capture_receipt: Mapping[str, object],
        freeze: TrackerFreeze,
        topology: TopologyFreeze,
        lineage: TrackerLineage,
        sidecar_out: Path,
        device: str,
        target_recipe_raw_sha256: str,
        target_recipe_semantic_sha256: str,
        initial_previous_final_tank_id: str | None,
        frame_rate_hz: float,
        frame_decoder,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ProductionFillStepError("tracker device must be cpu or cuda")
        if initial_previous_final_tank_id not in {None, "t100", "t200", "t400"}:
            raise ProductionFillStepError("initial previous final tank is invalid")
        if frame_rate_hz != 30.0:
            raise ProductionFillStepError("tracker capture requires verified 30.0 fps")
        self.loaded_cnn = loaded_cnn
        self.capture_receipt = dict(capture_receipt)
        self.freeze = freeze
        self.topology = topology
        self.lineage = lineage
        self.sidecar_out = Path(sidecar_out).absolute()
        self.device = device
        self.target_recipe_raw_sha256 = target_recipe_raw_sha256
        self.target_recipe_semantic_sha256 = target_recipe_semantic_sha256
        self.initial_previous_final_tank_id = initial_previous_final_tank_id
        self.frame_rate_hz = frame_rate_hz
        self._frame_decoder = frame_decoder
        self.runtime_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.config_sha256 = _canonical_hash({
            "lineage": lineage.__dict__, "device": device,
            "target_recipe_raw_sha256": target_recipe_raw_sha256,
            "target_recipe_semantic_sha256": target_recipe_semantic_sha256,
            "initial_previous_final_tank_id": initial_previous_final_tank_id,
            "frame_rate_hz": frame_rate_hz,
        })

    def _predict(self, current: np.ndarray, reference: np.ndarray) -> object:
        from .cnn_predictor import predict_pair
        return predict_pair(
            self.loaded_cnn.model, current, reference,
            threshold=0.5, device=self.device,
        )

    def analyze(self, *, video_bytes: bytes, recipe_sequence: Mapping[str, object],
                artifact_root: Path):
        from .carryover_pipeline import RuntimeProjectionInput
        from .vision_measurement import build_trusted_runtime_measured_evidence_v1

        if _canonical_hash(recipe_sequence) != self.target_recipe_semantic_sha256:
            raise ProductionFillStepError("target recipe semantic identity drifted")
        if hashlib.sha256(video_bytes).hexdigest() != self.capture_receipt.get("video_sha256"):
            raise ProductionFillStepError("full video identity drifted")
        frames = tuple(np.asarray(frame) for frame in self._frame_decoder(video_bytes))
        if not frames or any(
            frame.dtype != np.uint8 or frame.shape != (480, 640, 3)
            for frame in frames
        ):
            raise ProductionFillStepError("decoded tracker frames are invalid")
        windows = compose_fill_windows(
            self.capture_receipt["encoded_transition_evidence"],
            self.capture_receipt["fill_clips"],
        )
        references = bind_same_run_empty_references(
            frames, self.capture_receipt["encoded_transition_evidence"], windows,
        )
        vials = [
            (run, vial) for run in recipe_sequence.get("runs", [])
            for vial in run.get("vials", [])
        ]
        if len(vials) != len(windows) or len(references) != len(windows):
            raise ProductionFillStepError("capture plan and recipe vial denominator differ")

        root = Path(artifact_root).absolute() / "tracker-runtime"
        observations: list[Mapping[str, object]] = []
        receipts: list[Mapping[str, object]] = []
        abstentions: list[dict[str, object]] = []
        source_hashes: list[str] = []
        context: list[tuple[Mapping[str, object], Mapping[str, object], object, object]] = []
        block_vials: dict[tuple[str, str], list[Mapping[str, object]]] = {}
        for run, vial in vials:
            block_vials.setdefault(
                (str(run["run_id"]), str(vial["block_recipe_id"])), []
            ).append(vial)
        block_hashes = {
            key: _canonical_hash({
                "schema_version": "wp7-production-tracker-cohort-recipe-v1",
                "run_id": key[0], "block_recipe_id": key[1],
                "vials": rows,
            })
            for key, rows in block_vials.items()
        }
        previous_final = self.initial_previous_final_tank_id
        for position, ((run, vial), window, reference) in enumerate(
            zip(vials, windows, references, strict=True), 1
        ):
            order = tuple(str(item) for item in vial["dispense_order"])
            pours = tuple(
                RecipePour(tank, float(vial["programmed_tank_ml"][tank]))
                for tank in order if float(vial["programmed_tank_ml"][tank]) > 0
            )
            source_sha = hashlib.sha256(b"".join(
                frame.tobytes() for frame in frames[
                    window.start_frame_index:window.stop_frame_index
                ]
            )).hexdigest()
            source_hashes.append(source_sha)
            if previous_final is None:
                abstentions.append({
                    "case_id": vial["case_id"], "vial_position": position,
                    "reason_code": "unknown_previous_final_tank",
                    "recipe_stage_count": len(pours), "source_artifact_sha256": source_sha,
                })
                previous_final = pours[-1].tank_id
                continue
            tracker = IncrementalFillStepTracker(
                freeze=replace(
                    self.freeze,
                    recipe_artifact_sha256=block_hashes[
                        (str(run["run_id"]), str(vial["block_recipe_id"]))
                    ],
                ),
                topology=self.topology, recipe_pours=pours,
                previous_final_tank_id=previous_final,
                same_run_empty_reference_rgb=reference.rgb,
                predictor=self._predict,
                cohort_id=f"{run['run_id']}:{vial['block_recipe_id']}",
                case_id=str(vial["case_id"]), vial_id=str(vial["case_id"]),
                vial_position=int(vial["block_position"]),
                source_artifact_sha256=source_sha,
            )
            for frame_index in range(window.start_frame_index, window.stop_frame_index):
                observations.extend(tracker.observe(
                    frames[frame_index], frame_index=frame_index,
                    elapsed_seconds=frame_index / self.frame_rate_hz,
                ))
            closed = tracker.close()
            observations.extend(closed)
            receipt = closed[-1].get("tracker_receipt")
            if not isinstance(receipt, Mapping) or len(receipt.get("stages", [])) != len(pours):
                raise ProductionFillStepError("tracker receipt denominator is incomplete")
            receipts.append(dict(receipt))
            context.append((run, vial, window, reference))
            previous_final = pours[-1].tank_id
        if len(source_hashes) != len(set(source_hashes)):
            raise ProductionFillStepError("per-vial source hashes must be distinct")

        registry = persist_completed_observations(
            output_root=root / "selected", observations=observations,
        )
        # The Part-B measurement registry contains only the selected frames and
        # independently replayed raw CNN masks.  Tracker topology masks remain in
        # the selected registry and are bound by the terminal sidecar.
        images: dict[str, dict[str, object]] = {}
        selected_by_id = {str(row["image_id"]): row for row in observations}
        reference_paths: dict[str, Path] = {}
        reference_indices = {
            str(vial["case_id"]): int(reference.frame_index)
            for _run, vial, _window, reference in context
        }
        for row in observations:
            image_id, case_id = str(row["image_id"]), str(row["vial_id"])
            rgb = np.asarray(row["observation_rgb"])
            reference_rgb = np.asarray(row["same_run_empty_reference_rgb"])
            raw_mask = np.asarray(self._predict(rgb, reference_rgb).mask)
            rgb_path = root / "measurement-rgb" / f"{image_id}.png"
            mask_path = root / "measurement-mask" / f"{image_id}.png"
            reference_path = reference_paths.setdefault(
                case_id, root / "references" / f"{case_id}.png"
            )
            for path, array, mode in (
                (rgb_path, rgb, "RGB"),
                (mask_path, raw_mask.astype(np.uint8) * 255, "L"),
            ):
                buffer = BytesIO(); Image.fromarray(array, mode=mode).save(buffer, format="PNG")
                _write_new(path, buffer.getvalue())
            if not reference_path.exists():
                buffer = BytesIO(); Image.fromarray(reference_rgb, mode="RGB").save(buffer, format="PNG")
                _write_new(reference_path, buffer.getvalue())
            images[image_id] = {
                "case_id": case_id, "frame_index": int(row["frame_index"]),
                "current_rgb_path": str(rgb_path),
                "current_rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                "reference_frame_index": reference_indices[case_id],
                "reference_rgb_path": str(reference_path),
                "reference_rgb_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                "mask_path": str(mask_path),
                "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            }
        from .wp7_rgb_runtime import _accepted_cnn_lineage
        accepted = _accepted_cnn_lineage()
        artifact = {
            "schema_version": "wp7-camera-rgb-mask-registry-v1",
            "artifact_type": "camera_rgb_mask_registry",
            "data_origin": self.capture_receipt["capture_origin"],
            "producer": {
                "producer_id": "frozen-wp5-six-channel-cnn",
                "checkpoint_sha256": accepted["checkpoint"],
                "replay_sha256": accepted["replay"],
                "review_sha256": accepted["review"], "probability_threshold": 0.5,
                "input_fields": ["current_rgb", "same_run_pre_pour_rgb"],
                "mask_source": "frozen_cnn_prediction",
            },
            "lineage": {role: {
                "path": self.loaded_cnn.lineage_paths[role], "sha256": accepted[role],
            } for role in accepted},
            "images": images,
        }
        trusted = build_trusted_real_rgb_mask_registry_v1(artifact, self.loaded_cnn)
        requests: dict[str, dict[str, str]] = {}
        end_rows: dict[tuple[str, int], Mapping[str, object]] = {}
        for row in observations:
            if row["phase"] == "end":
                key = (str(row["vial_id"]), int(row["fill_step"]))
                end_rows[key] = row
                requests[str(row["image_id"])] = {
                    "measurement_id": f"MEAS-{row['vial_id']}-{row['tank_id']}",
                    "evidence_id": f"EVID-{row['vial_id']}-{row['tank_id']}",
                }
        measured = build_trusted_runtime_measured_evidence_v1(trusted, requests)
        templates, locators, cursors = [], {}, {}
        cursor = 0
        transitions = self.capture_receipt["encoded_transition_evidence"]
        for run, vial, window, _reference in context:
            for ordinal, tank in enumerate(vial["dispense_order"], 1):
                if float(vial["programmed_tank_ml"][tank]) <= 0:
                    continue
                row = end_rows[(str(vial["case_id"]), ordinal)]
                image_id = str(row["image_id"]); entry = trusted.entry(image_id)
                evidence_id = requests[image_id]["evidence_id"]
                templates.append({
                    "run_id": run["run_id"], "case_id": vial["case_id"],
                    "vial_ordinal": vial["vial_ordinal"],
                    "block_recipe_id": vial["block_recipe_id"],
                    "block_position": vial["block_position"], "recipe_id": vial["recipe_id"],
                    "previous_recipe_id": vial["previous_recipe_id"],
                    "recipe_ml": dict(vial["programmed_tank_ml"]),
                    "dispense_order": list(vial["dispense_order"]),
                    "split_role": "real_validation", "split_version": recipe_sequence["split_version"],
                    "stage_id": f"STAGE-{vial['case_id']}-{ordinal}", "tank_id": tank,
                    "stage_ordinal": ordinal, "fill_clip_path": "capture-plan-bound-video",
                    "fill_frame_range": [window.start_frame_index, window.stop_frame_index],
                    "transition_frame_ranges": {
                        "entry": [
                            int(transitions[window.fill_index]["start_frame_index"]),
                            int(transitions[window.fill_index]["stop_frame_index"]),
                        ],
                        "exit": [
                            int(transitions[window.fill_index + 1]["start_frame_index"]),
                            int(transitions[window.fill_index + 1]["stop_frame_index"]),
                        ],
                    },
                    "clip_sha256": self.capture_receipt["video_sha256"], "image_id": image_id,
                    "rgb_sha256": entry.rgb_sha256, "mask_sha256": entry.mask_sha256,
                    "mask_source": "predicted", "measurement_id": requests[image_id]["measurement_id"],
                    "median_lab": [0.0, 0.0, 0.0], "mad_lab": [0.0, 0.0, 0.0],
                    "iqr_lab": [0.0, 0.0, 0.0], "pairwise_delta_e76": [],
                    "colour_regime_id": None, "evidence_id": evidence_id,
                    "decision_id": None, "disposition": None, "limit_codes": [],
                    "history_refs": [], "explanation_id": None, "model_revision": None,
                    "prompt_sha256": None, "image_ids_sent": [], "guard_result": None,
                })
                locators[image_id] = entry.measurement_locator(); cursors[evidence_id] = cursor
                cursor += 1
        bindings = {row.image_id: row for row in __import__(
            "fluidmix.vision_measurement", fromlist=["trusted_measurement_bindings_from_result_v1"]
        ).trusted_measurement_bindings_from_result_v1(measured)}
        provenance_vials = []
        for run, vial, _window, _reference in context:
            rows = [row for row in templates if row["case_id"] == vial["case_id"]]
            provenance_vials.append({
                "case_id": vial["case_id"], "selection_status": "selected",
                "selected_frame_indices": [int(selected_by_id[row["image_id"]]["frame_index"]) for row in rows],
                "measurement_citation_sha256s": [bindings[row["image_id"]].measurement_sha256 for row in rows],
                "limit_codes": [],
            })
        for row in abstentions:
            provenance_vials.append({
                "case_id": row["case_id"], "selection_status": "abstain",
                "selected_frame_indices": [], "measurement_citation_sha256s": [],
                "limit_codes": [row["reason_code"]],
            })
        phase_locators: dict[str, dict[str, object]] = {}
        phase_timestamps: dict[str, float] = {}
        for image_id, binding in registry["images"].items():
            crop_path = Path(str(binding["crop_path"]))
            with Image.open(crop_path) as crop_image:
                crop_pixels = np.asarray(crop_image, dtype=np.uint8).copy()
                crop_dimensions = list(crop_image.size)
            phase_locators[image_id] = {
                "path": str(crop_path), "sha256": binding["crop_sha256"],
                "source_full_frame_sha256": binding["rgb_sha256"],
                "crop_pixel_sha256": hashlib.sha256(crop_pixels.tobytes()).hexdigest(),
                "crop_box_xyxy": binding["crop_box_xyxy"],
                "source_native_dimensions": [640, 480],
                "crop_native_dimensions": crop_dimensions,
            }
            phase_timestamps[image_id] = (
                int(selected_by_id[image_id]["frame_index"]) / self.frame_rate_hz
            )
        sidecar = {
            "schema_version": "wp7-fill-step-terminal-sidecar-v1",
            "artifact_type": "wp7_fill_step_terminal_sidecar",
            "status": "complete", "device": self.device,
            "verified_frame_rate_hz": self.frame_rate_hz,
            "target_recipe_raw_sha256": self.target_recipe_raw_sha256,
            "target_recipe_semantic_sha256": self.target_recipe_semantic_sha256,
            "video_sha256": self.capture_receipt["video_sha256"],
            "per_vial_source_sha256s": source_hashes,
            "cnn_lineage": {
                "checkpoint_sha256": self.lineage.checkpoint_sha256,
                "replay_sha256": self.lineage.replay_sha256,
                "review_sha256": self.lineage.review_sha256,
            },
            "tracker_lineage": self.lineage.__dict__,
            "source_code_sha256s": {
                "tracker_runtime": self.runtime_sha256,
                "fill_step_production": hashlib.sha256(
                    (Path(__file__).with_name("wp7_fill_step_production.py")).read_bytes()
                ).hexdigest(),
            },
            "represented_vial_count": len(receipts) + len(abstentions),
            "expected_vial_count": len(vials), "accepted_receipt_count": len(receipts),
            "abstentions": abstentions, "tracker_receipts": receipts,
            "cohort_recipe_artifact_sha256s": {
                f"{key[0]}:{key[1]}": value for key, value in block_hashes.items()
            },
            "selected_image_count": len(phase_locators),
            "phase_locators": phase_locators,
            "phase_timestamps_seconds": phase_timestamps,
            "tracker_registry_sha256": registry["registry_sha256"],
            "measurement_registry_sha256": trusted.registry_sha256,
        }
        sidecar["receipt_sha256"] = _canonical_hash(sidecar)
        _write_new(self.sidecar_out, json.dumps(
            sidecar, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode() + b"\n")
        return RuntimeProjectionInput(
            record_templates=tuple(templates), trusted_adapter_result=measured,
            image_locators=locators, event_cursors=cursors,
            runtime_provenance={
                "schema_version": "wp7-fill-step-runtime-provenance-v1",
                "vials": provenance_vials, "tracker_sidecar_sha256": sidecar["receipt_sha256"],
                "legacy_selector_used": False,
            },
            phase_image_locators=phase_locators,
        )


__all__ = ["FrozenFillStepTrackerRuntime", "TrackerLineage"]
