"""Thin orchestration adapter over accepted tracker/CNN/Lab/decision contracts.

The tracker and decision objects are injected because their frozen recipes,
calibrations, reference bank, and model snapshots are deployment artifacts. The
adapter does not invent any of those values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ...vision_measurement import measure_image
from ...cnn_predictor import predict_pair
from ...wp7_vlm_image_support import crop_vial_roi


PUBLIC_TRACKER_FIELDS = frozenset({
    "agreement", "measured_volume_ml", "twin_geometry_equivalent_volume_ml",
    "expected_dispense_ml", "expected_cumulative_ml", "stream_step_index",
    "tank_cluster_consistent", "limit_codes",
})


class AcceptedPipelineAdapter:
    """Bind incremental tracker outputs to accepted prediction and measurement."""

    def __init__(
        self,
        *,
        tracker: object,
        cnn_model: object,
        deterministic_decider: Callable[[Mapping[str, object]], Mapping[str, object]],
        vlm_bundle_builder: Callable[..., object],
        vlm_runner: Callable[[object], object],
        device: str,
        model_id: str,
        predictor: Callable[..., object] = predict_pair,
        measurer: Callable[..., dict[str, object]] = measure_image,
    ):
        self.tracker = tracker
        self.cnn_model = cnn_model
        self.deterministic_decider = deterministic_decider
        self.vlm_bundle_builder = vlm_bundle_builder
        self.vlm_runner = vlm_runner
        self.device = device
        self.model_id = model_id
        self._predictor = predictor
        self._measurer = measurer
        self._artifacts = tempfile.TemporaryDirectory(prefix="fluidmix-live-")
        self._image_locators: dict[str, dict[str, object]] = {}
        self._tracker_receipts: dict[str, Mapping[str, object]] = {}
        self._vlm_vials: dict[str, Mapping[str, object]] = {}
        self._cohort_context: Mapping[str, object] | None = None
        self._deterministic_receipt: Mapping[str, object] | None = None
        self._causal_cohort: dict[str, object] | None = None
        self._measurements: dict[tuple[str, str], dict[str, object]] = {}

    def process_frame(self, frame: object, *, frame_index: int, elapsed_seconds: float):
        rgb = np.asarray(frame)
        observed = self.tracker.observe(
            rgb, frame_index=frame_index, elapsed_seconds=elapsed_seconds
        )
        if observed is None:
            return ()
        stages = (observed,) if isinstance(observed, Mapping) else observed
        if (
            isinstance(stages, (str, bytes))
            or not isinstance(stages, Sequence)
            or any(not isinstance(stage, Mapping) for stage in stages)
        ):
            raise TypeError("accepted tracker output must contain objects")
        return tuple(self._process_observation(rgb, stage) for stage in stages)

    def _process_observation(
        self, current_rgb: np.ndarray, stage: Mapping[str, object]
    ) -> Mapping[str, object]:
        reference = np.asarray(stage["same_run_empty_reference_rgb"])
        observation_rgb = np.asarray(stage.get("observation_rgb", current_rgb))
        if "prediction_mask" in stage:
            mask = np.asarray(stage["prediction_mask"])
            occupied = np.flatnonzero(np.any(mask, axis=1))
            if (
                mask.dtype != np.bool_
                or mask.shape != observation_rgb.shape[:2]
                or occupied.size == 0
            ):
                raise TypeError("accepted tracker prediction mask is malformed")
            prediction = SimpleNamespace(
                status="visible_measured", mask=mask,
                surface_row=int(stage.get("surface_row", occupied[0])),
                visible_height_px=int(stage.get(
                    "visible_height_px", occupied[-1] - occupied[0] + 1
                )),
            )
        else:
            prediction = self._predictor(
                self.cnn_model, observation_rgb, reference, device=self.device
            )
        if prediction.status != "visible_measured":
            measurement = {
                "validity": "invalid", "median_lab": None,
                "limit_codes": [f"cnn_{prediction.status}"],
                "evidence_id": stage["evidence_id"],
                "measured_or_inferred": "measured",
            }
        else:
            measurement = self._measure(observation_rgb, prediction.mask, stage)
        self._retain_vlm_evidence(stage, measurement)
        public_tracker = {
            key: value for key, value in stage.items()
            if key in PUBLIC_TRACKER_FIELDS
        }
        return {
            "vial_id": stage["vial_id"],
            "vial_position": stage["vial_position"],
            "fill_step": stage["fill_step"],
            "tank_id": stage["tank_id"],
            "phase": stage.get("phase"),
            "tracker": public_tracker,
            "mask": {
                "status": prediction.status,
                "pixel_count": int(np.count_nonzero(prediction.mask)),
                "surface_row": prediction.surface_row,
                "visible_height_px": prediction.visible_height_px,
            },
            "measurement": measurement,
            "closed": stage.get("closed") is True,
        }

    def close_tracker_vial(self) -> tuple[Mapping[str, object], ...]:
        """Drain retrospective final-phase observations at a vial boundary."""

        closer = getattr(self.tracker, "close", None)
        if not callable(closer):
            raise TypeError("accepted tracker has no vial-close contract")
        observed = closer()
        if (
            isinstance(observed, (str, bytes))
            or not isinstance(observed, Sequence)
            or any(not isinstance(stage, Mapping) for stage in observed)
        ):
            raise TypeError("accepted tracker close output must contain objects")
        return tuple(
            self._process_observation(np.asarray(stage["observation_rgb"]), stage)
            for stage in observed
        )

    def decide_cohort(self, vials: Mapping[str, object]) -> Mapping[str, object]:
        """Run deterministic comparison only after all three closed vials exist."""
        if self._cohort_context is None:
            decision = self.deterministic_decider(vials)
        else:
            cohort = self._cohort_without_decision(vials)
            decision = self.deterministic_decider(cohort)
            self._causal_cohort = deepcopy(cohort)
            self._deterministic_receipt = deepcopy(dict(decision))
        public = dict(decision)
        if public.get("discard_score") is None:
            public["discard_score"] = {
                "keep": 0.0, "discard": 1.0,
            }.get(public.get("disposition"))
        if "evidence_citations" not in public:
            stages = public.get("stages", [])
            public["evidence_citations"] = list(dict.fromkeys(
                row["evidence_id"] for row in stages
                if isinstance(row, Mapping) and isinstance(row.get("evidence_id"), str)
            ))
        public.setdefault("limit_codes", [])
        return public

    @staticmethod
    def _canonical(value: object) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()

    def _retain_vlm_evidence(self, stage: Mapping[str, object],
                             measurement: Mapping[str, object]) -> None:
        context = stage.get("vlm_cohort_context")
        if context is not None:
            if not isinstance(context, Mapping):
                raise TypeError("VLM cohort context must be an object")
            if (self._cohort_context is not None
                    and self._canonical(context) != self._canonical(self._cohort_context)):
                if stage.get("vial_position") != 1:
                    raise ValueError("VLM cohort context drifted")
                self._image_locators.clear()
                self._tracker_receipts.clear()
                self._vlm_vials.clear()
                self._measurements.clear()
                self._deterministic_receipt = None
                self._causal_cohort = None
            self._cohort_context = deepcopy(dict(context))
        stage_id = stage.get("stage_id")
        if stage_id is not None:
            if not isinstance(stage_id, str) or not stage_id:
                raise TypeError("VLM stage ID must be nonempty text")
            allowed = {
                key: deepcopy(measurement[key]) for key in (
                    "evidence_id", "median_lab", "mad_lab", "iqr_lab", "limit_codes"
                ) if key in measurement
            }
            self._measurements[(str(stage["vial_id"]), stage_id)] = allowed
        if stage.get("closed") is not True:
            return
        receipt, vial = stage.get("tracker_receipt"), stage.get("vlm_vial")
        if receipt is None and vial is None:
            return
        if not isinstance(receipt, Mapping) or not isinstance(vial, Mapping):
            raise TypeError("closed VLM vial evidence is incomplete")
        vial_id = str(stage["vial_id"])
        if receipt.get("vial_id") != vial_id or vial.get("vial_id") != vial_id:
            raise ValueError("closed VLM vial identity drifted")
        checked_vial = deepcopy(dict(vial))
        stages = checked_vial.get("stages")
        if not isinstance(stages, list):
            raise TypeError("closed VLM vial stages must be a list")
        for vial_stage in stages:
            if not isinstance(vial_stage, dict) or not isinstance(vial_stage.get("stage_id"), str):
                raise TypeError("closed VLM vial stage is malformed")
            key = (vial_id, vial_stage["stage_id"])
            if key not in self._measurements:
                raise RuntimeError("measured VLM stage evidence is unavailable")
            vial_stage["measurement"] = deepcopy(self._measurements[key])
        self._tracker_receipts[vial_id] = deepcopy(dict(receipt))
        self._vlm_vials[vial_id] = checked_vial

    def _cohort_without_decision(self, vials: Mapping[str, object]) -> dict[str, object]:
        if self._cohort_context is None or set(vials) != set(self._vlm_vials):
            raise RuntimeError("complete internal VLM cohort evidence is unavailable")
        ordered = sorted(self._vlm_vials.values(), key=lambda row: int(row["vial_position"]))
        return {**deepcopy(dict(self._cohort_context)), "vials": deepcopy(ordered)}

    def _measure(
        self, rgb: np.ndarray, mask: np.ndarray, stage: Mapping[str, object]
    ) -> dict[str, object]:
        root = Path(self._artifacts.name)
        image_id = str(stage.get("image_id") or (
            f"{stage['vial_id']}-{stage['fill_step']}-{stage['evidence_id']}"
        ))
        artifact_id = hashlib.sha256(image_id.encode("utf-8")).hexdigest()
        rgb_path = root / f"{artifact_id}-rgb.png"
        mask_path = root / f"{artifact_id}-mask.png"
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(rgb_path)
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(mask_path)
        locator = {
            "rgb_path": str(rgb_path.resolve()),
            "rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
            "mask_path": str(mask_path.resolve()),
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        }
        vlm_image_id = stage.get("image_id")
        if vlm_image_id is not None:
            if not isinstance(vlm_image_id, str) or not vlm_image_id:
                raise TypeError("VLM image ID must be nonempty text")
            crop, crop_binding = self._crop_binding(rgb, stage)
            crop_path = root / f"{artifact_id}-vial-roi.png"
            Image.fromarray(crop, mode="RGB").save(crop_path)
            bound = {
                "path": str(crop_path.resolve()),
                "sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest(),
                **crop_binding,
            }
            prior = self._image_locators.get(vlm_image_id)
            if prior is not None and prior != bound:
                raise ValueError("VLM image ID was rebound")
            self._image_locators[vlm_image_id] = bound
        measured = self._measurer(image_id, locator, _include_dispersion=True)
        measured["evidence_id"] = stage["evidence_id"]
        measured["measured_or_inferred"] = "measured"
        return measured

    def _crop_binding(
        self, rgb: np.ndarray, stage: Mapping[str, object]
    ) -> tuple[np.ndarray, dict[str, object]]:
        context = stage.get("vlm_cohort_context", self._cohort_context)
        if not isinstance(context, Mapping):
            raise RuntimeError("VLM cohort context is unavailable for native ROI crop")
        if (
            self._cohort_context is not None
            and self._canonical(context) != self._canonical(self._cohort_context)
        ):
            raise ValueError("VLM cohort context drifted")
        camera = context.get("camera_metadata")
        if not isinstance(camera, Mapping):
            raise RuntimeError("VLM camera metadata is unavailable")
        resolution, roi = camera.get("resolution"), camera.get("vial_roi")
        if resolution != [int(rgb.shape[1]), int(rgb.shape[0])]:
            raise ValueError("VLM camera resolution differs from native frame")
        crop, binding = crop_vial_roi(rgb, roi)
        return crop, binding

    def build_vlm_bundle(self, vials: Mapping[str, object]) -> object:
        if self._cohort_context is None:
            return self.vlm_bundle_builder(vials)
        if self._deterministic_receipt is None:
            raise RuntimeError("causal deterministic receipt is unavailable")
        if self._causal_cohort is None:
            raise RuntimeError("causal cohort evidence is unavailable")
        cohort = {
            **deepcopy(self._causal_cohort),
            "deterministic_decision_receipt": deepcopy(dict(self._deterministic_receipt)),
        }
        self._bind_target_decision_measurements(
            cohort, cohort["deterministic_decision_receipt"]
        )
        ordered_ids = [str(row["vial_id"]) for row in cohort["vials"]]
        if set(ordered_ids) != set(self._tracker_receipts):
            raise RuntimeError("complete tracker receipt cohort is unavailable")
        return self.vlm_bundle_builder(
            cohort, deepcopy(self._image_locators),
            tracker_receipts=[deepcopy(self._tracker_receipts[key]) for key in ordered_ids],
        )

    @staticmethod
    def _bind_target_decision_measurements(
        cohort: dict[str, object], receipt: Mapping[str, object]
    ) -> None:
        stages = receipt.get("stages")
        vials = cohort.get("vials")
        if not isinstance(stages, list) or not isinstance(vials, list):
            raise RuntimeError("deterministic stage evidence is unavailable")
        by_stage = {
            row.get("stage_id"): row for row in stages if isinstance(row, Mapping)
        }
        target = next(
            (row for row in vials if isinstance(row, Mapping)
             and row.get("vial_position") == 1),
            None,
        )
        if not isinstance(target, Mapping) or not isinstance(target.get("stages"), list):
            raise RuntimeError("target vial stage evidence is unavailable")
        for stage in target["stages"]:
            if not isinstance(stage, dict) or not isinstance(stage.get("measurement"), dict):
                raise RuntimeError("target measurement evidence is unavailable")
            decision = by_stage.get(stage.get("stage_id"))
            if not isinstance(decision, Mapping):
                raise RuntimeError("deterministic stage evidence is incomplete")
            stage["measurement"].update({
                "delta_e76_vial2": decision.get("delta_e76_vial2"),
                "delta_e76_vial3": decision.get("delta_e76_vial3"),
                "deterministic_disposition": decision.get("disposition"),
                "limit_codes": deepcopy(decision.get("limit_codes")),
            })

    def run_vlm(self, bundle: object):
        return self.vlm_runner(bundle)

    def close(self) -> None:
        self._artifacts.cleanup()
