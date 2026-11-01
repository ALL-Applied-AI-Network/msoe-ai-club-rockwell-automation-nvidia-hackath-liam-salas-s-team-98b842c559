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
from ...wp7_vlm_image_support import crop_vial_roi


PUBLIC_TRACKER_FIELDS = frozenset({
    "agreement", "measured_volume_ml", "twin_geometry_equivalent_volume_ml",
    "expected_dispense_ml", "expected_cumulative_ml", "stream_step_index",
    "tank_cluster_consistent", "limit_codes",
})
MEASUREMENT_MODES = frozenset({"generic_live", "pinned_development_stage_stills"})


def _aggregate_masked_lab(
    frames: Sequence[object], masks: Sequence[object]
) -> dict[str, object]:
    """Aggregate the frozen early/mid/end masked pixels for one live stage."""
    if len(frames) != 3 or len(masks) != 3:
        raise ValueError("exactly early/mid/end frames and masks are required")
    from skimage.color import rgb2lab
    selected = []
    for frame_raw, mask_raw in zip(frames, masks, strict=True):
        frame, mask = np.asarray(frame_raw), np.asarray(mask_raw)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be uint8 RGB")
        if mask.dtype != np.bool_ or mask.shape != frame.shape[:2] or not np.any(mask):
            raise ValueError("CNN mask is empty or mismatched")
        selected.append(rgb2lab(frame.astype(np.float64) / 255.0)[mask])
    values = np.concatenate(selected, axis=0)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    q25, q75 = np.percentile(values, (25.0, 75.0), axis=0)
    return {
        "median_lab": [float(value) for value in median],
        "mad_lab": [float(value) for value in mad],
        "iqr_lab": [float(value) for value in q75 - q25],
    }


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
        measurement_mode: str = "generic_live",
        predictor: Callable[..., object] | None = None,
        measurer: Callable[..., dict[str, object]] = measure_image,
    ):
        if measurement_mode not in MEASUREMENT_MODES:
            raise ValueError("accepted adapter measurement mode is unavailable")
        self.tracker = tracker
        self.cnn_model = cnn_model
        self.deterministic_decider = deterministic_decider
        self.vlm_bundle_builder = vlm_bundle_builder
        self.vlm_runner = vlm_runner
        self.device = device
        self.model_id = model_id
        self.measurement_mode = measurement_mode
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
        self._phase_visuals: dict[
            tuple[str, str], dict[str, str]
        ] = {}
        self._phase_evidence: dict[
            tuple[str, str], dict[str, dict[str, object]]
        ] = {}
        self._stage_identities: dict[tuple[str, str], tuple[object, object]] = {}

    @property
    def requires_complete(self) -> bool:
        return bool(getattr(self.tracker, "requires_complete", False))

    @property
    def session_complete(self) -> bool:
        return bool(getattr(self.tracker, "session_complete", False))

    @property
    def source_profile_evidence(self) -> object:
        return deepcopy(getattr(self.tracker, "source_profile_evidence", None))

    def configure_source_profile(self, profile: Mapping[str, object]) -> None:
        configure = getattr(self.tracker, "configure_source_profile", None)
        if not callable(configure):
            raise TypeError("accepted tracker cannot bind a source profile")
        configure(profile)

    def end_of_stream(self) -> tuple[Mapping[str, object], ...]:
        closer = getattr(self.tracker, "end_of_stream", None)
        if not callable(closer):
            return ()
        observed = closer()
        return tuple(
            self._process_observation(np.asarray(stage["observation_rgb"]), stage)
            for stage in observed
        )

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
            ):
                raise TypeError("accepted tracker prediction mask is malformed")
            if occupied.size == 0:
                prediction = SimpleNamespace(
                    status=str(stage.get("height_status") or "topology_unavailable"),
                    mask=mask, surface_row=None, visible_height_px=None,
                )
            else:
                prediction = SimpleNamespace(
                    status="visible_measured", mask=mask,
                    surface_row=int(stage.get("surface_row", occupied[0])),
                    visible_height_px=int(stage.get(
                        "visible_height_px", occupied[-1] - occupied[0] + 1
                    )),
                )
        else:
            predictor = self._predictor
            if predictor is None:
                from ...cnn_predictor import predict_pair
                predictor = predict_pair
            prediction = predictor(
                self.cnn_model, observation_rgb, reference, device=self.device
            )
        if prediction.status != "visible_measured":
            measurement = {
                "validity": "invalid", "median_lab": None,
                "limit_codes": [f"cnn_{prediction.status}"],
                "evidence_id": stage["evidence_id"],
                "measured_or_inferred": "measured",
            }
            self._retain_image_locator(observation_rgb, stage)
        else:
            measurement = self._measure(observation_rgb, prediction.mask, stage)
        self._retain_vlm_evidence(
            stage, measurement, observation_rgb, np.asarray(prediction.mask)
        )
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

    def close_tracker_vial(
        self, boundary: Mapping[str, object] | None = None
    ) -> tuple[Mapping[str, object], ...]:
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
        results = tuple(
            self._process_observation(np.asarray(stage["observation_rgb"]), stage)
            for stage in observed
        )
        if boundary is not None:
            if not results or results[-1].get("closed") is not True:
                raise RuntimeError("provenance-bound tracker close did not close a vial")
            if (
                results[-1].get("vial_id") != boundary.get("vial_id")
                or results[-1].get("vial_position") != boundary.get("vial_position")
            ):
                raise ValueError("tracker close identity differs from boundary provenance")
            receipt = observed[-1].get("tracker_receipt")
            if (
                not isinstance(receipt, Mapping)
                or receipt.get("source_artifact_sha256")
                != boundary.get("source_sha256")
            ):
                raise ValueError("tracker receipt source differs from boundary provenance")
        return results

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

    def _retain_vlm_evidence(
        self, stage: Mapping[str, object], measurement: Mapping[str, object],
        observation_rgb: np.ndarray, prediction_mask: np.ndarray,
    ) -> None:
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
                self._phase_visuals.clear()
                self._phase_evidence.clear()
                self._stage_identities.clear()
                self._deterministic_receipt = None
                self._causal_cohort = None
            self._cohort_context = deepcopy(dict(context))
        stage_id = stage.get("stage_id")
        if stage_id is not None:
            if not isinstance(stage_id, str) or not stage_id:
                raise TypeError("VLM stage ID must be nonempty text")
            phase = stage.get("phase")
            if phase not in {"early", "mid", "end"}:
                raise ValueError("VLM stage phase must be early, mid, or end")
            image_id = stage.get("image_id")
            if not isinstance(image_id, str) or not image_id:
                raise TypeError("VLM phase image ID must be nonempty text")
            key = (str(stage["vial_id"]), stage_id)
            identity = (stage.get("fill_step"), stage.get("tank_id"))
            prior_identity = self._stage_identities.setdefault(key, identity)
            if prior_identity != identity:
                raise ValueError("VLM phase evidence stage identity drifted")
            phases = self._phase_visuals.setdefault(key, {})
            if phase in phases:
                raise ValueError("VLM stage phase evidence was duplicated")
            phases[phase] = image_id
            if self.measurement_mode == "generic_live":
                evidence = self._phase_evidence.setdefault(key, {})
                evidence[phase] = {
                    "frame": np.array(observation_rgb, copy=True),
                    "mask": np.array(prediction_mask, copy=True),
                    "measurement": deepcopy(dict(measurement)),
                }
                if set(evidence) == {"early", "mid", "end"}:
                    ordered = [evidence[name] for name in ("early", "mid", "end")]
                    if all(np.any(row["mask"]) for row in ordered):
                        statistics = _aggregate_masked_lab(
                            [row["frame"] for row in ordered],
                            [row["mask"] for row in ordered],
                        )
                        self._measurements[key] = {
                            "evidence_id": ordered[-1]["measurement"]["evidence_id"],
                            **statistics,
                            "limit_codes": list(dict.fromkeys(
                                code for row in ordered
                                for code in row["measurement"].get("limit_codes", [])
                            )),
                        }
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
        if not stages:
            self._tracker_receipts[vial_id] = deepcopy(dict(receipt))
            self._vlm_vials[vial_id] = checked_vial
            return
        self._validate_closed_phase_topology(vial_id, stages)
        for vial_stage in stages:
            if self.measurement_mode == "generic_live":
                key = (vial_id, str(vial_stage.get("stage_id")))
                if key not in self._measurements:
                    raise RuntimeError("measured VLM stage evidence is unavailable")
                vial_stage["measurement"] = deepcopy(self._measurements[key])
            measurement = vial_stage.get("measurement")
            if (
                not isinstance(measurement, dict)
                or set(("evidence_id", "median_lab", "mad_lab", "iqr_lab", "limit_codes"))
                - set(measurement)
            ):
                raise RuntimeError("pinned stage-still measurement is unavailable")
        self._tracker_receipts[vial_id] = deepcopy(dict(receipt))
        self._vlm_vials[vial_id] = checked_vial

    def _validate_closed_phase_topology(
        self, vial_id: str, stages: list[object]
    ) -> None:
        if self.measurement_mode == "pinned_development_stage_stills" and len(stages) != 3:
            raise RuntimeError("pinned vial requires three measured stages")
        expected_keys: set[tuple[str, str]] = set()
        for vial_stage in stages:
            if not isinstance(vial_stage, dict) or not isinstance(vial_stage.get("stage_id"), str):
                raise TypeError("closed VLM vial stage is malformed")
            stage_id = vial_stage["stage_id"]
            key = (vial_id, stage_id)
            expected_keys.add(key)
            phases = self._phase_visuals.get(key)
            if not isinstance(phases, dict) or set(phases) != {"early", "mid", "end"}:
                raise RuntimeError("complete early/mid/end phase topology is unavailable")
            frames = vial_stage.get("frames")
            if not isinstance(frames, list) or len(frames) != 3:
                raise RuntimeError("complete early/mid/end phase topology is unavailable")
            bound = {
                frame.get("phase"): frame.get("image_id")
                for frame in frames if isinstance(frame, Mapping)
            }
            if bound != phases:
                raise ValueError("closed VLM frame binding differs from phase visuals")
        observed_keys = {
            key for key in self._phase_visuals if key[0] == vial_id
        }
        if observed_keys != expected_keys:
            raise RuntimeError("closed VLM stages differ from phase topology")

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
        self._retain_image_locator(rgb, stage)
        measured = self._measurer(image_id, locator, _include_dispersion=True)
        measured["evidence_id"] = stage["evidence_id"]
        measured["measured_or_inferred"] = "measured"
        return measured

    def _retain_image_locator(
        self, rgb: np.ndarray, stage: Mapping[str, object]
    ) -> None:
        vlm_image_id = stage.get("image_id")
        if vlm_image_id is None:
            return
        if not isinstance(vlm_image_id, str) or not vlm_image_id:
            raise TypeError("VLM image ID must be nonempty text")
        crop, crop_binding = self._crop_binding(rgb, stage)
        artifact_id = hashlib.sha256(vlm_image_id.encode("utf-8")).hexdigest()
        crop_path = Path(self._artifacts.name) / f"{artifact_id}-vial-roi.png"
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
