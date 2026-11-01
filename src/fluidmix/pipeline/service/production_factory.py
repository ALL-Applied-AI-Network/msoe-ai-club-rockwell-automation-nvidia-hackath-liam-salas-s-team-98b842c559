"""Frozen V3/V4 production factory for operator-bound video sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .accepted_adapter import AcceptedPipelineAdapter
from .contract import PIPELINE_COMMIT


ARTIFACT_ROOT = Path("/home/hackathon5/liamreporoot")
PIPELINE_ROOT = ARTIFACT_ROOT / "outputs/pipeline-20261028"
CHECKPOINT = PIPELINE_ROOT / "wp5-cnn-predictor-01a18128/candidate-a5f29ea/resnet18-unet-current-empty-rgb.pt"
REPLAY = PIPELINE_ROOT / "wp5-cnn-harness-recovery-01a18128/replay-9edd739/deterministic-bounds-replay.json"
REVIEW = PIPELINE_ROOT / "wp5-cnn-harness-recovery-01a18128/replay-9edd739/independent-review-root-wp5_cnn_replay_review.md"
TRACKER_MATERIALIZATION = PIPELINE_ROOT / "wp7-fill-step-tracker-dev-01a18128-attempt04/development-materialization-receipt.json"
TRACKER_TRAIN_FREEZE = PIPELINE_ROOT / "wp7-fill-step-tracker-dev-01a18128-attempt04/attempt04-train-freeze.json"
TOPOLOGY_FREEZE = PIPELINE_ROOT / "wp7-fill-step-topology-freeze-01a18128-attempt03/topology-freeze.json"
VLM_FREEZE = PIPELINE_ROOT / "wp7-vlm-development-freeze-01a18128-attempt18-final-review-blockers/freeze.json"
CALIBRATION = PIPELINE_ROOT / "wp7-video5-attempt03-preclaim-01a18128/pixel-calibration.json"

EXPECTED_HASHES = {
    CHECKPOINT: "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba",
    REPLAY: "a4b9d412ce063fa6449fbfcacf49b1b3ba3e7d64724fd93d96381c467ee4836a",
    REVIEW: "17e1bb4809898c95b3af59cb6d187955a7bf36b4f6fffb39e342e07a59f2c8df",
    TRACKER_MATERIALIZATION: "255319ca4df8e668b0bceaaf211642cabcd59983a4a023163b0bfbf5a8a323ce",
    TRACKER_TRAIN_FREEZE: "876a2c5813eba6f2cbd018a16c144e5a9d28ca9cbf1170ba4529e5c212392206",
    TOPOLOGY_FREEZE: "62e56a9e2423b09b25cf735452ebdd36adfa056bf8f8821c3fc173dbc6415a3f",
    VLM_FREEZE: "8d5a884c8bf7438b291816473132d5c61f24c3d34d3c3654c827567006aae3e5",
    CALIBRATION: "a8963d8081999ed8b011ef7078d9f9358938c29dae1073a56a0f4daf7ffd23f5",
}
MODEL_REVISIONS = {
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "nvidia/Cosmos-Reason2-8B": "a9fae2cf89dc64db96b12860417f0eb403013bb9",
    "aacudad/AnomalyThink-Qwen3-VL-8B-KCR": "8331bc79ecf01e3466f0e6a27cec8c77f182f430",
}
MODEL_INVENTORY_SHA256 = {
    "Qwen/Qwen3-VL-8B-Instruct": "0d6bfb3639d237421ebf68726799c106ee09645173935de711a67c365bd645b4",
    "nvidia/Cosmos-Reason2-8B": "ee160e941c128236941f2a854617bc0cc448adba1ebf0d8d2f028f573c928e5a",
    "aacudad/AnomalyThink-Qwen3-VL-8B-KCR": "86694fb1fdb326c2d6a6b1d10c810ff08fc518a67907d884164178befcbc6c8d",
}


class ProductionFactoryError(RuntimeError):
    pass


def validate_session_profile(request: Mapping[str, object]) -> None:
    source, context = request.get("source"), request.get("run_context")
    if (
        not isinstance(source, Mapping)
        or source.get("kind") not in {"file", "video", "v4l2", "rtsp"}
    ):
        raise ProductionFactoryError("frozen profile requires an admitted video transport")
    if not isinstance(context, Mapping) or str(context.get("recipe_id")) != "5":
        raise ProductionFactoryError("frozen profile supports only recipe 5")


def model_attempt_rows(aggregate: object) -> tuple[Mapping[str, object], ...]:
    """Project the aggregate receipt to the per-model rows expected by the runner."""
    if not isinstance(aggregate, Mapping) or not isinstance(aggregate.get("sources"), list):
        raise ProductionFactoryError("VLM aggregate source rows are unavailable")
    rows = tuple(
        row for row in aggregate["sources"]
        if isinstance(row, Mapping) and isinstance(row.get("repository"), str)
    )
    if len(rows) != len(MODEL_REVISIONS):
        raise ProductionFactoryError("VLM aggregate does not contain three model attempts")
    return rows


def deterministic_stage_measurements(
    cohort: Mapping[str, object],
) -> list[list[dict[str, object]]]:
    vials = cohort.get("vials")
    if not isinstance(vials, Sequence) or len(vials) != 3:
        raise ProductionFactoryError("deterministic cohort is incomplete")
    stage_count = len(vials[0].get("stages", [])) if isinstance(vials[0], Mapping) else 0
    result: list[list[dict[str, object]]] = []
    for ordinal in range(stage_count):
        aligned = []
        for vial in vials:
            if not isinstance(vial, Mapping) or not isinstance(vial.get("stages"), list):
                raise ProductionFactoryError("deterministic vial stages are incomplete")
            stage = vial["stages"][ordinal]
            if not isinstance(stage, Mapping) or not isinstance(stage.get("measurement"), Mapping):
                raise ProductionFactoryError("deterministic stage measurement is incomplete")
            aligned.append({
                **deepcopy(dict(stage["measurement"])),
                "stage_id": stage.get("stage_id"),
            })
        result.append(aligned)
    return result


def prepare_cnn_model(model: object, device: str) -> object:
    if device not in {"cpu", "cuda"}:
        raise ProductionFactoryError("selected CNN device is invalid")
    selected = deepcopy(model) if device == "cuda" else model
    mover = getattr(selected, "to", None)
    if not callable(mover):
        raise ProductionFactoryError("loaded CNN does not support device selection")
    moved = mover(device)
    return selected if moved is None else moved


def normalize_tracker_frame(frame: object) -> np.ndarray:
    """Validate and retain the accepted contiguous RGB tracker byte contract."""
    rgb = np.asarray(frame)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ProductionFactoryError("tracker frame must be uint8 RGB")
    return np.ascontiguousarray(rgb)


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ProductionFactoryError(f"accepted artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measurement_mode_for_source(source: object) -> str:
    del source
    return "generic_live"


@dataclass(frozen=True)
class FrozenTiming:
    """Time interpretation of frame counts frozen against the 10 Hz V4 route."""

    source_fps: float
    canonical_fps: float
    stream_debounce_seconds: float
    stable_tail_seconds: float
    stream_debounce_source_frames: int
    stable_tail_source_frames: int

    @classmethod
    def for_source_fps(cls, source_fps: object) -> "FrozenTiming":
        if (
            isinstance(source_fps, bool)
            or not isinstance(source_fps, (int, float))
            or not math.isfinite(float(source_fps))
            or float(source_fps) <= 0.0
        ):
            raise ProductionFactoryError("source frame rate must be finite and positive")
        fps = float(source_fps)
        # Frozen V4: 12 frames / 10 Hz = 1.2 s stream-state debounce;
        # 3 frames / 10 Hz = 0.3 s stable-tail and hidden-stream support.
        debounce_seconds, stable_seconds = 12 / 10.0, 3 / 10.0
        return cls(
            source_fps=fps, canonical_fps=10.0,
            stream_debounce_seconds=debounce_seconds,
            stable_tail_seconds=stable_seconds,
            stream_debounce_source_frames=math.ceil(debounce_seconds * fps),
            stable_tail_source_frames=math.ceil(stable_seconds * fps),
        )


def build_session_cohort(
    run_context: Mapping[str, object], *, frame_rate_hz: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Create identities from operator input; no per-file case context is reused."""
    operator_run_id = str(run_context["operator_run_id"])
    tray_id = str(run_context["tray_id"])
    recipe_id = str(run_context["recipe_id"])
    cohort_id = f"{operator_run_id}:{tray_id}"
    context: dict[str, object] = {
        "schema_version": "wp7-vlm-cohort-v1", "cohort_id": cohort_id,
        "partition": "production", "recipe_id": recipe_id,
        "previous_recipe_id": None,
        "camera_metadata": {
            "resolution": [640, 480], "frame_rate_hz": float(frame_rate_hz),
            "exposure": "unavailable", "white_balance": "unavailable",
            "viewpoint_id": "fillcam-proportional-v1",
            "vial_roi": [283, 194, 338, 417],
        },
        "stages": [],
    }
    vials = [{
        "vial_position": position,
        "vial_id": f"{cohort_id}:vial-{position}", "stages": [],
    } for position in (1, 2, 3)]
    return context, vials


def normalize_fillcam_view(frame: object) -> tuple[np.ndarray, dict[str, object]]:
    """Scale a proportional 4:3 FillCam view to the frozen 640x480 raster."""
    rgb = normalize_tracker_frame(frame)
    height, width = rgb.shape[:2]
    if width * 3 != height * 4 or min(width, height) < 120 or not np.any(rgb):
        raise ProductionFactoryError(
            "camera view is incompatible with frozen FillCam geometry"
        )
    native_roi = [
        math.floor(283 * width / 640), math.floor(194 * height / 480),
        math.ceil(338 * width / 640), math.ceil(417 * height / 480),
    ]
    if (width, height) == (640, 480):
        canonical = rgb
    else:
        from PIL import Image
        canonical = np.asarray(Image.fromarray(rgb, mode="RGB").resize(
            (640, 480), resample=Image.Resampling.BILINEAR,
        ))
    evidence = {
        "native_resolution": [width, height], "canonical_resolution": [640, 480],
        "native_vial_roi": native_roi, "canonical_vial_roi": [283, 194, 338, 417],
        "scale_x": width / 640.0, "scale_y": height / 480.0,
        "source_full_frame_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "native_vial_roi_pixel_sha256": hashlib.sha256(
            rgb[native_roi[1]:native_roi[3], native_roi[0]:native_roi[2]].tobytes()
        ).hexdigest(),
    }
    return np.ascontiguousarray(canonical), evidence


def validate_artifacts(expected_hashes: Mapping[Path, str] = EXPECTED_HASHES) -> None:
    for path, expected in expected_hashes.items():
        if _sha(Path(path)) != expected:
            raise ProductionFactoryError(f"accepted artifact SHA-256 mismatch: {path}")
    freeze = json.loads(VLM_FREEZE.read_bytes())
    if freeze.get("schema_version") != "wp7-vlm-development-freeze-v12" or {
        row.get("repository"): row.get("revision")
        for row in freeze.get("models", []) if isinstance(row, Mapping)
    } != MODEL_REVISIONS:
        raise ProductionFactoryError("accepted VLM freeze identity drifted")


def _snapshot_paths() -> dict[str, Path]:
    root = PIPELINE_ROOT / "wp6-model-weights-01a18128"
    directory_names = {
        "Qwen/Qwen3-VL-8B-Instruct": "qwen3-vl-8b-instruct",
        "nvidia/Cosmos-Reason2-8B": "cosmos-reason2-8b",
        "aacudad/AnomalyThink-Qwen3-VL-8B-KCR": "anomalythink-qwen3-vl-8b-kcr",
    }
    from fluidmix.wp7_stock_runtime import snapshot_inventory
    result = {}
    for repository, revision in MODEL_REVISIONS.items():
        path = root / directory_names[repository]
        if path.is_symlink() or not path.is_dir():
            raise ProductionFactoryError(f"pinned local model snapshot unavailable: {repository}")
        if snapshot_inventory(path)["sha256"] != MODEL_INVENTORY_SHA256[repository]:
            raise ProductionFactoryError(f"pinned local model inventory drifted: {repository}")
        result[repository] = path
    return result


class SessionVialTracker:
    """Find three vial windows from debounced stream states and a stable tail."""

    def __init__(self, builder: Callable[[int, np.ndarray], object], *, auto_close=False,
                 stream_state: Callable[[np.ndarray, np.ndarray], str | None] | None = None,
                 profile_bound: Callable[[FrozenTiming, Mapping[str, object]], None] | None = None):
        self._builder = builder
        self._trackers: list[object | None] = [None, None, None]
        self._active = 0
        self._auto_close = bool(auto_close)
        self._stream_state = stream_state
        self._profile_bound = profile_bound
        self._timing: FrozenTiming | None = None
        self._reference: np.ndarray | None = None
        self._window_started_seconds: float | None = None
        self._canonical_index = 0
        self._states: list[str] = []
        self._candidate_state: str | None = None
        self._candidate_count = 0
        self._inactive_count = 0
        self._seam_empty_count = 0
        self._seam_frames: list[tuple[np.ndarray, np.ndarray]] = []
        self._profile_evidence: dict[str, object] | None = None
        self._source_resolution: list[int] | None = None
        self._native_frames: dict[int, tuple[np.ndarray, dict[str, object]]] = {}
        self._window_hash = hashlib.sha256()

    def configure_source_profile(self, profile: Mapping[str, object]) -> None:
        if self._timing is not None:
            raise ProductionFactoryError("source profile is already configured")
        resolution = profile.get("resolution")
        if (
            not isinstance(resolution, list) or len(resolution) != 2
            or any(type(value) is not int or value <= 0 for value in resolution)
            or resolution[0] * 3 != resolution[1] * 4
        ):
            raise ProductionFactoryError(
                "camera view is incompatible with frozen FillCam geometry"
            )
        self._timing = FrozenTiming.for_source_fps(profile.get("fps"))
        self._source_resolution = list(resolution)
        if self._profile_bound is not None:
            self._profile_bound(self._timing, profile)
        self._profile_evidence = {
            "fps": self._timing.source_fps,
            "fps_source": profile.get("fps_source", "injected_test_profile"),
            "resolution": list(resolution),
            "timing_conversions": {
                "stream_debounce": "12 frames / 10 Hz = 1.2 seconds",
                "stable_tail": "3 frames / 10 Hz = 0.3 seconds",
            },
        }

    @property
    def completed_vial_count(self) -> int:
        return self._active

    @property
    def source_profile_evidence(self) -> Mapping[str, object] | None:
        """Internal receipt data; intentionally not added to runner event schemas."""
        return deepcopy(self._profile_evidence)

    @property
    def requires_complete(self) -> bool:
        return self._auto_close

    @property
    def session_complete(self) -> bool:
        return self._auto_close and self._active == 3

    def observe(self, frame: object, *, frame_index: int, elapsed_seconds: float):
        if self._active >= 3:
            return ()
        if self._timing is None:
            # Direct unit callers retain an explicit, deterministic profile.
            self.configure_source_profile({"fps": 10.0, "resolution": [640, 480]})
        assert self._timing is not None
        if self._auto_close:
            native_rgb = normalize_tracker_frame(frame)
            if [native_rgb.shape[1], native_rgb.shape[0]] != self._source_resolution:
                raise ProductionFactoryError(
                    "camera view is incompatible with frozen FillCam geometry"
                )
            rgb, native_evidence = normalize_fillcam_view(native_rgb)
            self._profile_evidence = {**(self._profile_evidence or {}),
                                      "last_native_view": native_evidence}
        else:
            rgb = normalize_tracker_frame(frame)
        source_pixels = native_rgb if self._auto_close else rgb
        if self._reference is None:
            self._reference = rgb.copy()
            self._window_hash.update(memoryview(source_pixels).cast("B"))
            return ()
        if self._trackers[self._active] is None and self._stream_state is not None:
            state = self._stream_state(rgb, self._reference)
            if state is None:
                self._window_hash.update(memoryview(source_pixels).cast("B"))
                return ()
            self._trackers[self._active] = self._builder(self._active, self._reference)
        else:
            if self._trackers[self._active] is None:
                self._trackers[self._active] = self._builder(self._active, self._reference)
            state = self._detect_state(rgb)
        if state is None and self._canonical_index == 0:
            self._window_hash.update(memoryview(source_pixels).cast("B"))
            return ()
        tracker = self._trackers[self._active]
        expected = ["t200", "t400", "t100"]
        complete = self._states[-3:] == expected if len(self._states) >= 3 else False
        if self._auto_close and complete:
            topology_empty = self._topology_cleaned_empty(tracker, rgb)
            if topology_empty is True:
                self._seam_empty_count += 1
                self._seam_frames.append((rgb.copy(), source_pixels.copy()))
                if self._seam_empty_count >= self._timing.stable_tail_source_frames:
                    binder = getattr(tracker, "bind_source_artifact_sha256", None)
                    if not callable(binder):
                        raise ProductionFactoryError(
                            "dynamic tracker cannot bind source provenance"
                        )
                    binder(self._window_hash.hexdigest())
                    next_prefix = list(self._seam_frames)
                    closed = self.close()
                    self._window_hash = hashlib.sha256()
                    self._reference = next_prefix[0][0]
                    for _canonical, source_frame in next_prefix:
                        self._window_hash.update(memoryview(source_frame).cast("B"))
                    return closed
                return ()
            if topology_empty is False:
                for _canonical, source_frame in self._seam_frames:
                    self._window_hash.update(memoryview(source_frame).cast("B"))
                self._seam_empty_count = 0
                self._seam_frames = []
        if self._window_started_seconds is None:
            self._window_started_seconds = float(elapsed_seconds)
        local_index = self._canonical_index
        self._canonical_index += 1
        if self._auto_close:
            self._native_frames[local_index] = (native_rgb.copy(), native_evidence)
        observed = tuple(tracker.observe(
            rgb, frame_index=local_index,
            elapsed_seconds=float(elapsed_seconds) - self._window_started_seconds,
        ))
        self._window_hash.update(memoryview(source_pixels).cast("B"))
        if state is None:
            empty_candidate = True
            if self._auto_close and self._stream_state is None:
                predictor = getattr(tracker, "predictor", None)
                if not callable(predictor):
                    raise ProductionFactoryError("frozen empty-vial evidence is unavailable")
                prediction = predictor(rgb, self._reference)
                empty_candidate = not np.any(np.asarray(getattr(prediction, "mask", None)))
            self._inactive_count = self._inactive_count + 1 if empty_candidate else 0
            self._candidate_state = None
            self._candidate_count = 0
        else:
            self._inactive_count = 0
            if state == self._candidate_state:
                self._candidate_count += 1
            else:
                self._candidate_state, self._candidate_count = state, 1
            if (
                self._candidate_count >= math.ceil(
                    self._timing.stream_debounce_seconds
                    * self._timing.source_fps
                )
                and (not self._states or self._states[-1] != state)
            ):
                self._states.append(state)
        complete = self._states[-3:] == expected if len(self._states) >= 3 else False
        if (
            self._auto_close and complete
            and self._inactive_count >= math.ceil(
                self._timing.stable_tail_seconds * self._timing.source_fps
            )
        ):
            binder = getattr(tracker, "bind_source_artifact_sha256", None)
            if not callable(binder):
                raise ProductionFactoryError("dynamic tracker cannot bind source provenance")
            binder(self._window_hash.hexdigest())
            closed = self.close()
            self._window_hash = hashlib.sha256()
            return observed + closed
        return observed

    @staticmethod
    def _topology_cleaned_empty(tracker: object, rgb: np.ndarray) -> bool | None:
        predictor = getattr(tracker, "predictor", None)
        topology = getattr(tracker, "topology", None)
        parameters = getattr(topology, "parameters", None)
        reference = getattr(tracker, "reference", None)
        if not callable(predictor) or not callable(parameters) or reference is None:
            return None
        from fluidmix.wp7_fill_step_production import postprocess_cnn_mask_topology
        prediction = predictor(rgb, reference)
        cleaned = postprocess_cnn_mask_topology(
            getattr(prediction, "mask", None), **parameters(),
        )
        return not np.any(cleaned.mask)

    def end_of_stream(self):
        """Close only vial 3 at EOF after its complete debounced recipe sequence."""
        if self._active != 2 or self._states[-3:] != ["t200", "t400", "t100"]:
            return ()
        tracker = self._trackers[self._active]
        binder = getattr(tracker, "bind_source_artifact_sha256", None)
        if not callable(binder):
            raise ProductionFactoryError("dynamic tracker cannot bind source provenance")
        for _canonical, source_frame in self._seam_frames:
            self._window_hash.update(memoryview(source_frame).cast("B"))
        self._seam_frames = []
        binder(self._window_hash.hexdigest())
        return self.close()

    def _detect_state(self, rgb: np.ndarray) -> str | None:
        assert self._reference is not None
        if self._stream_state is not None:
            return self._stream_state(rgb, self._reference)
        from fluidmix.wp7_fill_step_production import measure_stream_lab
        tracker = self._trackers[self._active]
        freeze = getattr(tracker, "freeze", None)
        active, _count, centre, _spread = measure_stream_lab(rgb, self._reference, freeze)
        if not active or centre is None:
            return None
        return min(
            freeze.tank_lab_centres,
            key=lambda tank: math.dist(centre, freeze.tank_lab_centres[tank]),
        )

    def close(self):
        if self._active >= 3 or self._trackers[self._active] is None:
            raise ProductionFactoryError("vial boundary arrived before its accepted fill")
        rows = list(self._trackers[self._active].close())
        if self._auto_close:
            from PIL import Image
            for row in rows:
                if "frame_index" not in row or "prediction_mask" not in row:
                    continue
                native_rgb, native_evidence = self._native_frames[row["frame_index"]]
                canonical_mask = np.asarray(row["prediction_mask"])
                native_mask = np.asarray(Image.fromarray(
                    canonical_mask.astype(np.uint8) * 255, mode="L",
                ).resize(
                    tuple(native_evidence["native_resolution"]),
                    resample=Image.Resampling.NEAREST,
                )) > 0
                row["observation_rgb"] = native_rgb
                row["prediction_mask"] = native_mask
                row["native_view_evidence"] = deepcopy(native_evidence)
        final = rows[-1]
        template = final.get("vlm_vial")
        receipt = final.get("tracker_receipt")
        if isinstance(template, Mapping) and isinstance(receipt, Mapping):
            dynamic = deepcopy(dict(template))
            if not dynamic.get("stages"):
                dynamic["stages"] = [{
                    "stage_id": f"dispense-{stage['fill_step']}-{stage['tank_id']}",
                    "frames": [],
                } for stage in receipt["stages"]]
            context = final.get("vlm_cohort_context")
            if isinstance(context, dict) and not context.get("stages"):
                context["stages"] = [{
                    "stage_id": f"dispense-{stage['fill_step']}-{stage['tank_id']}",
                    "stage_ordinal": stage["fill_step"],
                    "dispense_order": stage["fill_step"],
                    "tank_id": stage["tank_id"],
                    "programmed_ml": stage["expected_dispense_ml"],
                    "stage_timing_seconds": [
                        min(stage["images"].values(), key=lambda value: int(value.rsplit("F", 1)[1])),
                        max(stage["images"].values(), key=lambda value: int(value.rsplit("F", 1)[1])),
                    ],
                    "stream": {
                        "evidence_id": f"STREAM-{context['cohort_id']}-S{stage['fill_step']}",
                        "median_lab": deepcopy(stage["stream_lab"]),
                        "spread_lab": deepcopy(stage["stream_spread_lab"]),
                    },
                } for stage in receipt["stages"]]
                for stage in context["stages"]:
                    stage["stage_timing_seconds"] = [
                        int(value.rsplit("F", 1)[1]) / self._timing.source_fps
                        for value in stage["stage_timing_seconds"]
                    ]
            for vial_stage, tracker_stage in zip(
                dynamic["stages"], receipt["stages"], strict=True,
            ):
                vial_stage["frames"] = [{
                    "image_id": tracker_stage["images"][phase], "phase": phase,
                    "timestamp_seconds": int(
                        str(tracker_stage["images"][phase]).rsplit("F", 1)[1]
                    ) / self._timing.source_fps,
                } for phase in ("early", "mid", "end")]
            rows[-1] = {**final, "vlm_vial": dynamic}
        self._active += 1
        self._canonical_index = 0
        self._window_started_seconds = None
        self._states = []
        self._candidate_state = None
        self._candidate_count = 0
        self._inactive_count = 0
        self._seam_empty_count = 0
        self._seam_frames = []
        self._reference = None
        self._native_frames = {}
        return tuple({
            **row,
            "stage_id": f"dispense-{row['fill_step']}-{row['tank_id']}",
        } for row in rows)


class ProductionAdapterFactory:
    accepted_pipeline_commit = PIPELINE_COMMIT
    development_cohort = "varied-v4"

    def __init__(self, *, vlm_runner: Callable[[object], object] | None = None):
        validate_artifacts()
        self._snapshots = _snapshot_paths()
        self._vlm_runner = vlm_runner

    def __call__(self, run_context: Mapping[str, object], device: str):
        raise ProductionFactoryError("production factory requires a session source binding")

    def for_session(self, request: Mapping[str, object], device: str):
        measurement_mode = measurement_mode_for_source(request.get("source"))
        validate_session_profile(request)

        from fluidmix.cnn_predictor import predict_pair
        from fluidmix.wp7_rgb_runtime import load_frozen_cnn
        from fluidmix.wp7_fill_step_production import (
            IncrementalFillStepTracker, load_pixel_to_metre_calibration,
            load_topology_freeze, load_tracker_freeze,
        )
        from fluidmix.wp7_fill_step_tracker import RecipePour
        from fluidmix.wp7_vlm_decision import build_vial1_bundle, run_sequential_vlm_runtime

        calibration = load_pixel_to_metre_calibration(
            CALIBRATION, expected_sha256=EXPECTED_HASHES[CALIBRATION],
        )
        topology = load_topology_freeze(
            TOPOLOGY_FREEZE, expected_sha256=EXPECTED_HASHES[TOPOLOGY_FREEZE],
        )
        recipe_hash = "f12c1fe65a994a89e9c5530c32a2c1f10954fc3ecb0d803511ca82296f99c626"
        freeze = load_tracker_freeze(
            TRACKER_MATERIALIZATION,
            expected_materialization_sha256=EXPECTED_HASHES[TRACKER_MATERIALIZATION],
            train_freeze_path=TRACKER_TRAIN_FREEZE,
            expected_train_freeze_sha256=EXPECTED_HASHES[TRACKER_TRAIN_FREEZE],
            topology=topology, calibration=calibration,
            target_recipe_artifact_sha256=recipe_hash,
        )
        loaded_cnn = load_frozen_cnn(
            checkpoint_path=CHECKPOINT, replay_path=REPLAY, review_path=REVIEW,
            expected_checkpoint_sha256=EXPECTED_HASHES[CHECKPOINT],
            expected_replay_sha256=EXPECTED_HASHES[REPLAY],
            expected_review_sha256=EXPECTED_HASHES[REVIEW],
        )
        cnn_model = prepare_cnn_model(loaded_cnn.model, device)
        run_context = request["run_context"]
        context, vial_templates = build_session_cohort(run_context, frame_rate_hz=10.0)

        def tracker_builder(index: int, reference: np.ndarray):
            return IncrementalFillStepTracker(
                freeze=freeze, topology=topology,
                recipe_pours=tuple(RecipePour(tank, 10.0) for tank in ("t200", "t400", "t100")),
                previous_final_tank_id="t100", same_run_empty_reference_rgb=reference,
                predictor=lambda rgb, ref: predict_pair(
                    cnn_model, rgb, ref, threshold=0.5, device=device,
                ),
                cohort_id=str(context["cohort_id"]),
                case_id=str(vial_templates[index]["vial_id"]),
                vial_id=str(vial_templates[index]["vial_id"]), vial_position=index + 1,
                source_artifact_sha256=None,
                vlm_cohort_context=context, vlm_vial=vial_templates[index],
                allow_unavailable_selected_phase=True, image_index_width=4,
                defer_source_artifact_binding=True,
                frame_rate_hz=float(context["camera_metadata"]["frame_rate_hz"]),
            )

        def deterministic(vials: Mapping[str, object]):
            from fluidmix.wp7_vlm_decision import (
                build_deterministic_decision, canonical_sha256,
            )
            measurements = deterministic_stage_measurements(vials)
            return build_deterministic_decision(
                cohort_id=str(vials["cohort_id"]),
                target_vial_id=str(vials["vials"][0]["vial_id"]),
                stage_measurements=measurements,
                source_artifact_sha256=canonical_sha256(measurements),
            )

        def runner(built: object):
            if self._vlm_runner is not None:
                return self._vlm_runner(built)
            from fluidmix.wp7_stock_runtime import invoke_stock_runtime, load_stock_runtime
            aggregate = run_sequential_vlm_runtime(
                built, self._snapshots, runtime_loader=load_stock_runtime,
                runtime_invoker=invoke_stock_runtime,
            )
            return model_attempt_rows(aggregate)

        def bind_profile(timing: FrozenTiming, profile: Mapping[str, object]) -> None:
            width, height = profile["resolution"]
            context["camera_metadata"].update({
                "frame_rate_hz": timing.source_fps,
                "resolution": [width, height],
                "vial_roi": [
                    math.floor(283 * width / 640), math.floor(194 * height / 480),
                    math.ceil(338 * width / 640), math.ceil(417 * height / 480),
                ],
            })

        def stream_state(rgb: np.ndarray, reference: np.ndarray) -> str | None:
            from fluidmix.wp7_fill_step_production import measure_stream_lab
            active, _count, centre, _spread = measure_stream_lab(rgb, reference, freeze)
            if not active or centre is None:
                return None
            return min(
                freeze.tank_lab_centres,
                key=lambda tank: math.dist(centre, freeze.tank_lab_centres[tank]),
            )

        adapter = AcceptedPipelineAdapter(
            tracker=SessionVialTracker(
                tracker_builder, auto_close=True,
                stream_state=stream_state, profile_bound=bind_profile,
            ), cnn_model=cnn_model,
            deterministic_decider=deterministic,
            vlm_bundle_builder=lambda value, locators, *, tracker_receipts: build_vial1_bundle(
                value, locators, tracker_receipts=tracker_receipts, production=True,
            ),
            vlm_runner=runner, device=device, model_id="accepted-varied-v4",
            measurement_mode=measurement_mode,
        )
        return adapter


class DefaultProductionFactory:
    """Lazy repository-native factory exported for ``runner serve``."""

    accepted_pipeline_commit = PIPELINE_COMMIT
    development_cohort = "varied-v4"

    def __init__(self):
        self._factory: ProductionAdapterFactory | None = None

    def __call__(self, run_context: Mapping[str, object], device: str):
        raise ProductionFactoryError("production factory requires a session source binding")

    def for_session(self, request: Mapping[str, object], device: str):
        validate_session_profile(request)
        if self._factory is None:
            self._factory = ProductionAdapterFactory()
        return self._factory.for_session(request, device)


factory = DefaultProductionFactory()

