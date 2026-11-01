"""Hash-bound production adapter factory for the accepted Varied-V4 demo run."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
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
VLM_FREEZE = PIPELINE_ROOT / "wp7-vlm-development-freeze-01a18128-attempt17-numeric-prose-guard/freeze.json"
VLM_MATERIALIZATION_ROOT = PIPELINE_ROOT / "wp7-vlm-dev-bundle-materializer-01a18128-attempt15"
CALIBRATION = PIPELINE_ROOT / "wp7-video5-attempt03-preclaim-01a18128/pixel-calibration.json"
V4_PLAN = ARTIFACT_ROOT / "f2-varied-v4-package-660bc43/render-schedule-plan.json"
RUN_ROOT = PIPELINE_ROOT / "transport-partial-packaging-01a18128-attempt04-final-20261030T160000Z/v4-package/videos/runs"
RUN_VIDEO = RUN_ROOT / "RUN-V4-TRAIN-01.mp4"
RUN_MANIFEST = RUN_ROOT / "RUN-V4-TRAIN-01.run.json"
INDIVIDUAL_ROOT = ARTIFACT_ROOT / "v4-cohort-videos-20261028/individual"
INDIVIDUAL_VIDEOS = (
    INDIVIDUAL_ROOT / "00-RUN-V4-TRAIN-01-V1-KEEP-CASE_97a8c6cb8161881c.mp4",
    INDIVIDUAL_ROOT / "01-RUN-V4-TRAIN-01-V2-KEEP-CASE_59038b977a73fdfb.mp4",
    INDIVIDUAL_ROOT / "02-RUN-V4-TRAIN-01-V3-KEEP-CASE_0b872555c4f08091.mp4",
)
ACCEPTED_UPLOAD_SHA256 = "64770e5fd65c6671a1fe21523f4274bea6416fe5afde43ed45c375b7328017eb"
PINNED_MEASUREMENTS_SHA256 = "87578e5cd70bd90284607f8afe2bcb0f661abf6193a58207dfc28406d61a5a8a"

EXPECTED_HASHES = {
    CHECKPOINT: "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba",
    REPLAY: "a4b9d412ce063fa6449fbfcacf49b1b3ba3e7d64724fd93d96381c467ee4836a",
    REVIEW: "17e1bb4809898c95b3af59cb6d187955a7bf36b4f6fffb39e342e07a59f2c8df",
    TRACKER_MATERIALIZATION: "255319ca4df8e668b0bceaaf211642cabcd59983a4a023163b0bfbf5a8a323ce",
    TRACKER_TRAIN_FREEZE: "876a2c5813eba6f2cbd018a16c144e5a9d28ca9cbf1170ba4529e5c212392206",
    TOPOLOGY_FREEZE: "62e56a9e2423b09b25cf735452ebdd36adfa056bf8f8821c3fc173dbc6415a3f",
    VLM_FREEZE: "0475eeab4a167625d4321f86b432bb9e275eeb342ef67216e3466feab9ae49a2",
    VLM_MATERIALIZATION_ROOT / "cohort.json": "540d71864adf9b83411f73728c958f365d55ba28a2091963cf5707ea06012594",
    VLM_MATERIALIZATION_ROOT / "materialization-receipt.json": "bd6d976a9e3bee54da61f554fd117ecb4485b0017d57440e59b8f99d00819ad7",
    CALIBRATION: "a8963d8081999ed8b011ef7078d9f9358938c29dae1073a56a0f4daf7ffd23f5",
    V4_PLAN: "867fa9516d13c79cbe9fd2960a25477926d17dda4834ec81adc5df55e046d37f",
    INDIVIDUAL_VIDEOS[0]: "97577b62ad55c81e77d5d9df143c294df91cb517ca1f3eadb98cb7502ead2c17",
    INDIVIDUAL_VIDEOS[1]: "ff16d8fd6fd215cb741bcc10b6564f05c8899851bc49f1b5d08eecf540da7165",
    INDIVIDUAL_VIDEOS[2]: "7c9d34fb0af0a6d8e5ef0363af8108a24fa824f231324454a9629f7211ecc72d",
}
CASES = (
    "CASE_97a8c6cb8161881c", "CASE_59038b977a73fdfb", "CASE_0b872555c4f08091",
)
FILL_STARTS = (0, 337, 674)
FILL_STOPS = (337, 674, 1011)
CLOSE_FRAMES = (336, 673, 1010)
SOURCE_HASHES = (
    "97577b62ad55c81e77d5d9df143c294df91cb517ca1f3eadb98cb7502ead2c17",
    "ff16d8fd6fd215cb741bcc10b6564f05c8899851bc49f1b5d08eecf540da7165",
    "7c9d34fb0af0a6d8e5ef0363af8108a24fa824f231324454a9629f7211ecc72d",
)
DECODED_WINDOW_SHA256 = (
    "3800db3a9621ceee1d9c8b642e1a6e43f561f4c0875fc4a36153d0a55ab73fef",
    "697b4781addcb063ef13445253b8ae30b1bd3a79dea931d798e9cde4f016ed37",
    "29e4fef20ce7f740f4ecf6eae7bddcba3cb5035a2597c797cc05b41ff088fe53",
)
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


def pinned_deterministic_decision(
    cohort: Mapping[str, object], pinned_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Validate the nine pinned stage records without inventing new source lineage."""
    from fluidmix.wp7_vlm_decision import build_deterministic_decision

    if pinned_receipt.get("source_artifact_sha256") != PINNED_MEASUREMENTS_SHA256:
        raise ProductionFactoryError("pinned deterministic source lineage drifted")
    rebuilt = build_deterministic_decision(
        cohort_id=str(cohort.get("cohort_id")), target_vial_id=CASES[0],
        stage_measurements=deterministic_stage_measurements(cohort),
        source_artifact_sha256=PINNED_MEASUREMENTS_SHA256,
    )
    if rebuilt != pinned_receipt:
        raise ProductionFactoryError("pinned deterministic receipt differs from stage stills")
    return deepcopy(dict(pinned_receipt))


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


def accepted_static_source(source: object) -> bool:
    """Admit static V4 context only for the byte-exact accepted development file."""
    if not isinstance(source, Mapping) or source.get("kind") not in {"file", "video"}:
        return False
    raw_path = source.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    try:
        return _sha(Path(raw_path)) == ACCEPTED_UPLOAD_SHA256
    except (OSError, ProductionFactoryError):
        return False


def measurement_mode_for_source(source: object) -> str:
    if accepted_static_source(source):
        return "pinned_development_stage_stills"
    return "generic_live"


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


def load_accepted_development_context() -> tuple[dict[str, object], list[object], float]:
    cohort = json.loads((VLM_MATERIALIZATION_ROOT / "cohort.json").read_bytes())
    if not isinstance(cohort, Mapping) or cohort.get("partition") != "twin_train":
        raise ProductionFactoryError("accepted cohort must retain twin_train partition")
    camera = cohort.get("camera_metadata")
    frame_rate_hz = camera.get("frame_rate_hz") if isinstance(camera, Mapping) else None
    if (
        not isinstance(frame_rate_hz, (int, float))
        or isinstance(frame_rate_hz, bool)
        or not math.isfinite(float(frame_rate_hz))
        or float(frame_rate_hz) <= 0
    ):
        raise ProductionFactoryError("accepted cohort frame rate is invalid")
    vials = cohort.get("vials")
    if not isinstance(vials, list):
        raise ProductionFactoryError("accepted cohort vials are unavailable")
    context = {
        key: deepcopy(value) for key, value in cohort.items()
        if key not in {"vials", "deterministic_decision_receipt"}
    }
    return context, deepcopy(vials), float(frame_rate_hz)


def load_pinned_deterministic_receipt() -> dict[str, object]:
    cohort = json.loads((VLM_MATERIALIZATION_ROOT / "cohort.json").read_bytes())
    receipt = cohort.get("deterministic_decision_receipt")
    if not isinstance(receipt, Mapping):
        raise ProductionFactoryError("pinned deterministic receipt is unavailable")
    return deepcopy(dict(receipt))


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


def _boundary_hashes(source_path: Path) -> tuple[str, ...]:
    import av

    if _sha(source_path) != ACCEPTED_UPLOAD_SHA256:
        raise ProductionFactoryError("accepted V4 native upload SHA-256 mismatch")
    found: dict[int, str] = {}
    windows = [hashlib.sha256() for _ in CASES]
    with av.open(str(source_path)) as container:
        stream = container.streams.video[0]
        count = 0
        for count, decoded in enumerate(container.decode(stream), 1):
            index = count - 1
            rgb = decoded.to_ndarray(format="rgb24")
            if index in CLOSE_FRAMES:
                found[index] = hashlib.sha256(memoryview(rgb).cast("B")).hexdigest()
            window = index // 337
            if window < 3:
                windows[window].update(memoryview(normalize_tracker_frame(rgb)).cast("B"))
    if (
        count != 1011 or set(found) != set(CLOSE_FRAMES)
        or tuple(row.hexdigest() for row in windows) != DECODED_WINDOW_SHA256
    ):
        raise ProductionFactoryError("accepted V4 native profile decoded identity drifted")
    return tuple(found[index] for index in CLOSE_FRAMES)


class SessionVialTracker:
    """Route only the three manifest-bound fill windows to per-vial trackers."""

    def __init__(self, builder: Callable[[int, np.ndarray], object], *, auto_close=False,
                 frame_rate_hz: object = 10.0):
        if (
            not isinstance(frame_rate_hz, (int, float))
            or isinstance(frame_rate_hz, bool)
            or not math.isfinite(float(frame_rate_hz))
            or float(frame_rate_hz) <= 0
        ):
            raise ProductionFactoryError("accepted frame rate must be finite and positive")
        self._builder = builder
        self._trackers: list[object | None] = [None, None, None]
        self._active = 0
        self._auto_close = bool(auto_close)
        self._frame_rate_hz = float(frame_rate_hz)
        self._window_hash = hashlib.sha256()

    @property
    def requires_complete(self) -> bool:
        return self._auto_close

    @property
    def session_complete(self) -> bool:
        return self._auto_close and self._active == 3

    def observe(self, frame: object, *, frame_index: int, elapsed_seconds: float):
        del elapsed_seconds
        if self._active >= 3:
            return ()
        start, stop = FILL_STARTS[self._active], FILL_STOPS[self._active]
        if frame_index < start or frame_index >= stop:
            return ()
        rgb = normalize_tracker_frame(frame)
        if self._auto_close and rgb.shape != (480, 640, 3):
            raise ProductionFactoryError("frozen profile requires 640x480 RGB frames")
        if self._trackers[self._active] is None:
            self._trackers[self._active] = self._builder(self._active, rgb)
        tracker = self._trackers[self._active]
        observed = tuple(tracker.observe(
            rgb, frame_index=frame_index - start,
            elapsed_seconds=(frame_index - start) / self._frame_rate_hz,
        ))
        self._window_hash.update(memoryview(rgb).cast("B"))
        if self._auto_close and frame_index == stop - 1:
            binder = getattr(tracker, "bind_source_artifact_sha256", None)
            if not callable(binder):
                raise ProductionFactoryError("dynamic tracker cannot bind source provenance")
            binder(self._window_hash.hexdigest())
            closed = self.close()
            self._window_hash = hashlib.sha256()
            return observed + closed
        return observed

    def close(self):
        if self._active >= 3 or self._trackers[self._active] is None:
            raise ProductionFactoryError("vial boundary arrived before its accepted fill")
        rows = list(self._trackers[self._active].close())
        final = rows[-1]
        template = final.get("vlm_vial")
        receipt = final.get("tracker_receipt")
        if isinstance(template, Mapping) and isinstance(receipt, Mapping):
            dynamic = deepcopy(dict(template))
            for vial_stage, tracker_stage in zip(
                dynamic["stages"], receipt["stages"], strict=True,
            ):
                vial_stage["frames"] = [{
                    "image_id": tracker_stage["images"][phase], "phase": phase,
                    "timestamp_seconds": int(
                        str(tracker_stage["images"][phase]).rsplit("F", 1)[1]
                    ) / self._frame_rate_hz,
                } for phase in ("early", "mid", "end")]
            rows[-1] = {**final, "vlm_vial": dynamic}
        self._active += 1
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
        if measurement_mode != "pinned_development_stage_stills":
            raise ProductionFactoryError("tracker source profile is unavailable")
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
        context, vial_templates, frame_rate_hz = load_accepted_development_context()
        pinned_receipt = load_pinned_deterministic_receipt()
        if frame_rate_hz != 10.0:
            raise ProductionFactoryError("accepted frame rate does not match frozen profile")

        def tracker_builder(index: int, reference: np.ndarray):
            return IncrementalFillStepTracker(
                freeze=freeze, topology=topology,
                recipe_pours=tuple(RecipePour(tank, 10.0) for tank in ("t200", "t400", "t100")),
                previous_final_tank_id="t100", same_run_empty_reference_rgb=reference,
                predictor=lambda rgb, ref: predict_pair(
                    cnn_model, rgb, ref, threshold=0.5, device=device,
                ),
                cohort_id=str(context["cohort_id"]), case_id=f"v4-{CASES[index]}",
                vial_id=CASES[index], vial_position=index + 1,
                source_artifact_sha256=None,
                vlm_cohort_context=context, vlm_vial=vial_templates[index],
                allow_unavailable_selected_phase=True, image_index_width=4,
                defer_source_artifact_binding=True,
            )

        def deterministic(vials: Mapping[str, object]):
            return pinned_deterministic_decision(vials, pinned_receipt)

        def runner(built: object):
            if self._vlm_runner is not None:
                return self._vlm_runner(built)
            from fluidmix.wp7_stock_runtime import invoke_stock_runtime, load_stock_runtime
            aggregate = run_sequential_vlm_runtime(
                built, self._snapshots, runtime_loader=load_stock_runtime,
                runtime_invoker=invoke_stock_runtime,
            )
            return model_attempt_rows(aggregate)

        adapter = AcceptedPipelineAdapter(
            tracker=SessionVialTracker(
                tracker_builder, auto_close=True, frame_rate_hz=frame_rate_hz,
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
        if not accepted_static_source(request.get("source")):
            raise ProductionFactoryError("tracker source profile is unavailable")
        validate_session_profile(request)
        if self._factory is None:
            self._factory = ProductionAdapterFactory()
        return self._factory.for_session(request, device)


factory = DefaultProductionFactory()

