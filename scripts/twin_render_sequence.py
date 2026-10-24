"""Render fail-closed FillCam RGB, semantic, instance, and liquid-mask evidence.

This keeps the verified Isaac Sim 6.0.1 FillCam recipe: PathTracing, no clipping-range
re-sync, warm-up updates, explicit time-code stepping, and bounded per-frame retries. Isaac
imports and ``SimulationApp`` startup occur only after CLI and filesystem preflight.

CPU tests cover argument handling, semantic planning, and captured array/mapping validity.
They cannot prove Isaac session-layer behavior, rendered geometry, or sim-to-real validity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fluidmix.render_capture import (  # noqa: E402
    LIQUID_OVERRIDE,
    CaptureValidation,
    plan_semantic_assignments,
    validate_capture,
)


SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Preflight:
    twin: Path
    stage: Path
    camera_file: Path
    script: Path
    output: Path
    camera_record: dict[str, Any]
    resolution: tuple[int, int]
    frame_range: tuple[int, int, int]
    time_codes: tuple[int, ...]


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_frame_spec(value: str) -> tuple[int, int, int]:
    """Parse an exclusive-stop ``start:stop:step`` time-code range."""

    if not isinstance(value, str):
        raise ValueError("frame specification must be a string")
    pieces = value.split(":")
    if len(pieces) != 3:
        raise ValueError("frames must use start:stop:step syntax")
    try:
        start, stop, step = (int(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError("frame start, stop, and step must be integers") from error
    if start < 0:
        raise ValueError("frame start must be nonnegative")
    if stop <= start:
        raise ValueError("frame stop must be greater than start")
    if step <= 0:
        raise ValueError("frame step must be positive")
    return start, stop, step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", required=True, help="csi_digital_twin_v1 folder")
    parser.add_argument(
        "--stage",
        default="stations/fill_station/fill_station.usda",
        help="stage path; relative paths resolve under --twin",
    )
    parser.add_argument(
        "--camera",
        choices=("FillCam",),
        default="FillCam",
        help="production capture is restricted to the reviewed FillCam view",
    )
    parser.add_argument(
        "--frames", default="1:687:15", help="exclusive start:stop:step time codes"
    )
    parser.add_argument("--subframes", type=positive_int, default=32)
    parser.add_argument(
        "--renderer",
        default="PathTracing",
        choices=("PathTracing", "RaytracedLighting"),
    )
    parser.add_argument(
        "--warmup",
        type=positive_int,
        default=60,
        help="kit.update() calls before the first capture",
    )
    parser.add_argument(
        "--max-tries",
        type=positive_int,
        default=20,
        help="shared-capture retries per frame",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=20261024,
        help="Replicator global seed; full renderer determinism is not claimed",
    )
    parser.add_argument(
        "--require-liquid-pixels",
        action="store_true",
        help="require both liquid and vial semantic pixels for every kept frame",
    )
    parser.add_argument("--out", required=True)
    return parser


def validate_output_destination(path: os.PathLike[str] | str) -> None:
    """Reject every occupied exact output path, including dangling symlinks."""

    candidate = os.fspath(path)
    if os.path.lexists(candidate):
        raise FileExistsError(f"output path already exists: {candidate}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_resolving_leaf(path: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def preflight(args: argparse.Namespace) -> Preflight:
    frame_range = parse_frame_spec(args.frames)
    time_codes = tuple(range(*frame_range))
    if not time_codes:
        raise ValueError("frame range selects no time codes")

    output = _absolute_without_resolving_leaf(args.out)
    validate_output_destination(output)

    twin = Path(args.twin).expanduser().resolve(strict=True)
    if not twin.is_dir():
        raise NotADirectoryError(f"twin root is not a directory: {twin}")
    requested_stage = Path(args.stage).expanduser()
    stage = (requested_stage if requested_stage.is_absolute() else twin / requested_stage).resolve(
        strict=True
    )
    if not stage.is_file():
        raise FileNotFoundError(f"stage is not a file: {stage}")
    camera_file = (twin / "cameras" / f"{args.camera}.json").resolve(strict=True)
    if not camera_file.is_file():
        raise FileNotFoundError(f"camera record is not a file: {camera_file}")
    with camera_file.open("r", encoding="utf-8") as stream:
        camera_record = json.load(stream)
    render_px = camera_record.get("render_px")
    if (
        not isinstance(render_px, list)
        or len(render_px) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in render_px
        )
    ):
        raise ValueError(f"camera render_px must be two positive integers: {render_px!r}")
    for key in ("prim_in_csi_line", "prim_in_station_file"):
        value = camera_record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"camera record has invalid {key}: {value!r}")

    return Preflight(
        twin=twin,
        stage=stage,
        camera_file=camera_file,
        script=Path(__file__).resolve(strict=True),
        output=output,
        camera_record=camera_record,
        resolution=(render_px[0], render_px[1]),
        frame_range=frame_range,
        time_codes=time_codes,
    )


def _create_partial_output(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_output_destination(output)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=os.fspath(output.parent))
    )
    for name in ("rgb", "semantic", "instance", "liquid"):
        (partial / name).mkdir()
    return partial


def _annotator_data_and_mapping(raw: object, *, name: str) -> tuple[object, object]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"{name} annotator did not return a data/info mapping")
    if "data" not in raw:
        raise RuntimeError(f"{name} annotator result has no data")
    info = raw.get("info")
    if not isinstance(info, dict) or "idToLabels" not in info:
        raise RuntimeError(f"{name} annotator result has no info.idToLabels")
    return raw["data"], info["idToLabels"]


def _stage_layer_records(stage: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for ordinal, layer in enumerate(stage.GetLayerStack(includeSessionLayers=False), start=1):
        real_path = str(layer.realPath) if layer.realPath else None
        record: dict[str, object] = {
            "ordinal": ordinal,
            "identifier": str(layer.identifier),
            "resolved_path": None,
            "sha256": None,
            "bytes": None,
        }
        if real_path:
            resolved = Path(real_path).resolve(strict=True)
            if not resolved.is_file():
                raise RuntimeError(f"non-session USD layer is not a file: {resolved}")
            record.update(
                {
                    "resolved_path": str(resolved),
                    "sha256": sha256_file(resolved),
                    "bytes": resolved.stat().st_size,
                }
            )
        records.append(record)
    if not records:
        raise RuntimeError("opened stage reported no non-session layer stack")
    return records


def _collect_semantic_plan(stage: object, Usd: object, UsdSemantics: object):
    source_classes: dict[str, object] = {}
    prim_types: dict[str, str] = {}
    existing_effective: dict[str, tuple[str, ...]] = {}
    query = UsdSemantics.LabelsQuery("class", Usd.TimeCode.Default())
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        prim_types[path] = prim.GetTypeName()
        source = prim.GetAttribute("userProperties:synthetic_class")
        source_classes[path] = source.Get() if source and source.HasValue() else None
        existing_effective[path] = tuple(
            str(label) for label in query.ComputeUniqueInheritedLabels(prim)
        )
    return plan_semantic_assignments(source_classes, prim_types, existing_effective)


def _author_semantic_plan(stage: object, plan: object, Usd: object, UsdSemantics: object) -> None:
    session_layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, session_layer):
        for assignment in plan.assignments:
            prim = stage.GetPrimAtPath(assignment.path)
            if not prim.IsValid():
                raise RuntimeError(f"semantic assignment prim disappeared: {assignment.path}")
            api = UsdSemantics.LabelsAPI.Apply(prim, "class")
            if not api:
                raise RuntimeError(f"could not apply class LabelsAPI at {assignment.path}")
            attribute = api.CreateLabelsAttr()
            if not attribute.Set([assignment.class_name]):
                raise RuntimeError(f"could not author class label at {assignment.path}")

        query = UsdSemantics.LabelsQuery("class", Usd.TimeCode.Default())
        for path in (LIQUID_OVERRIDE.root_path, LIQUID_OVERRIDE.mesh_path):
            effective = {
                str(label)
                for label in query.ComputeUniqueInheritedLabels(stage.GetPrimAtPath(path))
            }
            if effective != {LIQUID_OVERRIDE.class_name}:
                raise RuntimeError(
                    f"liquid override verification failed at {path}: {sorted(effective)!r}"
                )
    if not session_layer.dirty:
        raise RuntimeError("semantic override did not dirty the session layer")


def _write_png_checked(cv2: object, path: Path, array: object) -> None:
    if not cv2.imwrite(str(path), array):
        raise OSError(f"image writer failed: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.shape != array.shape or decoded.dtype != array.dtype:
        raise OSError(f"image write verification failed: {path}")
    if not (decoded == array).all():
        raise OSError(f"image write was not lossless: {path}")


def _write_npy_checked(np: object, path: Path, array: object) -> None:
    np.save(path, array, allow_pickle=False)
    decoded = np.load(path, allow_pickle=False)
    if decoded.shape != array.shape or decoded.dtype != array.dtype:
        raise OSError(f"array write verification failed: {path}")
    if not np.array_equal(decoded, array):
        raise OSError(f"array write was not lossless: {path}")


def _stats_record(stats: object) -> dict[str, object]:
    if stats is None:
        raise RuntimeError("valid capture is missing array statistics")
    record = asdict(stats)
    record["shape"] = list(record["shape"])
    return record


def _frame_record(
    *,
    validation: CaptureValidation,
    ordinal: int,
    time_code: int,
    attempts: int,
    seconds: float,
    artifacts: dict[str, dict[str, str]],
) -> dict[str, object]:
    if validation.class_mapping is None:
        raise RuntimeError("valid capture is missing a semantic mapping")
    return {
        "capture_ordinal": ordinal,
        "time_code": time_code,
        "attempts": attempts,
        "capture_seconds": round(seconds, 6),
        "reason_codes": list(validation.reason_codes),
        "rgb": _stats_record(validation.rgb),
        "semantic": _stats_record(validation.semantic),
        "instance": _stats_record(validation.instance),
        "semantic_ids": list(validation.semantic_ids),
        "instance_ids": list(validation.instance_ids),
        "class_ids": {
            "liquid": list(validation.class_mapping.liquid_ids),
            "vial": list(validation.class_mapping.vial_ids),
        },
        "class_pixel_counts": {
            record.class_name: record.pixel_count
            for record in validation.class_pixel_counts
        },
        "semantic_mapping_records": [
            {"id": record.semantic_id, "labels": list(record.labels)}
            for record in validation.class_mapping.records
        ],
        "instance_mapping_records": [
            {"id": record.instance_id, "path": record.path}
            for record in validation.instance_mapping
        ],
        "instance_agreement": {
            "passed": validation.instance_agreement,
            "selected_liquid_instance_ids": list(
                validation.selected_liquid_instance_ids
            ),
            "semantic_liquid_pixels": validation.pixel_count_for("liquid"),
            "instance_liquid_pixels": validation.liquid_instance_pixel_count,
        },
        "artifacts": artifacts,
    }


def _capture_to_partial(
    args: argparse.Namespace, inputs: Preflight, partial: Path
) -> dict[str, object]:
    from isaacsim import SimulationApp

    width, height = inputs.resolution
    kit = SimulationApp(
        {
            "headless": True,
            "renderer": args.renderer,
            "width": width,
            "height": height,
        }
    )
    started = time.time()
    failed = True
    try:
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Usd, UsdSemantics

        context = omni.usd.get_context()
        if not context.open_stage(str(inputs.stage)):
            raise RuntimeError(f"could not open stage: {inputs.stage}")
        while context.get_stage_loading_status()[2] > 0:
            kit.update()
        stage = context.get_stage()
        layer_records = _stage_layer_records(stage)

        start_bound = float(stage.GetStartTimeCode())
        end_bound = float(stage.GetEndTimeCode())
        outside = [
            code for code in inputs.time_codes if code < start_bound or code > end_bound
        ]
        if outside:
            raise ValueError(
                f"requested time codes outside opened stage bounds "
                f"[{start_bound}, {end_bound}]: {outside}"
            )
        time_codes_per_second = float(stage.GetTimeCodesPerSecond())
        if time_codes_per_second <= 0:
            raise RuntimeError(
                f"stage has invalid timeCodesPerSecond: {time_codes_per_second}"
            )

        camera_candidates = (
            inputs.camera_record["prim_in_csi_line"],
            inputs.camera_record["prim_in_station_file"],
        )
        camera_prim = next(
            (
                candidate
                for candidate in camera_candidates
                if stage.GetPrimAtPath(candidate).IsValid()
            ),
            None,
        )
        if camera_prim is None:
            raise RuntimeError(f"no reviewed FillCam prim found: {camera_candidates!r}")

        semantic_plan = _collect_semantic_plan(stage, Usd, UsdSemantics)
        _author_semantic_plan(stage, semantic_plan, Usd, UsdSemantics)

        rep.set_global_seed(args.seed)
        render_product = rep.create.render_product(camera_prim, (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        semantic_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        )
        instance_annotator = rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"colorize": False}
        )
        for annotator in (rgb_annotator, semantic_annotator, instance_annotator):
            annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)

        for _ in range(args.warmup):
            kit.update()

        timeline = omni.timeline.get_timeline_interface()
        frame_records: list[dict[str, object]] = []
        for ordinal, time_code in enumerate(inputs.time_codes, start=1):
            timeline.set_current_time(time_code / time_codes_per_second)
            kit.update()
            frame_started = time.time()
            last_validation: CaptureValidation | None = None
            for attempt in range(1, args.max_tries + 1):
                rep.orchestrator.step(
                    rt_subframes=args.subframes,
                    delta_time=0.0,
                    wait_for_render=True,
                )
                raw_rgb = rgb_annotator.get_data()
                if (
                    isinstance(raw_rgb, np.ndarray)
                    and raw_rgb.ndim == 3
                    and raw_rgb.shape[2] == 4
                ):
                    capture_rgb = raw_rgb[..., :3]
                else:
                    capture_rgb = raw_rgb
                semantic, semantic_mapping = _annotator_data_and_mapping(
                    semantic_annotator.get_data(), name="semantic"
                )
                instance, instance_mapping = _annotator_data_and_mapping(
                    instance_annotator.get_data(), name="instance"
                )
                last_validation = validate_capture(
                    capture_rgb,
                    semantic,
                    instance,
                    semantic_mapping,
                    instance_mapping,
                    raster_shape=(height, width),
                    require_liquid_pixels=args.require_liquid_pixels,
                )
                if last_validation.valid:
                    break
            else:
                assert last_validation is not None
                raise RuntimeError(
                    f"time code {time_code} failed shared capture after "
                    f"{args.max_tries} attempts: {last_validation.reason_codes!r}"
                )

            assert last_validation is not None
            assert isinstance(capture_rgb, np.ndarray)
            assert isinstance(semantic, np.ndarray)
            assert isinstance(instance, np.ndarray)
            stem = f"{time_code:04d}"
            rgb_path = partial / "rgb" / f"{stem}.png"
            semantic_path = partial / "semantic" / f"{stem}.npy"
            instance_path = partial / "instance" / f"{stem}.npy"
            liquid_path = partial / "liquid" / f"{stem}.png"
            _write_png_checked(cv2, rgb_path, capture_rgb[..., ::-1].copy())
            _write_npy_checked(np, semantic_path, semantic)
            _write_npy_checked(np, instance_path, instance)
            liquid_ids = last_validation.class_mapping.liquid_ids
            liquid_mask = np.isin(semantic, liquid_ids).astype(np.uint8) * 255
            _write_png_checked(cv2, liquid_path, liquid_mask)

            paths = (rgb_path, semantic_path, instance_path, liquid_path)
            artifacts = {
                path.parent.name: {
                    "path": path.relative_to(partial).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in paths
            }
            frame_record = _frame_record(
                validation=last_validation,
                ordinal=ordinal,
                time_code=time_code,
                attempts=attempt,
                seconds=time.time() - frame_started,
                artifacts=artifacts,
            )
            frame_records.append(frame_record)
            print(json.dumps(frame_record, sort_keys=True), flush=True)

        summary: dict[str, object] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "mechanical_gates_passed": True,
            "visual_check_status": "pending",
            "inputs": {
                "twin_root": str(inputs.twin),
                "stage": {
                    "requested": args.stage,
                    "resolved_path": str(inputs.stage),
                    "sha256": sha256_file(inputs.stage),
                },
                "camera": {
                    "name": args.camera,
                    "resolved_path": str(inputs.camera_file),
                    "sha256": sha256_file(inputs.camera_file),
                    "prim_path": camera_prim,
                },
                "script": {
                    "resolved_path": str(inputs.script),
                    "sha256": sha256_file(inputs.script),
                },
                "non_session_usd_layer_stack": layer_records,
            },
            "render_settings": {
                "headless": True,
                "renderer": args.renderer,
                "resolution": [width, height],
                "frame_spec": args.frames,
                "time_codes": list(inputs.time_codes),
                "time_codes_per_second": time_codes_per_second,
                "stage_time_code_bounds": [start_bound, end_bound],
                "warmup_updates": args.warmup,
                "rt_subframes": args.subframes,
                "max_tries_per_frame": args.max_tries,
                "capture_on_play": False,
                "delta_time": 0.0,
                "wait_for_render": True,
                "annotators": {
                    "rgb": {},
                    "semantic_segmentation": {
                        "colorize": False,
                        "semanticTypes": ["class"],
                    },
                    "instance_segmentation": {"colorize": False},
                },
                "require_liquid_pixels": args.require_liquid_pixels,
                "replicator_global_seed": args.seed,
                "seed_limit": "The recorded Replicator seed does not prove full renderer determinism.",
            },
            "semantic_override": asdict(LIQUID_OVERRIDE),
            "semantic_plan": {
                "copied_assignments": [
                    asdict(assignment) for assignment in semantic_plan.copied_assignments
                ],
                "liquid_assignment": asdict(semantic_plan.liquid_assignment),
                "excluded_subtree": list(semantic_plan.excluded_subtree),
            },
            "frame_count": len(frame_records),
            "total_seconds": round(time.time() - started, 6),
            "frames": frame_records,
        }
        summary_path = partial / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise OSError(f"summary write verification failed: {summary_path}")
        failed = False
        return summary
    finally:
        if failed:
            kit.close(exit_code=1)
        else:
            kit.close()


def _validate_complete_partial(
    partial: Path, summary: dict[str, object], expected_frames: int
) -> None:
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise RuntimeError("summary schema version failed")
    if summary.get("mechanical_gates_passed") is not True:
        raise RuntimeError("summary mechanical gate failed")
    if summary.get("visual_check_status") != "pending":
        raise RuntimeError("summary visual status must remain pending")
    frames = summary.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_frames:
        raise RuntimeError("summary frame denominator failed")
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("artifacts"), dict):
            raise RuntimeError("summary artifact records are malformed")
        for artifact in frame["artifacts"].values():
            if not isinstance(artifact, dict):
                raise RuntimeError("summary artifact record is malformed")
            relative = artifact.get("path")
            expected_hash = artifact.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise RuntimeError("summary artifact path/hash is malformed")
            path = partial / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"artifact hash gate failed: {relative}")
    summary_path = partial / "summary.json"
    if not summary_path.is_file() or json.loads(
        summary_path.read_text(encoding="utf-8")
    ) != summary:
        raise RuntimeError("summary file gate failed")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = preflight(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        build_parser().error(str(error))
    partial = _create_partial_output(inputs.output)
    try:
        summary = _capture_to_partial(args, inputs, partial)
        _validate_complete_partial(partial, summary, len(inputs.time_codes))
        validate_output_destination(inputs.output)
        partial.rename(inputs.output)
    except BaseException:
        print(f"retained partial output for diagnosis: {partial}", file=sys.stderr)
        raise
    print(f"published complete render: {inputs.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
