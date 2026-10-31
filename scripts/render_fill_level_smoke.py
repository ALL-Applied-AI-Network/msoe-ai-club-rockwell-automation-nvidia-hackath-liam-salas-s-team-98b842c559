"""Render the unchanged public FillCam frame 686 with bounded readiness checks.

Run with Isaac Sim's Python under an outer process timeout. The source USD and
camera are read-only; Kit writes data, cache and logs below --portable-root.
This is a renderer-feasibility smoke test, not evidence of camera matching.
"""

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import time
import traceback


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", type=Path, required=True)
    parser.add_argument(
        "--stage",
        type=Path,
        default=Path("stations/fill_station/fill_station.usda"),
        help="absolute path or path relative to --twin",
    )
    parser.add_argument(
        "--camera-json",
        type=Path,
        default=Path("cameras/FillCam.json"),
        help="absolute path or path relative to --twin",
    )
    parser.add_argument("--frame", type=int, default=686)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument("--subframes", type=int, default=32)
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--stage-timeout-seconds", type=float, default=120)
    parser.add_argument("--warmup-updates", type=int, default=60)
    parser.add_argument("--retry-updates", type=int, default=30)
    args = parser.parse_args()

    twin = args.twin.expanduser().resolve()
    stage_path = args.stage.expanduser()
    if not stage_path.is_absolute():
        stage_path = twin / stage_path
    stage_path = stage_path.resolve()
    camera_path = args.camera_json.expanduser()
    if not camera_path.is_absolute():
        camera_path = twin / camera_path
    camera_path = camera_path.resolve()
    output_path = args.out.expanduser().resolve()
    summary_path = output_path.with_suffix(output_path.suffix + ".json")
    portable_root = args.portable_root.expanduser().resolve()

    if not twin.is_dir() or not stage_path.is_file() or not camera_path.is_file():
        parser.error("--twin, --stage and --camera-json must resolve to existing inputs")
    if output_path.exists() or summary_path.exists():
        parser.error("output PNG and summary must be fresh paths")
    if any(
        value < 1
        for value in (
            args.frame,
            args.subframes,
            args.max_tries,
            args.warmup_updates,
            args.retry_updates,
        )
    ):
        parser.error("frame, subframes, tries and update counts must be positive")
    if args.stage_timeout_seconds <= 0:
        parser.error("stage timeout must be positive")

    camera = json.loads(camera_path.read_text())
    width, height = camera["render_px"]
    first_frame, last_frame = camera["animation_frames"]
    if (width, height) != (640, 480):
        parser.error("Step 2 requires the public 640x480 FillCam")
    if camera.get("overscan", 1.0) != 1.0:
        parser.error("Step 2 requires the unwarped public FillCam")
    if not first_frame <= args.frame <= last_frame:
        parser.error(f"frame must be in the public range {first_frame}-{last_frame}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    portable_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    faulthandler.dump_traceback_later(120, repeat=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "PathTracing",
            "width": width,
            "height": height,
            "extra_args": [
                "--portable-root",
                str(portable_root),
                "--/telemetry/enableAnonymousData=false",
                "--/structuredLog/enable=false",
            ],
        }
    )
    print("FILL_LEVEL_RUNTIME_READY", flush=True)
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.replicator.core")
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, UsdGeom

        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"could not open stage: {stage_path}")
        deadline = time.monotonic() + args.stage_timeout_seconds
        while context.get_stage_loading_status()[2] > 0:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"stage load exceeded {args.stage_timeout_seconds:g} seconds"
                )
            app.update()
        stage = context.get_stage()
        print("FILL_LEVEL_STAGE_LOADED", flush=True)

        camera_prim_path = next(
            (
                path
                for path in (
                    camera["prim_in_csi_line"],
                    camera["prim_in_station_file"],
                )
                if stage.GetPrimAtPath(path).IsValid()
            ),
            None,
        )
        if camera_prim_path is None:
            raise RuntimeError(f"FillCam was not found in {stage_path}")
        camera_prim = stage.GetPrimAtPath(camera_prim_path)
        if not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"{camera_prim_path} is not a USD camera")

        timeline = omni.timeline.get_timeline_interface()
        timeline.set_current_time(args.frame / stage.GetTimeCodesPerSecond())
        app.update()

        render_product = rep.create.render_product(camera_prim_path, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        app.update()

        # Preserve the public renderer's session-layer clipping re-sync. This
        # restores the original value immediately and never edits the source USD.
        clipping = UsdGeom.Camera(camera_prim).GetClippingRangeAttr()
        original_clipping = clipping.Get()
        stage.SetEditTarget(stage.GetSessionLayer())
        clipping.Set(Gf.Vec2f(original_clipping[0] * 0.5, original_clipping[1]))
        app.update()
        clipping.Set(original_clipping)
        app.update()

        for _ in range(args.warmup_updates):
            app.update()
        print("FILL_LEVEL_WARMUP_COMPLETE", flush=True)

        pixels = None
        channel_std = None
        attempt = 0
        for attempt in range(1, args.max_tries + 1):
            rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
            candidate = np.asarray(annotator.get_data())
            valid_shape = (
                candidate.ndim == 3
                and candidate.shape[:2] == (height, width)
                and candidate.shape[2] >= 3
            )
            if valid_shape:
                candidate = candidate[:, :, :3].copy()
                channel_std = candidate.astype(np.float64).std(axis=(0, 1))
                if float(channel_std.max()) > 1.0:
                    pixels = candidate
                    break
            print(
                json.dumps(
                    {
                        "attempt": attempt,
                        "shape": list(candidate.shape),
                        "spatial_std": (
                            channel_std.tolist() if channel_std is not None else None
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            for _ in range(args.retry_updates):
                app.update()
        if pixels is None:
            raise RuntimeError(
                "missing or spatially constant 640x480 RGB "
                f"after {args.max_tries} attempts"
            )

        if not cv2.imwrite(str(output_path), pixels[:, :, ::-1]):
            raise OSError(f"could not write {output_path}")
        decoded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != (height, width, 3):
            raise RuntimeError("written PNG could not be decoded at 640x480 RGB")
        decoded_rgb = decoded[:, :, ::-1]
        decoded_std = decoded_rgb.astype(np.float64).std(axis=(0, 1))
        if float(decoded_std.max()) <= 1.0:
            raise RuntimeError("written PNG is spatially constant")

        elapsed = time.monotonic() - started
        summary = {
            "schema_version": 1,
            "purpose": "renderer feasibility smoke; not camera-match evidence",
            "twin": str(twin),
            "stage": str(stage_path),
            "stage_sha256": file_sha256(stage_path),
            "camera_json": str(camera_path),
            "camera_json_sha256": file_sha256(camera_path),
            "camera_prim": camera_prim_path,
            "frame": args.frame,
            "renderer": "PathTracing",
            "resolution": [width, height],
            "subframes": args.subframes,
            "tries": attempt,
            "warmup_updates": args.warmup_updates,
            "retry_updates": args.retry_updates,
            "rgb_mean": decoded_rgb.astype(np.float64).mean(axis=(0, 1)).tolist(),
            "rgb_spatial_std": decoded_std.tolist(),
            "png_sha256": file_sha256(output_path),
            "renderer_script_sha256": file_sha256(Path(__file__)),
            "seconds": elapsed,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("FILL_LEVEL_SMOKE_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
        faulthandler.cancel_dump_traceback_later()
    except BaseException:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()


if __name__ == "__main__":
    main()
