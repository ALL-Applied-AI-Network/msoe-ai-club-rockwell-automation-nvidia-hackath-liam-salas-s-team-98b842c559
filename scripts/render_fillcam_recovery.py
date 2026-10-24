"""Reviewed Isaac entry point for the one fixed frame-686 FillCam recovery smoke."""
import argparse, json, os, sys, tempfile, traceback
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from fluidmix.render_runner import (
    attest_live_async_setting,
    resolve_private_new_output,
    validate_recovery_kit_args,
    validate_recovery_render_request,
)
ap = argparse.ArgumentParser()
ap.add_argument("--twin", required=True)
ap.add_argument("--camera", required=True, choices=["FillCam"],
                help="camera name; settings come from cameras/<name>.json")
ap.add_argument("--stage", required=True, help="station file under the supplied twin root")
ap.add_argument("--frame", type=int, default=1, help="time code (fill station animation: 1-686 at 30 fps)")
ap.add_argument("--out", default="render.png")
ap.add_argument("--subframes", type=int, default=32, help="path-tracing subframes (quality)")
ap.add_argument("--kit-arg", action="append", default=[], help="extra Isaac Sim / Kit argument (repeatable)")
a = ap.parse_args()
validate_recovery_kit_args(a.kit_arg)

twin = Path(a.twin).expanduser().resolve(strict=True)
cam = json.load((twin / "cameras" / (a.camera + ".json")).open())
stage_path = Path(a.stage).expanduser()
if not stage_path.is_absolute():
    stage_path = twin / stage_path
stage_path = str(stage_path.resolve())
if not os.path.isfile(stage_path):
    ap.error(f"stage does not exist: {stage_path}")
if a.frame < 1 or (a.camera == "FillCam" and a.frame > 686):
    ap.error("use a positive time code; the fill animation covers 1-686")
if a.subframes < 1:
    ap.error("--subframes must be positive")
requested_output = resolve_private_new_output(a.out, expected_name="S00.png")
output_parent = requested_output.parent
a.out = str(requested_output)
rw, rh = cam["render_px"]
stage_path = str(validate_recovery_render_request(
    twin, stage_path, a.camera, a.frame, a.subframes, (rw, rh)
))

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": rw, "height": rh, "extra_args": a.kit_arg})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import carb.settings
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep

    live_setting = attest_live_async_setting(carb.settings.get_settings())
    print("live_setting_attestation", json.dumps(live_setting, sort_keys=True), flush=True)

    ctx = omni.usd.get_context()
    if not ctx.open_stage(stage_path):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    prim = next((p for p in [cam["prim_in_csi_line"], cam["prim_in_station_file"]]
                 if stage.GetPrimAtPath(p).IsValid()), None)
    if prim is None:
        raise RuntimeError(f"camera {a.camera} not found in {stage_path}")

    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(a.frame / stage.GetTimeCodesPerSecond())
    kit.update()

    rp = rep.create.render_product(prim, (rw, rh))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    # Preserve the public renderer's pre-existing session-layer clipping-range re-sync. This does
    # not alter the authored twin and is unchanged from the prior black-frame attempt.
    from pxr import Gf, UsdGeom
    clip = UsdGeom.Camera(stage.GetPrimAtPath(prim)).GetClippingRangeAttr()
    near_far = clip.Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
    kit.update()
    clip.Set(near_far)
    kit.update()

    for _ in range(2):
        rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
    data = np.asarray(ann.get_data())
    if data.ndim != 3 or data.shape[:2] != (rh, rw) or data.shape[2] < 3:
        raise RuntimeError(f"RGB capture returned shape {data.shape}; check GPU/RTX startup errors")
    img = data[..., :3].copy()

    # Save before close(): Isaac's default fast shutdown exits the process. The fixed FillCam
    # contract has no lens post-processing. Link an owned temporary into the final name so a
    # newly appeared final path fails instead of being overwritten or followed.
    if cam.get("overscan", 1.0) != 1.0:
        raise RuntimeError("recovery contract requires the unwarped FillCam view")
    with tempfile.NamedTemporaryFile(
        prefix=".S00.", suffix=".png", dir=output_parent, delete=False
    ) as temporary:
        temporary_path = temporary.name
    try:
        if not cv2.imwrite(temporary_path, img[..., ::-1]):
            raise OSError(f"could not write temporary PNG in {output_parent}")
        os.link(temporary_path, a.out, follow_symlinks=False)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    print("wrote", a.out, flush=True)
except BaseException:
    traceback.print_exc()
    # Preserve failure status even when Kit's fast shutdown terminates Python.
    kit.close(exit_code=1)
    raise
kit.close()
