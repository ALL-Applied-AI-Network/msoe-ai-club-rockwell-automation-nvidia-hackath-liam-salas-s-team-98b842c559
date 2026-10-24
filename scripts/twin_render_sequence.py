"""Render a FillCam frame sequence (RGB + semantic mask) from the CSI twin with Isaac Sim, headless.

Derived from the twin's tools/render_example.py. Differences:
  * skips render_example.py's clipping-range re-sync for FillCam; on FillCam it left the rgb annotator
    empty (shape (0,)), which is what the 2026-10-24 Codex Brev recovery run hit;
  * retries each capture until the rgb frame is full-size and not constant (a fresh Brev VM returned
    three uniformly black frames before materials finished loading);
  * steps through the fill animation instead of rendering one time code;
  * labels prims from their userProperties:synthetic_class so a semantic mask is captured too
    (no separate liquid class exists in the twin; liquid pixels fall under "vial").

Isaac Sim 6.0.1 pip install (Python 3.12) on a GPU host, then:

    OMNI_KIT_ACCEPT_EULA=YES python scripts/twin_render_sequence.py --twin <csi_digital_twin_v1> --out <dir>
    # aarch64 / DGX Spark (GB10): prefix with LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
"""
import argparse, json, os, time, traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--twin", required=True, help="csi_digital_twin_v1 folder")
ap.add_argument("--stage", default="stations/fill_station/fill_station.usda", help="relative to --twin")
ap.add_argument("--camera", default="FillCam")
ap.add_argument("--frames", default="1:687:15", help="start:stop:step time codes")
ap.add_argument("--subframes", type=int, default=32)
ap.add_argument("--renderer", default="PathTracing", choices=["PathTracing", "RaytracedLighting"])
ap.add_argument("--warmup", type=int, default=60, help="kit.update() calls before the first capture")
ap.add_argument("--max-tries", type=int, default=20, help="capture retries per frame until data is full-size")
ap.add_argument("--out", required=True)
ap.add_argument("--skip", default="", help="comma list of: clip,semantic (debug bisection)")
a = ap.parse_args()

twin = Path(a.twin).resolve()
cam = json.load(open(twin / "cameras" / f"{a.camera}.json"))
rw, rh = cam["render_px"]
out = Path(a.out).resolve(); (out / "rgb").mkdir(parents=True, exist_ok=True); (out / "semantic").mkdir(exist_ok=True)
start, stop, step = (int(x) for x in a.frames.split(":"))

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": a.renderer, "width": rw, "height": rh})
t0 = time.time()
skip = set(filter(None, a.skip.split(",")))
try:
    # omni.replicator.core is already loaded by Isaac Sim 6.0.1's default experience, so render_example.py's
    # enable_extension() call is unnecessary (it is harmless: retested with semantics, 22 s, frame OK).
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep
    from pxr import Gf, Usd, UsdGeom

    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(twin / a.stage)):
        raise RuntimeError(f"could not open stage {a.stage}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    stage = ctx.get_stage()
    prim = next(p for p in (cam["prim_in_csi_line"], cam["prim_in_station_file"]) if stage.GetPrimAtPath(p).IsValid())

    # Semantic labels from the twin's userProperties, written to the session layer only.
    stage.SetEditTarget(stage.GetSessionLayer())
    classes = {}
    try:
        if "semantic" in skip:
            raise RuntimeError("skipped")
        from isaacsim.core.utils.semantics import add_labels
        for p in stage.Traverse():
            attr = p.GetAttribute("userProperties:synthetic_class")
            if attr and attr.HasValue():
                cls = str(attr.Get()); add_labels(p, labels=[cls], instance_name="class")
                classes[cls] = classes.get(cls, 0) + 1
    except Exception as e:  # RGB is still useful without labels
        print("semantic labelling failed:", repr(e), flush=True)
    print("labelled prims per class:", classes, flush=True)

    rp = rep.create.render_product(prim, (rw, rh))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
    sem = None
    if "semantic" not in skip:
        sem = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False}); sem.attach([rp])
    rep.orchestrator.set_capture_on_play(False)

    # render_example.py's clipping-range re-sync is meant for CapSealCam's inch-scaled camera. On FillCam it
    # left the rgb annotator empty (shape (0,)) locally and in an earlier Brev run, so apply it only there.
    # Observed, not documented by NVIDIA.
    if a.camera == "CapSealCam" and "clip" not in skip:
        clip = UsdGeom.Camera(stage.GetPrimAtPath(prim)).GetClippingRangeAttr(); nf = clip.Get()
        clip.Set(Gf.Vec2f(nf[0] * 0.5, nf[1])); kit.update(); clip.Set(nf)
    for _ in range(a.warmup):
        kit.update()
    print(f"warm-up done at {time.time() - t0:.1f}s", flush=True)

    tl = omni.timeline.get_timeline_interface(); tps = stage.GetTimeCodesPerSecond()
    records = []
    for f in range(start, stop, step):
        tl.set_current_time(f / tps); kit.update()
        ts = time.time()
        for attempt in range(1, a.max_tries + 1):
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
            img = np.asarray(rgb.get_data())
            # On a fresh machine the first path-traced captures come back full-size but uniformly black
            # while materials/shaders finish loading, so require spatial content as well as shape.
            if img.ndim == 3 and img.shape[:2] == (rh, rw) and img[..., :3].std() > 1.0:
                break
        else:
            raise RuntimeError(f"frame {f}: rgb shape {img.shape} still empty or constant after {a.max_tries} steps")
        img = img[..., :3]
        cv2.imwrite(str(out / "rgb" / f"{f:04d}.png"), img[..., ::-1])
        sd = sem.get_data() if sem else np.zeros(0); mask = np.asarray(sd["data"]) if isinstance(sd, dict) else np.asarray(sd)
        if mask.shape[:2] == (rh, rw):
            cv2.imwrite(str(out / "semantic" / f"{f:04d}.png"), mask.astype(np.uint16))
        rec = {"frame": f, "tries": attempt, "seconds": round(time.time() - ts, 2),
               "rgb_mean": round(float(img.mean()), 2), "rgb_std": round(float(img.std()), 2),
               "semantic_ids": sorted(int(x) for x in np.unique(mask)) if mask.shape[:2] == (rh, rw) else None,
               "semantic_info": (sd.get("info", {}).get("idToLabels") if isinstance(sd, dict) else None)}
        records.append(rec); print(json.dumps(rec, default=str), flush=True)

    summary = {"camera": a.camera, "stage": a.stage, "renderer": a.renderer, "subframes": a.subframes,
               "resolution": [rw, rh], "frames": len(records), "total_seconds": round(time.time() - t0, 1),
               "classes": classes, "records": records}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("wrote", out, flush=True)
except BaseException:
    traceback.print_exc()
    kit.close(exit_code=1)
    raise
kit.close()
