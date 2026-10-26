"""Capture tank USD frames with the existing Isaac Sim runtime.

Use the runtime's Python. On Spark the existing install needs
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1. Run under an outer timeout.
No source layers are modified; runtime semantic labels live in the session layer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def semantic_ids(labels, category):
    """Resolve a semantic class independent of runtime label capitalization."""
    return [int(key) for key,value in labels.items()
            if str(value.get("class", "")).casefold() == category.casefold()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",type=Path,required=True)
    parser.add_argument("--camera",default="/TankStation/TankCam")
    parser.add_argument("--frames",default="1,343,686")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--subframes",type=int,default=32)
    parser.add_argument("--max-tries",type=int,default=10)
    parser.add_argument("--portable-root",type=Path,help="optional private Kit data/cache/log root")
    args = parser.parse_args()
    frames = [float(frame) for frame in args.frames.split(",")]
    if not args.stage.is_file() or args.out.exists():
        parser.error("stage must exist and output directory must be fresh")
    if args.subframes < 1 or args.max_tries < 1:
        parser.error("subframes and max-tries must be positive")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)
    from isaacsim import SimulationApp
    config = {"headless":True,"renderer":"PathTracing","width":640,"height":480}
    if args.portable_root:
        config["extra_args"] = ["--portable-root",str(args.portable_root.resolve())]
    app = SimulationApp(config)
    print("TANK_RUNTIME_READY",flush=True)
    try:
        import numpy as np
        from PIL import Image
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from isaacsim.core.utils.semantics import add_labels
        from pxr import UsdGeom

        context = omni.usd.get_context()
        if not context.open_stage(str(args.stage.resolve())):
            raise RuntimeError("failed to open tank stage")
        deadline = time.monotonic()+120
        while context.get_stage_loading_status()[2] > 0:
            if time.monotonic()>deadline:
                raise TimeoutError("stage load exceeded 120 seconds")
            app.update()
        stage = context.get_stage()
        print("TANK_STAGE_LOADED",flush=True)
        camera = stage.GetPrimAtPath(args.camera)
        if not camera or not camera.IsA(UsdGeom.Camera):
            raise ValueError("camera does not exist or is not a USD camera")
        if len(set(frames)) != len(frames) or any(not np.isfinite(f) or f != int(f) or not stage.GetStartTimeCode() <= f <= stage.GetEndTimeCode() for f in frames):
            raise ValueError("capture frames must be unique integers within the stage timeline")
        stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            attr = prim.GetAttribute("userProperties:synthetic_class")
            if attr and attr.HasValue():
                category = str(attr.Get())
                if category == "liquid":
                    category += "_"+prim.GetParent().GetName().lower()
                add_labels(prim, labels=[category], instance_name="class")
        shells = [UsdGeom.Imageable(prim) for prim in stage.Traverse()
                  if prim.GetName() == "Shell" and prim.GetParent().GetAttribute("fluidmix:volumeMl")]
        product = rep.create.render_product(args.camera,(640,480))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        semantic = rep.AnnotatorRegistry.get_annotator("semantic_segmentation",init_params={"colorize":False})
        semantic.attach([product])
        rep.orchestrator.set_capture_on_play(False)
        print("TANK_PRODUCTS_READY",flush=True)
        for _ in range(60):
            app.update()
        timeline = omni.timeline.get_timeline_interface()
        print("TANK_WARMUP_COMPLETE",flush=True)
        records = []
        for frame in frames:
            timeline.set_current_time(frame/stage.GetTimeCodesPerSecond())
            app.update()
            for attempt in range(1,args.max_tries+1):
                rep.orchestrator.step(rt_subframes=args.subframes,delta_time=0.0)
                pixels = np.asarray(rgb.get_data())
                if pixels.ndim == 3 and pixels.shape[:2] == (480,640) and pixels.shape[2] >= 3:
                    pixels = pixels[:,:,:3].copy()
                    if np.max(pixels.astype(float).std(axis=(0,1))) > 1:
                        break
                print(json.dumps({"frame":frame,"attempt":attempt,"shape":list(pixels.shape),
                                  "spatial_std":pixels.astype(float).std(axis=(0,1)).tolist()
                                  if pixels.ndim == 3 else None}),flush=True)
                # Permit asynchronous texture/shader work to progress between
                # bounded capture attempts; an empty annotator is never success.
                for _ in range(30):
                    app.update()
            else:
                raise RuntimeError(f"frame {frame}: missing or spatially constant RGB after {args.max_tries} tries")
            # Label the projected liquid geometry through the translucent wall.
            # Keep labels, level strips and plumbing as occluders. This separate
            # pass avoids treating the wall's renderer-dependent alpha as truth.
            for shell in shells:
                shell.GetVisibilityAttr().Set("invisible")
            app.update()
            rep.orchestrator.step(rt_subframes=args.subframes,delta_time=0.0)
            data = semantic.get_data()
            mask = np.asarray(data["data"])
            labels = data.get("info",{}).get("idToLabels",{})
            print(json.dumps({"annotation_frame":frame,"labels":labels,
                              "mask_ids":np.unique(mask).tolist()}),flush=True)
            liquid_ids = [key for tank in ("T100","T200","T300","T400")
                          for key in semantic_ids(labels,"liquid_"+tank)]
            if mask.shape != (480,640):
                raise RuntimeError("semantic image has wrong shape")
            name = f"{frame:07.2f}"
            image_path = args.out/(name+".png")
            Image.fromarray(pixels).save(image_path)
            np.save(args.out/(name+".semantic.npy"),mask)
            liquid = np.isin(mask,liquid_ids)
            volumes = {str(prim.GetPath()): prim.GetAttribute("fluidmix:volumeMl").Get(frame)
                       for prim in stage.Traverse() if prim.GetAttribute("fluidmix:volumeMl")}
            if any(volume > 0 for volume in volumes.values()) and not liquid.any():
                raise RuntimeError("nonempty tanks produced no liquid mask; check transparency or occlusion")
            instances = np.zeros(mask.shape,dtype="uint8")
            per_tank = {}
            for index,tank in enumerate(("T100","T200","T300","T400"),1):
                ids = semantic_ids(labels,"liquid_"+tank)
                region = np.isin(mask,ids)
                volume = next((v for p,v in volumes.items() if p.endswith("/"+tank)),None)
                if volume is None or (volume > 0 and not region.any()) or (volume == 0 and region.any()):
                    raise RuntimeError(f"{tank}: mask disagrees with authored empty/nonempty volume")
                instances[region] = index
                ys,xs = np.nonzero(region)
                per_tank[tank] = {"instance_id":index,"volume_ml":volume,"pixels":int(region.sum()),
                                  "bbox_xyxy": [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if region.any() else None}
            Image.fromarray(instances).save(args.out/(name+".instances.png"))
            Image.fromarray((liquid*255).astype("uint8")).save(args.out/(name+".liquid.png"))
            record = {"frame":frame,"tries":attempt,"rgb_mean":float(pixels.mean()),
                      "rgb_std":float(pixels.std()),"liquid_pixels":int(liquid.sum()),
                      "labels":labels,"sha256":hashlib.sha256(image_path.read_bytes()).hexdigest()}
            record["volumes_ml"] = volumes
            record["tanks"] = per_tank
            records.append(record)
            print(json.dumps(record),flush=True)
            for shell in shells:
                shell.GetVisibilityAttr().Clear()
            app.update()
        summary = {"schema_version":1,"stage":str(args.stage.resolve()),"camera":args.camera,"renderer":"PathTracing",
                   "renderer_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "purpose":"synthetic capture; assign sequence splits separately before training",
                   "resolution":[640,480],"subframes":args.subframes,"records":records,
                   "mask_policy":"projected liquid geometry; shells hidden only during annotation; other occluders retained",
                   "seconds":time.monotonic()-started,
                   "stage_sha256":hashlib.sha256(args.stage.read_bytes()).hexdigest()}
        (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print("TANK_RENDER_COMPLETE "+str(args.out),flush=True)
        faulthandler.cancel_dump_traceback_later()
    except BaseException:
        import traceback
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()


if __name__ == "__main__":
    main()
