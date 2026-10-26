"""Validate captured RGB/instance pairs against the exact authored USD stage.

This checks synthetic data integrity, not physical accuracy or real-world transfer.
The optional sweep gate applies only to build_tank_station --validation-sweep.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd
from tank_capture_provenance import verify_source_dependencies

TANK_IDS = ("T100", "T200", "T300", "T400")


def check_capture(directory, stage_path, sweep=False):
    """Return a compact audit, raising ValueError on any mismatched pair.

    Args:
        directory: Capture folder containing summary.json and lossless images.
        stage_path: Exact standalone or composed USD used during capture.
        sweep: Require all nine synthetic validation states and monotonic masks.
    """
    directory, stage_path = Path(directory), Path(stage_path)
    summary = json.loads((directory/"summary.json").read_text())
    verify_source_dependencies(stage_path, summary.get("source_dependencies"))
    if summary["stage_sha256"] != hashlib.sha256(stage_path.read_bytes()).hexdigest():
        raise ValueError("capture source hash does not match supplied stage")
    stage = Usd.Stage.Open(str(stage_path))
    roots = [p for p in stage.Traverse() if p.GetAttribute("fluidmix:volumeMl")]
    if len(roots) != 4 or {p.GetName() for p in roots} != set(TANK_IDS):
        raise ValueError("capture requires exactly four uniquely named tanks")
    rows, seen = [], set()
    for record in summary["records"]:
        frame = record["frame"]
        if not np.isfinite(frame) or frame != int(frame) or frame in seen or not stage.GetStartTimeCode() <= frame <= stage.GetEndTimeCode():
            raise ValueError("invalid or duplicate capture frame")
        seen.add(frame)
        stem = f"{frame:07.2f}"
        rgb_path = directory/(stem+".png")
        if hashlib.sha256(rgb_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("RGB hash mismatch")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        instances = np.asarray(Image.open(directory/(stem+".instances.png")))
        binary = np.asarray(Image.open(directory/(stem+".liquid.png")))
        if rgb.shape != (480,640,3) or instances.shape != (480,640) or binary.shape != (480,640):
            raise ValueError("capture dimensions must be 640x480")
        if max(rgb.astype(float).std(axis=(0,1))) <= 1:
            raise ValueError("spatially constant RGB")
        if not set(np.unique(instances)).issubset({0,1,2,4}):
            raise ValueError("unknown instance or inactive T300 pixels")
        if not np.array_equal(binary, (instances > 0).astype("uint8")*255):
            raise ValueError("binary and instance masks disagree")
        if record["liquid_pixels"] != int((instances > 0).sum()):
            raise ValueError("liquid count mismatch")
        for prim in roots:
            tank = prim.GetName()
            index = TANK_IDS.index(tank)+1
            info = record["tanks"][tank]
            volume = prim.GetAttribute("fluidmix:volumeMl").Get(frame)
            if not np.isfinite(volume) or not 0 <= volume <= prim.GetAttribute("fluidmix:capacityMl").Get():
                raise ValueError("invalid authored volume")
            if info["volume_ml"] != volume or record["volumes_ml"][str(prim.GetPath())] != volume:
                raise ValueError("volume label differs from USD")
            region = instances == index
            ys,xs = np.nonzero(region)
            bbox = [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if region.any() else None
            if info["instance_id"] != index or info["pixels"] != int(region.sum()) or info["bbox_xyxy"] != bbox:
                raise ValueError("instance count or bounding box mismatch")
            if (volume > 0) != bool(region.any()) or (tank == "T300" and volume != 0):
                raise ValueError("empty/nonempty mask disagrees with volume")
        rows.append({"frame":frame,"liquid_pixels":record["liquid_pixels"],
                     "tanks":record["tanks"],"rgb_sha256":record["sha256"]})
    if not rows:
        raise ValueError("capture has no records")
    if sweep:
        by_frame = {r["frame"]:r for r in rows}
        if set(by_frame) != set(range(1,10)):
            raise ValueError("sweep requires frames 1 through 9")
        for tank in ("T100","T200","T400"):
            areas = [by_frame[f]["tanks"][tank]["pixels"] for f in range(1,6)]
            if any(b <= a for a,b in zip(areas,areas[1:])):
                raise ValueError("sweep liquid masks must grow from empty to full")
        if by_frame[1]["liquid_pixels"] or by_frame[9]["liquid_pixels"]:
            raise ValueError("sweep must start and end empty")
        # Different authored states must not silently capture the same RGB.
        if len({by_frame[f]["rgb_sha256"] for f in range(1,9)}) != 8:
            raise ValueError("distinct sweep states reused RGB")
    return {"passed":True,"scope":"synthetic capture integrity only","frames":len(rows),
            "source_dependencies":summary["source_dependencies"],
            "stage_sha256":summary["stage_sha256"],"sweep":sweep,"records":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",type=Path,required=True)
    parser.add_argument("--stage",type=Path,required=True)
    parser.add_argument("--sweep",action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_capture(args.capture,args.stage,args.sweep),indent=2))


if __name__ == "__main__":
    main()
