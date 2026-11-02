"""Measure the frozen stream detector on a real clip, frame by frame (no models).

Usage: stream_colour_probe.py VIDEO OUT_JSON
Uses the frozen tracker freeze and measure_stream_lab with the first frame as the empty
reference (exactly what the live path does), and records per native frame: active flag,
changed-pixel count, Lab centre/spread and nearest frozen tank. Also records the frozen
thresholds and tank Lab centres so the report can compare them with the real colour.
"""
import json, math, sys
import av, numpy as np
from pathlib import Path
from fluidmix.pipeline.service import production_factory as pf
from fluidmix.perception.tracker.fill_step_production import (
    load_pixel_to_metre_calibration, load_topology_freeze, load_tracker_freeze, measure_stream_lab,
)
from fluidmix.storage.artifacts.fill_step_files import read_regular_file

def load_freeze():
    cal = load_pixel_to_metre_calibration(read_regular_file(pf.CALIBRATION, "c"), expected_sha256=pf.EXPECTED_HASHES[pf.CALIBRATION])
    topo = load_topology_freeze(read_regular_file(pf.TOPOLOGY_FREEZE, "t"), expected_sha256=pf.EXPECTED_HASHES[pf.TOPOLOGY_FREEZE])
    return load_tracker_freeze(
        read_regular_file(pf.TRACKER_MATERIALIZATION, "m"),
        expected_materialization_sha256=pf.EXPECTED_HASHES[pf.TRACKER_MATERIALIZATION],
        train_freeze_raw=read_regular_file(pf.TRACKER_TRAIN_FREEZE, "f"),
        expected_train_freeze_sha256=pf.EXPECTED_HASHES[pf.TRACKER_TRAIN_FREEZE],
        topology=topo, calibration=cal,
        target_recipe_artifact_sha256="f12c1fe65a994a89e9c5530c32a2c1f10954fc3ecb0d803511ca82296f99c626")

if __name__ == "__main__":
    video, out = sys.argv[1], sys.argv[2]
    fz = load_freeze()
    c = av.open(video); s = c.streams.video[0]; fps = float(s.average_rate)
    ref = None; rows = []
    for n, fr in enumerate(c.decode(s)):
        rgb = fr.to_ndarray(format="rgb24")
        if ref is None:
            ref = rgb.copy(); continue
        act, cnt, ctr, spr = measure_stream_lab(rgb, ref, fz)
        near = None
        if ctr is not None:
            near = min(fz.tank_lab_centres, key=lambda t: math.dist(ctr, fz.tank_lab_centres[t]))
        rows.append({"n": n, "t": n / fps, "active": act, "count": cnt, "lab": ctr, "spread": spr, "nearest": near,
                     "dist": None if ctr is None else {t: math.dist(ctr, v) for t, v in fz.tank_lab_centres.items()}})
    json.dump({"video": video, "fps": fps, "frozen": {
        "roi": fz.stream_roi_xyxy, "rgb_change_threshold": fz.stream_rgb_change_threshold,
        "min_changed_pixels": fz.stream_minimum_changed_pixels,
        "min_consecutive_frames": fz.stream_minimum_consecutive_frames,
        "tank_lab_centres": {k: list(v) for k, v in fz.tank_lab_centres.items()},
        "tank_cluster_delta_e76": fz.tank_cluster_delta_e76}, "rows": rows}, open(out, "w"))
    a = [r for r in rows if r["active"]]
    print(video[-22:], "frames", len(rows), "active", len(a))
