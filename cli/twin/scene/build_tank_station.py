"""Build an approximate tank asset and optionally compose it with the public twin."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from twin.scene.tank_station import TankGeometry, TANK_IDS, build_station, compose_line, set_colors, export_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--twin", type=Path, help="original csi_digital_twin_v1 directory")
    parser.add_argument("--samples", type=Path, help='JSON list: [{"frame":1,"volumes":{"T100":1000,...}}, ...]')
    parser.add_argument("--colors", type=Path, help='JSON mapping tank IDs to linear RGB triples in [0,1]')
    parser.add_argument("--end-frame", type=int, default=686)
    parser.add_argument("--snapshot", type=Path, help="also export a compact static USD asset at frame 1")
    parser.add_argument("--validation-sweep", action="store_true", help="nine-frame synthetic empty/full/independent-level capture canary")
    parser.add_argument("--radius", type=float, default=.07)
    parser.add_argument("--body-height", type=float, default=.16)
    parser.add_argument("--cone-height", type=float, default=.10)
    parser.add_argument("--position", type=float, nargs=3, default=[-1.1,0,0],
                        help="assumed offset from fill-station frame, in metres")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists; choose a fresh run directory")
    geometry = TankGeometry(args.radius,args.body_height,args.cone_height)
    samples = None
    if args.samples:
        rows = json.loads(args.samples.read_text())
        samples = [(row["frame"],row["volumes"]) for row in rows]
    if args.validation_sweep:
        if args.samples:
            parser.error("validation-sweep cannot be combined with samples")
        fractions = [(0,0,0,0),(.02,.02,0,.02),(.17,.17,0,.17),(.5,.5,0,.5),
                     (1,1,0,1),(.8,0,0,.1),(0,.8,0,.1),(.1,.1,0,0),(0,0,0,0)]
        args.end_frame = len(fractions)
        samples = [(i+1,{t:f*geometry.capacity_ml for t,f in zip(TANK_IDS,row)})
                   for i,row in enumerate(fractions)]
    path = args.out / "tank_station.usda"
    stage = build_station(path,geometry,samples,args.end_frame)
    if args.colors:
        set_colors(stage,json.loads(args.colors.read_text()))
        stage.GetRootLayer().Save()
    # Layer export preserves relative texture paths. Stage.Export flattens and
    # resolves them to this machine's absolute paths, breaking remote packages.
    stage.GetRootLayer().Export(str(args.out/"tank_station.usdc"))
    if args.snapshot:
        export_snapshot(stage,args.snapshot)
    if args.twin:
        compose_line(path,args.twin,args.out/"csi_line_with_tanks.usda",args.position)
    manifest = {"station": "tank_station.usda", "camera": "/TankStation/TankCam",
                "line_camera": "/CSI_Line/TankPlacement/TankStation/TankCam",
                "capacity_ml_per_tank":geometry.capacity_ml,"dimensions_measured":False,
                "camera_calibrated":False,"placement_calibrated":False,
                "volume_source":str(args.samples.resolve()) if args.samples else
                "synthetic validation sweep" if args.validation_sweep else "illustrative drawdown",
                "position_in_fill_frame_m": args.position,"end_frame":args.end_frame,"fps":30}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
