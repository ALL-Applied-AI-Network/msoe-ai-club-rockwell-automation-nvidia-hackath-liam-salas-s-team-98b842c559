#!/usr/bin/env python3
"""Build the reviewable Phase 1 mock fixture and local presentation assets."""
import argparse, copy, hashlib, json, shutil, subprocess
from pathlib import Path
from PIL import Image

PAIRS = [
("PAIR-BATCH_0ebf3f287cc1291a-dispense-1-t200-V1-V2",6.699801316215474),
("PAIR-BATCH_0ebf3f287cc1291a-dispense-1-t200-V1-V3",6.2970588604995275),
("PAIR-BATCH_0ebf3f287cc1291a-dispense-2-t400-V1-V2",6.438614236887922),
("PAIR-BATCH_0ebf3f287cc1291a-dispense-2-t400-V1-V3",6.980229689829405),
("PAIR-BATCH_0ebf3f287cc1291a-dispense-3-t100-V1-V2",4.494294414282912),
("PAIR-BATCH_0ebf3f287cc1291a-dispense-3-t100-V1-V3",3.7824264172035935),
("PAIR-BATCH_5a4a586e3ec3b085-dispense-1-t200-V1-V2",29.07563861068292),
("PAIR-BATCH_5a4a586e3ec3b085-dispense-1-t200-V1-V3",28.067765245336204),
("PAIR-BATCH_5a4a586e3ec3b085-dispense-2-t400-V1-V2",28.92631630447299),]
OUTCOMES=["keep","discard","insufficient_evidence","keep","discard","keep","discard","keep","keep"]

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(__file__).parents[1]); ap.add_argument("--locator",type=Path,required=True); ap.add_argument("--source",type=Path,required=True); ap.add_argument("--ffmpeg",type=Path,default=Path("ffmpeg")); a=ap.parse_args()
    root=a.root; out=root/"ui-data"; assets=out/"assets"; (assets/"vials").mkdir(parents=True,exist_ok=True); (assets/"run").mkdir(parents=True,exist_ok=True)
    loc=json.loads(a.locator.read_text())["images"]
    pairs=[]
    meas=json.loads((a.source/"measurements.json").read_text()) if (a.source/"measurements.json").exists() else None
    for i,(pid,de) in enumerate(PAIRS,1):
        key=pid.removeprefix("PAIR-").rsplit("-",2)[0]
        # select the pair's right stage locator from its exact evidence key
        stage=pid.split("-V1-V")[0].removeprefix("PAIR-")
        vial="V"+pid.rsplit("-V",1)[1]
        batch, stage_name = stage.split("-",1)
        lkey=f"{batch}-{vial}-{stage_name}"
        if lkey not in loc: raise SystemExit(f"unvalidated locator key: {lkey}")
        src=Path(loc[lkey]["rgb_path"]); mask=Path(loc[lkey]["mask_path"])
        if not src.is_file() or not mask.is_file() or sha(src)!=loc[lkey]["rgb_sha256"] or sha(mask)!=loc[lkey]["mask_sha256"]: raise SystemExit(f"source drift: {lkey}")
        Image.open(src).convert("RGB").resize((320,240)).save(assets/"vials"/f"vial-{i:03d}-rgb.jpg",quality=88)
        Image.open(mask).convert("L").resize((320,240)).save(assets/"vials"/f"vial-{i:03d}-mask.png")
        pairs.append((pid,de,lkey))
    d=json.loads((out/"example.json").read_text()); d["dataset_id"]="phase1-run-01-mock"; d["label"]="Synthetic twin data — RUN-01 mock"; d["run"]["run_id"]="WP1-TEST-RUN-01"; d["run"]["title"]="RUN-01 operator mock"; d["run"]["video"]["src"]="/assets/run/run-01-composite.mp4"; d["run"]["video"]["asset_id"]="run-01-composite"
    for i,v in enumerate(d["vials"]):
        pid,de,key=pairs[i]; outc=OUTCOMES[i]; ev=f"evidence-{pid}"; v["slot"]=i+1; v["vial_id"]=f"vial-{i+1:03d}"; v["predicted_mask"]["src"]=f"/assets/vials/vial-{i+1:03d}-mask.png"; v["predicted_mask"]["asset_id"]=f"mask-vial-{i+1:03d}"; v["decision"]["outcome"]=outc; v["decision"]["delta_e"]={"value":de,"provenance":{"kind":"measured","source_id":pid,"note":"WP5 audited measured CIE76 Delta E."}}; v["decision"]["threshold"]={"value":10.0,"provenance":{"kind":"documented","source_id":"phase1-mock-threshold-10","note":"Documented mock threshold; not a plant limit."}}; v["decision"]["limit_codes"]=["measurement_unavailable"] if outc=="insufficient_evidence" else (["within_delta_e_limit"] if outc=="keep" else ["exceeds_delta_e_limit"]); v["decision"]["evidence_refs"]=[ev]; v["decision"]["explanation"]={"source":"deterministic_template","text":f"Mock-only decision from measured Delta E {de:.6f} against documented threshold 10.0; WP5 provides measurements only.","evidence_refs":[ev]}; v["_mock_evidence"]= [{"evidence_id":ev,"vial_id":v["vial_id"],"target":"delta_e","label":pid,"provenance":{"kind":"measured","source_id":pid,"note":"WP5 audited evidence identifier."}}]
        obs = d["top_down"]["observations"][i]
        obs["vial_id"]=v["vial_id"]; obs["decision_id"]=v["decision"]["record_id"]
    d["evidence"]=sum((v.pop("_mock_evidence") for v in d["vials"]),[]); d["top_down"]["observations"]=[dict(d["top_down"]["observations"][i]) for i in range(9)]; d["run"]["cue_times"]=[{"vial_id":v["vial_id"],"start_seconds":i*2,"end_seconds":i*2+1,"provenance":{"kind":"inferred","source_id":"run-01-composite","note":"Derived from built composite timing."}} for i,v in enumerate(d["vials"])]
    (out/"mock-run.json").write_text(json.dumps(d,indent=2)+"\n")
    # A compact playable local MP4 from accepted RGB states (presentation-only sequence).
    frames=sorted(a.source.glob("segment-*/render/states/state-*.png"))[:10]
    if not frames: raise SystemExit("no accepted state frames")
    tmp=assets/"run"/".frames"; tmp.mkdir(exist_ok=True)
    for i,p in enumerate(frames): Image.open(p).convert("RGB").resize((640,480)).save(tmp/f"{i:03d}.jpg",quality=80)
    subprocess.run([str(a.ffmpeg),"-y","-framerate","2","-i",str(tmp/"%03d.jpg"),"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(assets/"run"/"run-01-composite.mp4")],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp)
if __name__=="__main__": main()
