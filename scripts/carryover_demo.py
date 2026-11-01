#!/usr/bin/env python
"""End-to-end carryover demo for video 5: link each vial's colour decision to its top-down camera ID
and render three-camera clips (tank_level | vial_fill | cap_seal) with cv2 only (no ffmpeg on this box).

Inputs (all produced earlier, none of them changed here; RECORDS_DIR defaults to outputs/carryover-video5):
  RECORDS_DIR/records.json            per-vial colour decisions (scripts/carryover_video5.py)
  RECORDS_DIR/topdown/detections.jsonl per-frame boxes and tracks from the top-down tracker
  RECORDS_DIR/topdown/summary.json     tracker exit events (tracker_exit_events; older runs: exits) and
                                       reference-line crossings (line_crossings)
  data/video5_vial_recipes.csv         PLC recipe volumes (label stand-in for the control signal)
  the three frame-synchronized videos *_video_5.mp4 under --dataset (same frame index = same instant);
  --dataset defaults to the csi-vial-fill-dataset folder below fluidmix.paths.data_root(), so FLUIDMIX_DATA
  (or .env) is honoured, else <repo>/external

Step 1 (links.json): a vial leaves the fill station a second or two after its fill_end_frame and later
enters the inspection lane of the cap_seal camera from the bottom, where the tracker gives it a new
camera ID (birth centre y > 436). Rule, applied greedily in vial order: the first lane-entry event born
after fill_end_frame + 30 that is not yet assigned, within 90 s. Two documented adjustments to the raw
track list: (a) one or two-frame tracks with no follow-on birth within 60 frames are treated as
detector flicker, not an entry; (b) a track born within 60 frames after another entry track has ended
and that ended while still near the lane bottom (centre y > 400) is treated as the same vial re-identified (the tracker often fragments an entry into a short ID at
y~476 and a long ID at y~446), so one entry event can carry several fragment IDs; the longest-lived
fragment is reported as the camera ID. Entry events are fewer than vials (the tracker missed some),
so the greedy chain can be shifted by one vial after a miss; delay_s is reported so the reader can judge.

Step 2 (OUT/demo/): clips from fill_end_frame - 150 to fill_end_frame + 600 (extended to the linked ID's
birth + 150 frames when that birth is later, so the linked ID is actually seen), 30 fps, mp4v. The
decision card shows PENDING until the measurement window has closed, then the recorded decision.
Every number drawn comes from records.json or links.json. The clip range ends well before the vial's
reference-line crossing (on video 5 the crossing comes 748 to 1450 frames, 25 to 48 s, after the lane birth), so these
clips show the lane entry and the linked ID, not the crossing; scripts/carryover_lane_clip.py follows an
ID to the crossing and flashes it.

    python scripts/carryover_demo.py [--dataset DIR] [--records-dir DIR] [--out DIR]   # links + 4 changeover clips + 1 keep clip
    python scripts/carryover_demo.py --links-only
"""
import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix.paths import data_root, require

DATASET_LAYOUT = Path("csi-vial-fill-dataset")
DEFAULT_RECORDS_DIR = ROOT / "outputs/carryover-video5"
CAMS = ["tank_level", "vial_fill", "cap_seal"]
FPS = 30
ENTRY_Y = 436            # tracker entry line (models/topdown/real-evaluation-protocol.json)
LANE = (139.5, 0, 205, 480)
MIN_DELAY = 30           # frames after fill_end_frame before an entry may belong to the vial
MAX_DELAY = 90 * FPS     # 90 s
FLICKER_FRAMES = 3       # tracks shorter than this with no follow-on are ignored
FRAGMENT_GAP = 60        # a birth this soon after an entry track ended is the same vial ...
FRAGMENT_MAX_Y = 400     # ... provided that track ended while still near the lane bottom (centre y > 400)
CROP = (180, 150, 420, 430)   # x0, y0, x1, y1 of the measurement crop (fluidmix.carryover.measure)
PANEL_W, PANEL_H, HDR = 640, 480, 70
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, GREEN, GREY, YELLOW, WHITE, CYAN = (40, 40, 230), (40, 200, 60), (160, 160, 160), (0, 220, 255), (255, 255, 255), (230, 200, 0)

_RGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def default_dataset(explicit=None):
    """The organizers' csi-vial-fill-dataset folder: --dataset itself when given, else that folder below
    fluidmix.paths.data_root() (FLUIDMIX_DATA or .env, else <repo>/external)."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return data_root() / DATASET_LAYOUT


def tracker_exits(summary):
    """The tracker's own exit events: the current run script writes tracker_exit_events, older runs wrote exits."""
    return summary.get("tracker_exit_events", summary.get("exits", []))


def lab_to_bgr(lab):
    """Inverse of fluidmix.carryover.measure.lab (D65, sRGB matrix) -> uint8 BGR for display only."""
    L, a, b = [float(x) for x in lab]
    fy = (L + 16) / 116
    f = np.array([a / 500 + fy, fy, fy - b / 200])
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _D65
    lin = np.linalg.inv(_RGB_TO_XYZ) @ xyz
    lin = np.clip(lin, 0, 1)
    srgb = np.where(lin > 0.0031308, 1.055 * lin ** (1 / 2.4) - 0.055, 12.92 * lin)
    rgb = np.clip(np.round(srgb * 255), 0, 255).astype(int)
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def load_tracks(path):
    """Per-frame boxes/tracks plus a per-ID summary (birth, birth box, last frame, exit frame)."""
    frames, ids = {}, {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            f = d["frame"]
            frames[f] = d
            for t in d["tracks"]:
                s = ids.setdefault(t["id"], {"id": t["id"], "birth": f, "birth_xywh": t["bbox_xywh"], "last": f,
                                             "exit_frame": None})
                s["last"] = f
                s["last_xywh"] = t["bbox_xywh"]
                if t.get("exit_frame") is not None:
                    s["exit_frame"] = t["exit_frame"]
    return frames, ids


def entry_events(ids):
    """Group lane-entry births (centre y > ENTRY_Y) into entry events; drop flicker."""
    cands = sorted((s for s in ids.values() if s["birth_xywh"][1] + s["birth_xywh"][3] / 2 > ENTRY_Y),
                   key=lambda s: s["birth"])
    groups = []
    for s in cands:
        prev = groups[-1][-1] if groups else None
        if prev is not None and 0 <= s["birth"] - prev["last"] <= FRAGMENT_GAP \
                and prev["last_xywh"][1] + prev["last_xywh"][3] / 2 > FRAGMENT_MAX_Y:
            groups[-1].append(s)
        else:
            groups.append([s])
    events = []
    for g in groups:
        life = max(s["last"] for s in g) - g[0]["birth"] + 1
        if life < FLICKER_FRAMES:
            continue
        primary = max(g, key=lambda s: s["last"] - s["birth"])
        exits = [s["exit_frame"] for s in g if s["exit_frame"] is not None]
        events.append({"camera_id": primary["id"], "fragment_ids": [s["id"] for s in g],
                       "birth_frame": g[0]["birth"], "birth_centre_y": round(g[0]["birth_xywh"][1] + g[0]["birth_xywh"][3] / 2, 1),
                       "last_seen_frame": primary["last"], "last_seen_xywh": primary["last_xywh"],
                       "exit_frame": min(exits) if exits else None, "assigned_vial": None})
    return events


def link_vials(vials, events):
    links = []
    for r in vials:
        lo, hi = r["fill_end_frame"] + MIN_DELAY, r["fill_end_frame"] + MAX_DELAY
        ev = next((e for e in events if e["assigned_vial"] is None and lo <= e["birth_frame"] <= hi), None)
        link = {"vial": r["vial"], "recipe_id": r["recipe_id"], "decision": r["decision"],
                "fill_end_frame": r["fill_end_frame"], "camera_id": None, "fragment_ids": [], "birth_frame": None,
                "exit_frame": None, "last_seen_frame": None, "delay_s": None}
        if ev is not None:
            ev["assigned_vial"] = r["vial"]
            link.update({k: ev[k] for k in ("camera_id", "fragment_ids", "birth_frame", "exit_frame", "last_seen_frame")})
            link["delay_s"] = round((ev["birth_frame"] - r["fill_end_frame"]) / FPS, 1)
        links.append(link)
    return links


def print_links(links):
    print(f"{'vial':>4} {'recipe':>6} {'decision':>21} {'camera id':>9} {'fragments':>16} {'birth':>6} {'exit':>5} {'delay_s':>7} {'last seen':>9}")
    for l in links:
        print(f"{l['vial']:>4} {l['recipe_id']:>6} {l['decision']:>21} {str(l['camera_id']):>9} {'+'.join(l['fragment_ids']) or '-':>16} "
              f"{str(l['birth_frame']):>6} {str(l['exit_frame']):>5} {str(l['delay_s']):>7} {str(l['last_seen_frame']):>9}")
    n = sum(1 for l in links if l["camera_id"])
    print(f"linked {n} of {len(links)} vials to a top-down camera ID; "
          f"{sum(1 for l in links if l['exit_frame'] is not None)} have a recorded exit frame")
    return n


def read_recipes():
    with open(ROOT / "data/video5_vial_recipes.csv") as fh:
        return {int(r["vial"]): r for r in csv.DictReader(fh)}


def put(img, text, xy, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, text, xy, FONT, scale, color, thick, cv2.LINE_AA)


def header(title, sub, sub_color=GREY):
    h = np.zeros((HDR, PANEL_W, 3), np.uint8)
    put(h, title, (8, 28), 0.62, WHITE, 2)
    put(h, sub, (8, 56), 0.48, sub_color, 1)
    return h


def overlay_box(img, x0, y0, x1, y1, alpha=0.7):
    sub = img[y0:y1, x0:x1]
    dark = np.zeros_like(sub)
    img[y0:y1, x0:x1] = cv2.addWeighted(sub, 1 - alpha, dark, alpha, 0)


def first_sentence(text):
    i = text.find(". ")
    return text if i < 0 else text[:i + 1]


def vial_fill_panel(img, r, recipe, f):
    x0, y0, x1, y1 = CROP
    cv2.rectangle(img, (x0, y0), (x1, y1), YELLOW, 1)
    put(img, "measurement crop", (x0 + 4, y1 - 6), 0.42, YELLOW)
    w = r["window"]
    fill_done = f > r["fill_end_frame"]
    measured = w is not None and f >= w[1]
    # decision card
    overlay_box(img, 0, 0, PANEL_W, 132, 0.68)
    prev = "unknown" if r["prev_recipe_id"] is None else r["prev_recipe_id"]
    ctx = {1: "first after changeover", 0: f"repeat {r['position_in_run']} of same recipe"}.get(r["recipe_changed"], "previous recipe unknown")
    put(img, f"Vial {r['vial']}   recipe {r['recipe_id']}   previous recipe {prev}   ({ctx})", (8, 24), 0.5, WHITE, 1)
    if measured:
        dec = r["decision"]
        badge, color = {"discard": ("DISCARD", RED), "keep": ("KEEP", GREEN)}.get(dec, ("NO DECISION", GREY))
        de = "null" if r["dE_vs_reference"] is None else f"{r['dE_vs_reference']:.2f}"
        put(img, f"dE vs reference {de}   tau {r['tau']:.2f}   known-colour check: {r['known_colour_check']}", (8, 50), 0.48, WHITE)
        lines = textwrap.wrap(first_sentence(r["explanation"]), 78)[:3]
        for i, ln in enumerate(lines):
            put(img, ln, (8, 76 + 20 * i), 0.42, (220, 220, 220))
    elif fill_done:
        badge, color = "PENDING", GREY
        put(img, f"fill ended at frame {r['fill_end_frame']}; measuring final mixture" + (f" in frames {w[0]}-{w[1]}" if w else " (no window)"), (8, 50), 0.48, WHITE)
    else:
        badge, color = "FILLING", GREY
        put(img, f"PLC recipe {r['recipe_id']} dispensing (frames {r['fill_start_frame']}-{r['fill_end_frame']})", (8, 50), 0.48, WHITE)
    (tw, th), _ = cv2.getTextSize(badge, FONT, 0.9, 2)
    bx = PANEL_W - tw - 24
    cv2.rectangle(img, (bx - 8, 8), (PANEL_W - 8, 8 + th + 18), color, -1)
    put(img, badge, (bx, 8 + th + 8), 0.9, WHITE, 2)
    # swatches
    sy = 150
    for i, (lab, label) in enumerate([(r["measured_lab"], "measured"), (r["reference_lab"], f"reference (recipe {r['recipe_id']})")]):
        y = sy + i * 78
        overlay_box(img, 4, y - 4, 172, y + 70, 0.55)
        if lab is not None and (measured or i == 1):
            cv2.rectangle(img, (8, y), (56, y + 48), lab_to_bgr(lab), -1)
            cv2.rectangle(img, (8, y), (56, y + 48), WHITE, 1)
            put(img, label, (62, y + 16), 0.4, WHITE)
            put(img, f"L {lab[0]:.1f}", (62, y + 32), 0.38, (220, 220, 220))
            put(img, f"a {lab[1]:.1f} b {lab[2]:.1f}", (62, y + 46), 0.38, (220, 220, 220))
        else:
            cv2.rectangle(img, (8, y), (56, y + 48), GREY, 1)
            put(img, label, (62, y + 16), 0.4, WHITE)
            put(img, "not available" if lab is None else "pending", (62, y + 36), 0.38, GREY)
    vols = ", ".join(f"{k} {recipe[k + '_ml']} ml" for k in ("t100", "t200", "t400") if recipe and recipe[k + "_ml"])
    return np.concatenate([header(f"vial_fill_video_5   frame {f}   t = {f / FPS:.1f} s", f"PLC recipe {r['recipe_id']}: {vols}"), img])


def cap_seal_panel(img, link, det, f, exits):
    lx0, ly0, lx1, ly1 = LANE
    cv2.rectangle(img, (int(lx0), ly0), (int(lx1), ly1 - 1), CYAN, 1)
    cv2.line(img, (int(lx0), ENTRY_Y), (int(lx1), ENTRY_Y), CYAN, 1)
    put(img, "entry y=436", (int(lx1) + 4, ENTRY_Y + 4), 0.38, CYAN)
    d = det.get(f)
    linked_ids = set(link["fragment_ids"]) if link else set()
    seen_linked = False
    if d:
        for b in d["boxes_xyxy"]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREY, 1)
        for t in d["tracks"]:
            x, y, w, h = [int(v) for v in t["bbox_xywh"]]
            if t["id"] in linked_ids:
                seen_linked = True
                cv2.circle(img, (x + w // 2, y + h // 2), 26, YELLOW, 3)
                cv2.rectangle(img, (x, y), (x + w, y + h), YELLOW, 2)
                ly = y - 34 if y + h // 2 > PANEL_H - 50 else y + h // 2 + 5   # keep clear of the burned-in timestamp
                put(img, f"{t['id']} = vial {link['vial']}", (x + w + 30, ly), 0.55, YELLOW, 2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 1)
                put(img, t["id"], (x + w + 4, y + h // 2 + 5), 0.42, GREEN, 1)
    recent = [e for e in exits if f - 20 <= e["frame"] <= f]
    if recent:
        cv2.rectangle(img, (0, 0), (PANEL_W - 1, PANEL_H - 1), YELLOW, 6)
        put(img, "EXIT " + str(recent[-1].get("track_id", "")), (int(lx1) + 10, 60), 0.9, YELLOW, 2)
    if link and link["exit_frame"] is not None and link["exit_frame"] <= f <= link["exit_frame"] + 20:
        cv2.rectangle(img, (0, 0), (PANEL_W - 1, PANEL_H - 1), YELLOW, 6)
        put(img, f"EXIT {link['camera_id']} (vial {link['vial']})", (int(lx1) + 10, 60), 0.9, YELLOW, 2)
    if not link or link["camera_id"] is None:
        sub, col = f"vial {link['vial'] if link else '?'}: no top-down ID linked (no lane entry within 90 s)", GREY
    elif f < link["birth_frame"]:
        sub, col = f"linked ID {link['camera_id']} not born yet (birth at frame {link['birth_frame']}, {(link['birth_frame'] - f) / FPS:.1f} s ahead)", GREY
    elif seen_linked:
        sub, col = f"linked ID {link['camera_id']} = vial {link['vial']} (born {link['birth_frame']}, exit {link['exit_frame']})", YELLOW
    else:
        sub, col = f"linked ID {link['camera_id']} not visible this frame (fragments {'+'.join(link['fragment_ids'])})", GREY
    return np.concatenate([header(f"cap_seal_video_5   frame {f}   top-down tracker", sub, col), img])


def tank_panel(img, r, f):
    return np.concatenate([header(f"tank_level_video_5   frame {f}   t = {f / FPS:.1f} s", "tanks T100 / T200 / T400 feed the shared dispensing path"), img])


def render_clip(dataset, r, link, recipe, det, exits, start, end, out_mp4, out_jpg, jpg_frame):
    caps = {c: cv2.VideoCapture(str(dataset / c / "videos" / f"{c}_video_5.mp4")) for c in CAMS}
    for c in caps.values():
        c.set(cv2.CAP_PROP_POS_FRAMES, start)
    wr = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (3 * PANEL_W + 16, PANEL_H + HDR))
    n = 0
    for f in range(start, end + 1):
        imgs = {}
        for c, cap in caps.items():
            ok, im = cap.read()
            if not ok:
                break
            imgs[c] = im
        if len(imgs) < 3:
            break
        panels = [tank_panel(imgs["tank_level"], r, f), vial_fill_panel(imgs["vial_fill"], r, recipe, f),
                  cap_seal_panel(imgs["cap_seal"], link, det, f, exits)]
        sep = np.full((PANEL_H + HDR, 8, 3), 60, np.uint8)
        frame = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
        wr.write(frame)
        n += 1
        if f == jpg_frame:
            cv2.imwrite(str(out_jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    wr.release()
    for c in caps.values():
        c.release()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--changeover-vials", type=int, nargs="*", default=[7, 13, 16, 22],
                    help="position-1 vials flagged by the rule to render (default: four with a linked camera ID, "
                         "spanning dE 4.5 to 41.7 and including vial 22, whose known-colour check passes)")
    ap.add_argument("--keep-vial", type=int, default=17, help="a kept vial to render (default 17, repeat 2 of recipe 10)")
    ap.add_argument("--links-only", action="store_true")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="csi-vial-fill-dataset folder (default: fluidmix.paths data root, i.e. FLUIDMIX_DATA)")
    ap.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR,
                    help="folder with records.json and topdown/ (default: <repo>/outputs/carryover-video5)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where links.json and demo/ go (default: the records directory)")
    a = ap.parse_args()
    OUT = a.out or a.records_dir
    OUT.mkdir(parents=True, exist_ok=True)

    rec = json.load(open(a.records_dir / "records.json"))
    vials = {r["vial"]: r for r in rec["vials"]}
    det, ids = load_tracks(a.records_dir / "topdown/detections.jsonl")
    summary = json.load(open(a.records_dir / "topdown/summary.json"))
    # exit events: the tracker's own exit line (y=22) never fires on the real videos; the run script records
    # reference-line crossings (y=100, the scorer's definition) under line_crossings; both are flashed if they
    # fall inside a clip's frame range (on video 5 the crossings come after the clips end)
    exits = tracker_exits(summary) + [dict(track_id=c["track_id"], frame=c["frame"]) for c in summary.get("line_crossings", [])]
    events = entry_events(ids)
    links = link_vials(rec["vials"], events)
    # exit frame = first reference-line crossing (y=100) of any of the vial's fragment ids, from the run script
    cross = {}
    for c in summary.get("line_crossings", []): cross.setdefault(c["track_id"], c["frame"])
    for l in links:
        xs = [cross[i] for i in l["fragment_ids"] if i in cross]
        if xs and l["exit_frame"] is None: l["exit_frame"] = min(xs)
    n_linked = print_links(links)
    json.dump({"video_id": 5, "fps": FPS,
               "rule": {"entry_centre_y_gt": ENTRY_Y, "min_delay_frames": MIN_DELAY, "max_delay_frames": MAX_DELAY,
                        "flicker_min_frames": FLICKER_FRAMES, "fragment_gap_frames": FRAGMENT_GAP,
                        "assignment": "greedy in vial order: first unassigned entry event born in [fill_end+30, fill_end+2700]"},
               "n_tracks": len(ids), "n_entry_events": len(events), "n_vials": len(links), "n_linked": n_linked,
               "n_tracker_exits": len(exits),
               "n_line_crossings": len(summary.get("line_crossings", [])),
               "caveats": ["exit_frame is the first reference-line crossing (y=100, the scorer's definition) recorded by the run script for any "
                           "of the vial's fragment ids; the tracker's own exit line (y=22) never fires on this video. last_seen_frame is the last "
                           "frame the primary ID was logged.",
                           "Entry events (%d) are fewer than vials (%d): the tracker missed some entries, so the greedy chain can be "
                           "shifted by one vial after a miss. delay_s between fill end and lane birth ranges widely; no label verifies "
                           "the links." % (len(events), len(links))],
               "entry_events": events, "links": links}, open(OUT / "links.json", "w"), indent=1)
    print("wrote", OUT / "links.json")
    if a.links_only:
        return

    dataset = default_dataset(a.dataset)
    for c in CAMS:
        require(dataset / c / "videos" / f"{c}_video_5.mp4", f"{c} video 5")
    recipes = read_recipes()
    demo = OUT / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    link_by_vial = {l["vial"]: l for l in links}
    flagged = [v for v in a.changeover_vials if vials[v]["recipe_changed"] == 1 and vials[v]["decision"] == "discard"][:4]
    for v in a.changeover_vials:
        if v not in flagged:
            print(f"skipping vial {v}: not a flagged changeover vial")
    jobs = [(v, f"changeover_vial{v:02d}_recipe{vials[v]['prev_recipe_id']}to{vials[v]['recipe_id']}_discard") for v in flagged]
    kv = a.keep_vial
    if vials[kv]["decision"] == "keep":
        jobs.append((kv, f"keep_vial{kv:02d}_recipe{vials[kv]['recipe_id']}_keep"))
    else:
        print(f"vial {kv} is not a kept vial; no keep clip")
    for v, stem in jobs:
        r, link = vials[v], link_by_vial[v]
        start, end = r["fill_end_frame"] - 150, r["fill_end_frame"] + 600
        if link["birth_frame"] is not None:
            end = max(end, link["birth_frame"] + 150)
        jpg_frame = r["window"][1] if r["window"] else r["fill_end_frame"] + 60
        n = render_clip(dataset, r, link, recipes.get(v), det, exits, start, end, demo / f"{stem}.mp4", demo / f"{stem}.jpg", jpg_frame)
        print(f"vial {v}: {stem}.mp4 frames {start}-{end} ({n} written, {n / FPS:.1f} s), jpg at frame {jpg_frame}, "
              f"camera id {link['camera_id']} born {link['birth_frame']}")


if __name__ == "__main__":
    main()
