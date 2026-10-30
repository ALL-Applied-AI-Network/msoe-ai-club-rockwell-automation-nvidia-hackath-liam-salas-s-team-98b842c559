"""INN-52: how stream and vial colour change *while* each dose pours (inspection only).

    python scripts/stream_sequence.py     # first: writes outputs/inn52/stream_frames.csv
    python scripts/burst_trace.py         # reads it, prints the checks
    python scripts/burst_trace.py --store # also rewrites data/video5_burst_trace.csv

stream_sequence.py summarises each burst by one median colour. That hides any change inside the
dose: liquid left in the shared dispensing tube comes out first, so a dose can start in the
previous tank's colour and only later turn into its own. Here each burst's flowing frames are
split into SEGMENTS equal parts in time, and per part the stream colour, the vial colour and the
liquid level are kept. Frames whose liquid level is at or above SUBMERGED_ROW are dropped: there
the vial's contents are behind the stream rows and tint the stream measurement.

Checks, on bursts whose count and tank match the recipe log (see stream_sequence.py):
  1) drift: a*b* distance between the first and last part, against a noise floor (odd vs even
     frames of the last part; adjacent frames are correlated, so the floor is a lower bound);
  2) start vs end: tank references are the median last-part colour of each tank's bursts in the
     other recipes (leave-one-recipe-out; no preset tank colours). Where the previous dose came
     from another tank, is the first part nearer the previous tank's reference and the last part
     nearer the current one? Each is a two-way choice, so guessing gets half;
  3) carryover without references: within a vial, is a dose's first part nearer the previous
     dose's last part than its own last part? If the shared tube pushes out the previous liquid
     first, it should be. Drift is also split by burst position: a vial's first burst follows a
     tank change too (the previous vial's last dose), so carryover alone predicts no difference;
  4) vial during the dose: does the clear-view level rise (row falls) within each burst?

Real video, inspection only. Segments, thresholds and regions were chosen on these vials, so
the counts describe the signal; they are not a held-out evaluation.
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fluidmix.paths import REPO, outputs_root
from stream_sequence import ROW1, hue_chroma, nearest_tank, tank_refs

SEGMENTS = 5                 # equal parts in time per burst
SUBMERGED_ROW = ROW1         # level at or above this row: vial liquid is behind the stream rows
MIN_PART_FRAMES = 3          # parts with fewer usable frames are left empty


def load_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def num(x):
    return float(x) if x not in ("", "nan", "None") else np.nan


def split_parts(n, parts=SEGMENTS):
    """Index arrays splitting range(n) into `parts` near-equal consecutive pieces."""
    return np.array_split(np.arange(n), parts)


def ab_distance(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def ab(part):
    return part["dA"], part["dB"]


def level_trend(rows):
    """Slope (rows per frame) of the level row over a burst; negative means the level rises."""
    rows = np.asarray(rows, float)
    ok = ~np.isnan(rows)
    if ok.sum() < 5:
        return np.nan
    return float(np.polyfit(np.flatnonzero(ok), rows[ok], 1)[0])


def build(frames, bursts, recipes):
    """Per-burst traces: usable frames, parts and context."""
    prev_last = {v["vial"]: (recipes[i - 1]["dispense_order"].split(">")[-1] if i else None)
                 for i, v in enumerate(recipes)}
    recipe_of = {v["vial"]: v["recipe_id"] for v in recipes}
    by_burst = {}
    for f in frames:
        if f["burst"] != "0" and f["flowing"] == "1":
            by_burst.setdefault((f["vial"], f["burst"]), []).append(f)
    traces, prev = [], {}
    for b in bursts:
        prev_tank = prev.get(b["vial"], prev_last[b["vial"]])
        prev[b["vial"]] = b["tank"] or None
        if not b["tank"]:
            continue
        rows = by_burst[(b["vial"], b["burst"])]
        level = np.array([num(r["level_row"]) for r in rows])
        keep = ~(level <= SUBMERGED_ROW)
        used = [r for r, k in zip(rows, keep) if k]
        parts = []
        for idx in split_parts(len(used)):
            sel = [used[i] for i in idx]
            if len(sel) < MIN_PART_FRAMES:
                parts.append(None)
                continue
            parts.append({c: float(np.median([num(r[c]) for r in sel])) for c in
                          ("dL", "dA", "dB", "vial_dL", "vial_dA", "vial_dB")}
                         | {"first": int(sel[0]["frame"]), "last": int(sel[-1]["frame"]), "n": len(sel),
                            "level_row": num(sel[-1]["level_row"]), "level_zone": sel[-1]["level_zone"]})
        traces.append({"vial": b["vial"], "recipe": recipe_of[b["vial"]], "burst": int(b["burst"]),
                       "tank": b["tank"], "prev": prev_tank, "frames": len(rows), "used": used,
                       "dropped": int((~keep).sum()), "parts": parts, "level": level})
    return traces


def end_refs(traces, exclude_recipe):
    return tank_refs([(t["tank"], t["parts"][-1]["dA"], t["parts"][-1]["dB"]) for t in traces
                      if t["recipe"] != exclude_recipe and t["parts"][-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="store_true", help="write data/video5_burst_trace.csv")
    a = ap.parse_args()
    out = outputs_root() / "inn52"
    if not (out / "stream_frames.csv").is_file():
        raise SystemExit(f"missing {out / 'stream_frames.csv'}; run scripts/stream_sequence.py first")
    frames = load_csv(out / "stream_frames.csv")
    if "flowing" not in frames[0]:
        raise SystemExit("stream_frames.csv has no per-frame vial columns; rerun scripts/stream_sequence.py")
    bursts = load_csv(out / "stream_bursts.csv")
    recipes = load_csv(REPO / "data/video5_vial_recipes.csv")
    traces = build(frames, bursts, recipes)
    whole = [t for t in traces if t["parts"][0] and t["parts"][-1]]
    print(f"{len(traces)} bursts with a matched tank; {len(whole)} with a usable first and last part; "
          f"{sum(t['dropped'] for t in traces)} of {sum(t['frames'] for t in traces)} flowing frames dropped "
          f"because the liquid level reached the stream rows")

    # 1) drift vs noise
    drift, noise = [], []
    for t in whole:
        first, last = t["parts"][0], t["parts"][-1]
        drift.append(ab_distance((first["dA"], first["dB"]), (last["dA"], last["dB"])))
        tail = [(num(r["dA"]), num(r["dB"])) for r in t["used"][-last["n"]:]]
        noise.append(ab_distance(np.median(tail[0::2], axis=0), np.median(tail[1::2], axis=0)))
    drift, noise = np.array(drift), np.array(noise)
    print(f"1) first-to-last a*b* drift: median {np.median(drift):.1f} (middle half "
          f"{np.percentile(drift, 25):.1f}-{np.percentile(drift, 75):.1f}); odd/even noise floor median "
          f"{np.median(noise):.2f}; drift > 3x that burst's floor in {int((drift > 3 * noise).sum())} of {len(whole)}")

    # 2) start nearer previous, end nearer current
    start_prev = end_cur = n2 = whole_cur = 0
    refs = end_refs(traces, None)
    spread = [ab_distance(ab(t["parts"][-1]), refs[t["tank"]]) for t in traces if t["parts"][-1]]
    separation = [ab_distance(refs[p], refs[q]) for i, p in enumerate(sorted(refs)) for q in sorted(refs)[i + 1:]]
    for t in whole:
        if not t["prev"] or t["prev"] == t["tank"]:
            continue
        refs = end_refs(traces, t["recipe"])
        if t["prev"] not in refs or t["tank"] not in refs:
            continue
        pair = {k: refs[k] for k in (t["prev"], t["tank"])}
        first, last = t["parts"][0], t["parts"][-1]
        start_prev += nearest_tank(first["dA"], first["dB"], pair) == t["prev"]
        end_cur += nearest_tank(last["dA"], last["dB"], pair) == t["tank"]
        mid = np.median([(num(r["dA"]), num(r["dB"])) for r in t["used"]], axis=0)
        whole_cur += nearest_tank(mid[0], mid[1], pair) == t["tank"]
        n2 += 1
    print(f"2) bursts after a different tank: {n2}. First part nearer the previous tank: {start_prev}; "
          f"last part nearer the current tank: {end_cur}; whole-burst median nearer the current tank: "
          f"{whole_cur} (guessing: {n2 / 2:.0f}). Last-part references, all recipes: "
          f"{min(separation):.1f}-{max(separation):.1f} apart; a burst's last part is a median "
          f"{np.median(spread):.1f} from its own tank's reference")

    # 3) carryover without references
    nearer_prev, pairs = 0, 0
    by_vial = {}
    for t in traces:
        by_vial.setdefault(t["vial"], []).append(t)
    for ts in by_vial.values():
        for before, t in zip(ts, ts[1:]):
            if before["parts"][-1] and t["parts"][0] and t["parts"][-1]:
                start_ab = ab(t["parts"][0])
                nearer_prev += ab_distance(start_ab, ab(before["parts"][-1])) < ab_distance(start_ab, ab(t["parts"][-1]))
                pairs += 1
    print(f"3) first part nearer the previous dose's last part than its own last part: {nearer_prev} of {pairs}")
    for k in sorted({t["burst"] for t in whole}):
        d = [x for t, x in zip(whole, drift) if t["burst"] == k]
        print(f"   burst {k} of its vial: median drift {np.median(d):.1f}, n={len(d)}")

    # 4) vial level during the dose
    slopes = []
    for t in traces:
        rows = [num(r["level_row"]) if r["level_zone"] == "below_holder" else np.nan for r in t["used"]]
        s = level_trend(rows)
        if not np.isnan(s):
            slopes.append((t["tank"], s))
    rising = sum(s < 0 for _, s in slopes)
    print(f"4) clear-view level within the burst: rises in {rising} of {len(slopes)} bursts "
          f"(median slope {np.median([s for _, s in slopes]):+.3f} rows per frame)")
    for tank in sorted({tk for tk, _ in slopes}):
        v = [s for tk, s in slopes if tk == tank]
        print(f"   {tank}: rises in {sum(s < 0 for s in v)} of {len(v)}, median {np.median(v):+.3f} rows per frame")

    rows = []
    for t in traces:
        for k, p in enumerate(t["parts"], 1):
            row = {"vial": t["vial"], "recipe_id": t["recipe"], "burst": t["burst"], "tank": t["tank"],
                   "previous_tank": t["prev"] or "", "part": k, "parts": SEGMENTS,
                   "burst_frames": t["frames"], "dropped_submerged": t["dropped"]}
            if p:
                hue, chroma = hue_chroma(p["dA"], p["dB"])
                vhue, vchroma = hue_chroma(p["vial_dA"], p["vial_dB"])
                row |= {"first_frame": p["first"], "last_frame": p["last"], "frames_used": p["n"],
                        "stream_dL": f"{p['dL']:.2f}", "stream_dA": f"{p['dA']:.2f}", "stream_dB": f"{p['dB']:.2f}",
                        "stream_hue_deg": f"{hue:.0f}", "stream_chroma": f"{chroma:.2f}",
                        "vial_dL": f"{p['vial_dL']:.2f}", "vial_dA": f"{p['vial_dA']:.2f}",
                        "vial_dB": f"{p['vial_dB']:.2f}", "vial_hue_deg": f"{vhue:.0f}",
                        "vial_chroma": f"{vchroma:.2f}",
                        "level_row": "" if np.isnan(p["level_row"]) else f"{p['level_row']:.0f}",
                        "level_zone": p["level_zone"]}
            rows.append(row)
    fields = list(max(rows, key=len))
    targets = [out / "burst_trace.csv"] + ([REPO / "data/video5_burst_trace.csv"] if a.store else [])
    for path in targets:
        with path.open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", restval="")
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {', '.join(map(str, targets))}")


if __name__ == "__main__":
    main()
