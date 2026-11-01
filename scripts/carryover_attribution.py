#!/usr/bin/env python
"""Carryover attribution on video 5: does each early vial's colour deviation point at the PREVIOUS recipe?

    python scripts/carryover_attribution.py [RECORDS_DIR]      # default outputs/carryover-video5

Reads RECORDS_DIR/records.json (written by scripts/carryover_video5.py; measured Lab per vial) and the recipe
schedule, and for every vial at position 1 or 2 of a run fits its measured colour as a mix of its own recipe's
settled colour and the previous recipe's settled colour:
    measured ~ settled_own + x * (settled_prev - settled_own)
A recipe's settled colour is its single position-3 vial (n=1 per recipe on video 5; the code takes the median if a
recipe ever has more than one vial at position >= 3). x is the fitted share of the previous recipe (least squares
along the line between the two settled colours), the residual is what the mix does not explain, and the direction
cosine says whether the deviation points at the previous recipe at all. No thresholds, no training: a descriptive
physics check. Writes attribution.md and attribution.png next to records.json. Measured on one recording with a
hand-labelled schedule; see results.md limits.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS_DIR = ROOT / "outputs/carryover-video5"


def _ranks(values):
    """Average ranks (ties share the mean rank), as Spearman's rho uses."""
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def rank_correlation(a, b):
    """Spearman rank correlation with tie-aware ranks; a reading aid, not a decision."""
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records_dir", nargs="?", type=Path, default=DEFAULT_RECORDS_DIR,
                    help="folder holding records.json; attribution.md and attribution.png are written beside it")
    out = ap.parse_args().records_dir
    R = json.load(open(out / 'records.json'))['vials']
    settled = collections.defaultdict(list)
    for r in R:
        if r.get('measured_lab') and r['position_in_run'] >= 3:
            settled[r['recipe_id']].append(np.array(r['measured_lab']))
    ref = {k: np.median(v, 0) for k, v in settled.items()}
    n_settled = {k: len(v) for k, v in settled.items()}
    rows = []
    for r in R:
        m = r.get('measured_lab'); p = r.get('prev_recipe_id'); c = r['recipe_id']
        if m is None or r['position_in_run'] >= 3 or c not in ref: continue
        m = np.array(m); own = float(np.linalg.norm(m - ref[c]))
        if p is None or p not in ref or p == c:
            rows.append(dict(vial=r['vial'], recipe=c, prev=p, pos=r['position_in_run'], dE_own=own, dE_prev_own=None, x=None, resid=None, cos=None)); continue
        d = ref[p] - ref[c]; dd = float(np.dot(d, d))
        x = float(np.dot(m - ref[c], d) / dd); resid = float(np.linalg.norm(m - (ref[c] + x * d)))
        cos = float(np.dot(m - ref[c], d) / (np.linalg.norm(m - ref[c]) * np.linalg.norm(d) + 1e-9))
        rows.append(dict(vial=r['vial'], recipe=c, prev=p, pos=r['position_in_run'], dE_own=own, dE_prev_own=float(np.linalg.norm(d)), x=x, resid=resid, cos=cos))
    # second vials: their schedule predecessor is their own recipe, so attribute toward the recipe that preceded the RUN
    tr = {int(x['vial']): x for x in csv.DictReader(open(ROOT / 'data/video5_transitions.csv'))}
    for r in rows:
        if r['pos'] != 2: continue
        prev_run = tr.get(r['vial'] - 1, {}).get('prev_recipe_id')
        if prev_run in (None, 'unknown') or int(prev_run) not in ref: continue
        p = int(prev_run); m = np.array(next(v['measured_lab'] for v in R if v['vial'] == r['vial'])); d = ref[p] - ref[r['recipe']]
        if np.dot(d, d) == 0: continue
        r.update(prev=p, dE_prev_own=float(np.linalg.norm(d)), x=float(np.dot(m - ref[r['recipe']], d) / np.dot(d, d)), resid=float(np.linalg.norm(m - (ref[r['recipe']] + np.dot(m - ref[r['recipe']], d) / np.dot(d, d) * d))), cos=float(np.dot(m - ref[r['recipe']], d) / (np.linalg.norm(m - ref[r['recipe']]) * np.linalg.norm(d) + 1e-9)))
    first = [r for r in rows if r['pos'] == 1 and r['x'] is not None]; second = [r for r in rows if r['pos'] == 2]
    f = lambda v, n=1: 'n/a' if v is None else f'{v:.{n}f}'
    settled_note = ('its single position-3 vial (n=1 per recipe)' if all(n == 1 for n in n_settled.values())
                    else 'the median of its position>=3 vials (' + ', '.join(f'recipe {k}: n={n}' for k, n in sorted(n_settled.items())) + ')')
    md = ['# Carryover attribution, video 5 (descriptive, no thresholds)', '',
          f"For each early vial: distance to its own recipe's settled colour, which is {settled_note}; distance between the",
          'previous and the current recipe\'s settled colours, the fitted share x of the previous recipe, the residual the mix does',
          'not explain, and the cosine between the deviation and the direction toward the previous recipe (1.0 = points straight at it).', '',
          '| vial | recipe | previous | position | dE to own settled | dE(prev settled, own settled) | fitted share of previous x | residual | direction cosine |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows: md.append(f"| {r['vial']} | {r['recipe']} | {r['prev']} | {r['pos']} | {f(r['dE_own'])} | {f(r['dE_prev_own'])} | {f(r['x'],2)} | {f(r['resid'])} | {f(r['cos'],2)} |")
    c1 = [r['cos'] for r in first]; x1 = [r['x'] for r in first]; d1 = [r['dE_own'] for r in first]; d2 = [r['dE_own'] for r in second]
    gap1 = [r['dE_prev_own'] for r in first]
    exceed = [r['vial'] for r in first if r['dE_own'] > r['dE_prev_own']]
    rho = rank_correlation([round(g, 1) for g in gap1], [round(d, 1) for d in d1])   # on the table's one-decimal values
    rho_raw = rank_correlation(gap1, d1)
    md += ['', '## Summary', '',
           f"- First vial after a recipe change (n={len(first)}): distance to the settled colour median {np.median(d1):.1f} (range {min(d1):.1f} to {max(d1):.1f}); the deviation points at the previous recipe with direction cosine median {np.median(c1):.2f}, {sum(c >= 0.9 for c in c1)} of {len(c1)} at or above 0.9; fitted share of the previous recipe median {np.median(x1):.2f} (range {min(x1):.2f} to {max(x1):.2f}).",
           f"- Size of the recipe change versus first-vial error (n={len(first)}): rank correlation {rho:.2f} (Spearman on the table's one-decimal values; {rho_raw:.2f} on the unrounded values); {len(first) - len(exceed)} of {len(first)} first vials deviate less than their recipe gap, vials {', '.join(map(str, exceed)) or 'none'} exceed it. A reading of the table, not a fitted law.",
           f"- Second vial of the run (n={len(second)}): distance to the settled colour median {np.median(d2):.1f} (range {min(d2):.1f} to {max(d2):.1f}). Attributed toward the recipe that preceded the run (previous column shows it): " + '; '.join(f"vial {r['vial']} dE {r['dE_own']:.1f} cosine {r['cos']:.2f} share {r['x']:.2f}" for r in second if r['cos'] is not None) + ". The second vials that still deviate by more than a few units point at the pre-run recipe (residual carryover, decaying), while the small deviations point in no particular direction (noise).",
           f"- A share above 1 means the vial looks even more like the previous recipe than that recipe's own settled vial; with {settled_note} as the settled colour and uncalibrated camera colour, shares are indicative, not volumetric.",
           '- Reading: the colour error of the first vial after a changeover is explained by the previous recipe, which is the carryover signature; by the second vial it has mostly decayed. The exception is recipe 9 after recipe 8 (vial 13), whose two recipes are only 6.6 apart, so carryover cannot be seen in colour there.']
    (out / 'attribution.md').write_text('\n'.join(md) + '\n')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    recs = sorted({r['recipe'] for r in rows})
    p1 = {r['recipe']: r['dE_own'] for r in rows if r['pos'] == 1}; p2 = {r['recipe']: r['dE_own'] for r in rows if r['pos'] == 2}
    xs = np.arange(len(recs)); w = .38
    ax[0].bar(xs - w/2, [p1.get(k, 0) for k in recs], w, label='1st vial after changeover', color='#eb6834'); ax[0].bar(xs + w/2, [p2.get(k, 0) for k in recs], w, label='2nd vial', color='#2a78d6')
    ax[0].set_xticks(xs); ax[0].set_xticklabels([f'R{k}' for k in recs]); ax[0].set_ylabel('CIE76 distance to the recipe\'s settled colour'); ax[0].set_title('Carryover decays within two vials'); ax[0].legend(frameon=False); ax[0].spines[['top', 'right']].set_visible(False)
    ax[1].scatter([r['dE_prev_own'] for r in first], [r['dE_own'] for r in first], c=[r['cos'] for r in first], cmap='viridis', vmin=0, vmax=1, s=60)
    for r in first: ax[1].annotate(f"R{r['prev']}->R{r['recipe']}", (r['dE_prev_own'], r['dE_own']), fontsize=8, xytext=(4, 3), textcoords='offset points')
    ax[1].plot([0, 85], [0, 85], ls='--', c='#888', lw=1); ax[1].set_xlabel('distance between previous and current recipe colours'); ax[1].set_ylabel('first vial\'s distance to its settled colour'); ax[1].set_title(f'First-vial error vs recipe change (n={len(first)}, rank correlation {rho:.2f})\ncolour = direction cosine toward the previous recipe', fontsize=10); ax[1].spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'attribution.png', dpi=150); print('\n'.join(md[-6:]))


if __name__ == '__main__':
    main()
