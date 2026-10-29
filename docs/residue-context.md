# Residue-impact context packet (proposal for INN-62/63/64)

Question: before a vial starts a new recipe, will residue in the shared dispense path from the
previous vial visibly change it? This page defines **what to pull from the evidence store**,
**how the call is made**, and **what the reasoning model receives**. It is written against an
interface so it can plug into the INN-62 store after the pipeline's clean push. Status: tested
only on a synthetic history (`scripts/residue_packet.py`); not connected to the store; no real
or twin data.

Code: `src/fluidmix/residue_context.py`. Demo: `python scripts/residue_packet.py`.

## Rules it follows

From the pipeline-run rules (Linear, "Agent operating rules: pipeline run 2026-10-28"):

- **Inputs:** the programmed recipe sequence (tank, mL, order, vial sequence) and *measured*
  camera evidence only. No tank-colour presets, palettes, residue settings or labels.
- **Code decides, the model explains.** The risk is computed deterministically. The model gets
  the finished packet and may only explain it, citing evidence IDs; `check_explanation` rejects
  uncited or unknown IDs, numbers not in the packet, and a different risk. `template_explanation`
  is the fallback.
- **History is strictly earlier and from allowed roles.** The store is expected to filter. The
  builder checks again and raises `LeakageError` instead of trusting it.

## Why more than "the last tank dispensed"

Measured on video 5 ([stream](stream.md)): vial 8's clear T-100 dose came out visibly blue,
although the previous dose was also T-100. A one-dose-lag model fitted no better than the current
dose (25 vs 24 of 73 bursts). The shared path holds more than one dose, inferred at about 20 mL or
more. So the packet uses the previous **vial's measured colour** as the residue, plus a hold-up
volume measured from past switches, not a tank name. That residue choice is an assumption the
usefulness test below has to confirm.

## What is pulled

| Field | Source (INN-62 name) | Use |
|---|---|---|
| New recipe: doses in programmed order, total mL | programmed sequence | residue share = hold-up / total |
| Previous vial's programmed recipe | programmed sequence | context |
| Clean references for the new recipe: measured colour + empty-vial colour, strictly earlier | `clean_references` (`reference_bank`) | what "no impact" looks like |
| Previous vial: measured colour + empty-vial colour | `previous_vial` (`vial_history`) | the residue's colour |
| Past switches: previous, first and clean vial of each | `switches` (`switch_history`) | hold-up volume estimates |
| Image IDs (previous vial, clean references, past first vials), max 3 | store artifacts | images for the model |

Every measurement carries `evidence_id`, `cursor` (programmed-sequence position), `role`,
`vial_id`, `recipe_id`, `rgb`, `empty_rgb` and optional `image_id`. Integration means writing a
small adapter that implements `HistorySource` over the store; the field mapping still has to be
confirmed against its schema.

## How the call is made

All in absorbance, A = −log10(I / I0), using each measurement's own empty-vial I0, because
absorbance mixes linearly by volume fraction:

1. **Each past switch → a hold-up volume.** Fit f in A_first ≈ (1 − f)·A_clean + f·A_previous
   (closed form). f outside 0–1 is **excluded with a reason, never clipped**. Hold-up =
   f × that first vial's total mL.
2. **Scale to the new recipe.** Residue share = hold-up / new total mL, at the 25th, 50th and
   75th percentiles of the hold-ups.
3. **Predict the colour shift.** Mix the previous vial's absorbance into the clean reference at
   each share; CIE76 ΔE against the clean colour.
4. **Decide** against a frozen threshold:
   - `likely_impact`: even the low estimate exceeds the threshold, or the low hold-up exceeds
     the whole new volume;
   - `unlikely`: even the high estimate is within it;
   - `uncertain`: the threshold falls inside the range;
   - `insufficient_evidence`: fewer than 2 clean references, no measured previous vial, or fewer
     than 3 usable switches. The reasons say which.

**The threshold default (2.0, the store's measured-regime tolerance) is a placeholder.** It must
be frozen on validation before any test use (`Config.config_version` records this).

## Durability

- `schema_version` and `config_version` are written into every packet.
- Inputs are sorted by evidence ID and numbers rounded before hashing, so the same history gives
  the same `packet_sha256` whatever order the store returns it in (tested).
- Records after the target cannot change the packet (tested), and a source that leaks them is
  rejected (tested).

## Worked example (synthetic)

Your example recipe: 20 mL T-100, 10 mL T-200, 5 mL T-400, dispensed T-200 10 → T-400 5 →
T-100 20 (35 mL). The synthetic line holds 20 mL.

| Previous vial | Recovered hold-up | Residue share | Predicted ΔE | Call |
|---|---|---|---|---|
| a blue recipe | 19.8–20.3 mL | 0.57–0.58 | 23.7–24.3 | `likely_impact` |
| the same recipe | 19.8–20.3 mL | 0.57–0.58 | 0.3 | `unlikely` |

A 20 mL hold-up is more than half of a 35 mL recipe, so small recipes after a different colour
are the risky case. These numbers only show the protocol works on data built to fit its model.

## Tests to run once connected to the store

1. **Leakage:** a query at cursor t never returns anything at or after t; adding a future record
   changes nothing. (Done here on the synthetic source; repeat on the store.)
2. **Repeatability:** the same query on the same store snapshot gives the same `packet_sha256`.
3. **Coverage per test run:** the share of switches whose packet is not `insufficient_evidence`.
4. **Usefulness:** on validation, does the predicted ΔE rank the measured first-vial ΔE better
   than a baseline that ignores history (for example, "same recipe or not")? If not, revisit the
   residue assumption (the previous vial's mix) before trusting the call.

## Limits

- The residue is modelled as the previous vial's mix and the hold-up as one volume that mixes
  evenly. Video 5 shows the real path is messier.
- Camera channels only approximate a spectrum, and the empty-vial reference is air, not water,
  so this is not the calibrated absorbance in [absorbance](absorbance.md).
- Nothing here is plant evidence or an instruction to change recipes or flush the line.
