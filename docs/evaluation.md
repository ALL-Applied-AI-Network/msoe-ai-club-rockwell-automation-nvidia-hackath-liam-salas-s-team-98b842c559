# Evaluation

## Test sets
- Real video 5, leave-one-recipe-out. 10 changeovers only: report counts, not just %.
- Bench test (food coloring) for new colors.
- Live test at CSI if allowed.
Always use corrected timing (docs/data.md).

## Report separately
- False alarms (clean vial flagged).
- Missed defects (tinted vial passed).
- Correct cause (carryover vs other).
- Carryover % error, if ground truth exists.

## Baselines
| Baseline | What it does | Video 5 |
|---|---|---|
| Teledyne color check | Passes any color in its database | not measured |
| "Flag first vial after a change" | Flags every first vial | 8/10 caught, 2 false alarms |

Our value over the rule: say how bad, skip clean changeovers (e.g. R8->R9, dE 3.5),
work on new colors, and forecast before the fill.
