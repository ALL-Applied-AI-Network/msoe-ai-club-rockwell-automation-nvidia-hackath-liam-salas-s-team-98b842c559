---
name: evaluate-vs-baseline
description: Use when scoring the detector or forecast on real data, or writing any result number for the team or judges.
---
# Evaluate against baselines

Read `docs/evaluation.md` first.

## Steps
1. Use corrected timing (skill `video-timing`).
2. Split leave-one-recipe-out. Never tune on the recipe you test.
3. Score our method AND both baselines on the same vials:
   - "Flag the first vial after every change"
   - Teledyne check (if we can get its output; otherwise say "not measured")
4. Report counts: false alarms, missed defects, correct cause, carryover % error.
5. With 10 changeovers, write counts like "7/9", not only percentages.
6. State the color method used and its known limits.
