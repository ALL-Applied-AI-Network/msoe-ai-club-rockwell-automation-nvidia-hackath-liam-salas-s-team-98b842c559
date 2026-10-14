---
name: pre-push-review
description: Use before every git push or PR on this repo. Runs tests and an adversarial review of the diff against the project's hard facts.
---
# Pre-push review

## Why
The team pushes code written with AI agents. An adversarial check catches wrong physics,
data leaks and made-up numbers before teammates build on them.

## Steps
1. Run the tests: `python tests/test_absorbance.py` (and `pytest` if installed). All must pass.
2. Review the diff (`git diff origin/main...HEAD`) as a critic, not an author. Look for:
   - Color math done in RGB instead of absorbance, or sRGB values used without `srgb_to_linear()`.
   - Negative absorbance clipped or hidden.
   - Real video used for training, or calibration vials reused as test vials.
   - Timing that ignores the 1.216x video fix (skill `video-timing`).
   - Recipes that break line rules: >80 mL per vial, >50 mL from T-200, any T-300.
   - Numbers in docs or PR text with no source (measured / read in a file / guess).
   - Large files: videos, images, zips, model weights, USD files.
3. If you use Claude Code, `/code-review` on the diff is a good extra pass.
4. Fix what you find, or write in the PR why it stays.
5. Write the PR text with the skill `project-writing`.
