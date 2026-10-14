# Fluid mixing carryover: agent instructions

Team project for the MSOE AI Club x Rockwell Automation x NVIDIA Innovation Lab (Fall 2026),
Dataset 2 (Fluid Mixing). Read this file first. Details live in `docs/`.

## The problem we solve
After a recipe change, fluid left in the shared mixing path tints the **first vial** of the
new batch. Vials 2 and 3 are clean. We detect that vial, say *why* (carryover, with a %),
and forecast it before the fill. See `docs/plan.md`.

## How to write
- Simple English, close to ASD-STE100: short sentences, one idea each, active voice.
- Team members are students. Explain the *why*. Define jargon on first use.
- **Never invent facts.** If you do not know, write "I do not know" or "not checked".
  Mark a claim as *measured*, *read in a file* or *guess*.
- Be critical. Say when an idea is weak, and why.

## Hard facts (do not "fix" these away)
- **Videos are 1.216x fast.** Real ~24.7 fps, tagged 30 fps. 1 video second = 1.216 real
  seconds. The original still labels (`video5_frame_labels.csv`) are wrong for ~95% of
  filling stills. Use the corrected file. See `docs/data.md` and skill `video-timing`.
- **Train on twin images only.** Real data is for testing and calibration.
- **Line rules:** max 80 mL per vial, max 50 mL from T-200, T-300 is broken,
  no recipe change without the plant manager.
- **Tanks:** T-100 clear, T-200 orange, T-400 blue.
- **Baselines to beat:** (1) the Teledyne color check, (2) the rule "always flag the first
  vial after a change" (8 of 10 in video 5). Report both. See `docs/evaluation.md`.
- **Color math is in absorbance, not RGB.** Dyes mix by Beer-Lambert. See `docs/absorbance.md`.

## Layout
`README.md` is the organizers' event README. Replace it with our project README before the finals (their rule).

```
src/fluidmix/absorbance.py   Beer-Lambert mixing, carryover fit, tank calibration
scripts/                     data fixes, color measure, twin variants, fits
tests/                       unit tests (python tests/test_absorbance.py)
data/                        small CSVs only (labels, measured colors). No videos.
docs/                        plan, data, twin, absorbance, evaluation, glossary
.claude/skills/              repeatable workflows for agents
```

## Data you need but is not in git
- Real dataset (~710 MB) and twin (~213 MB): public Drive folder linked from
  https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026 . Put them in `external/`
  (git-ignored). Never commit videos, images or zips.

## Commands
```bash
pip install numpy pillow usd-core pytest
python tests/test_absorbance.py              # unit tests
python scripts/fit_absorbance_video5.py      # fit tank absorbance on real colors
```

## Deadlines
Team plan 2026-10-14. Final 2026-11-01. Finals 2026-11-05.
