---
name: video-timing
description: Align video-5 still timestamps and frames before making labels or evaluation sets.
---
# Video timing

## Inputs
Source video, recipe labels and still index; see [data](../../../docs/data.md) for paths.

## Workflow
1. Run python scripts/check_data.py, then python scripts/fixlabels.py (data root: FLUIDMIX_DATA or external/).
2. Inspect ratio, in_fill count and coverage. Spot-check early, middle and late stills
   against video clocks and fill boundaries.
3. Measure another video's own clocks and ratio; do not reuse video-5 constants.

## Output and validation
Corrected labels and last-still map under external/, with counts and alignment evidence.
The embedded ratio (~1.2155 real seconds/video second) needs source-video confirmation.

## Limits
Original labels are untrusted. Report missing stills and uncertain boundaries.
All-video timing and historical 95% error claims remain unverified.
