# Five-minute video: script and shot list

Target 5:00. Narration is written to be read at a natural pace (about 140 words a minute). Each block
names the shot and where the asset is. Record narration over screen captures; no slides are needed
beyond the three title cards.

Assets on the DGX Spark (not in git): demo clips in
`/home/hackathon5/outputs/topdown-01a17f15-20261029/iteration-20261031-liam/demo/` (`demo_v4.mp4`,
60 s, three panels: raw lane, YOLO baseline, ours; `demo_v5_exit.mp4`, 40 s, includes a reference-line
exit). End-to-end carryover clips (three panels: tank level, fill station with the decision card,
top-down tracker) and lane-pass clips in
`/home/hackathon5/liamreporoot/topdown-results-20261101/outputs/carryover-video5/demo/`
(`changeover_vial07_recipe6to7_discard.mp4`, `changeover_vial22_recipe11to12_discard.mp4`,
`keep_vial17_recipe10_keep.mp4`, `lanepass_vial25_discard.mp4` and more; 25 to 44 s each at 30 fps).
Figures and the scoreboard page are in `docs/assets/topdown/`. Regenerate any clip with
`scripts/topdown/demo_overlay.py VIDEO START END OUT.mp4` or a live run with
`scripts/topdown/run_topdown_segmenter.py`; regenerate the carryover clips with the README "Run it" chain
(`scripts/carryover_video5.py`, `scripts/carryover_demo.py`, `scripts/carryover_lane_clip.py 25 17`; see
[docs/carryover-video5.md](../carryover-video5.md)). The attribution figure is
`outputs/carryover-video5/attribution.png`, written by `scripts/carryover_attribution.py`.

---

**0:00 to 0:35, the problem.** *Shot: the real top-down camera, raw, then the tank-level camera.*

> This is Station 3 on the UWM CSI line. Three tanks feed one nozzle, and every vial gets a recipe. When
> the recipe changes, residue in that shared path tints the first vials of the new batch. The line's
> colour check can tell you a vial looks wrong. It cannot tell you which recipe that vial belongs to, which
> vial came before it, or that the next recipe will be contaminated. So the plant scraps, flushes by hand,
> and keeps no record. We set out to turn the three cameras into evidence: which vial, what it should have
> looked like, what it did look like, and what to do.

**0:35 to 1:05, the approach.** *Shot: pipeline diagram (README table rendered), then Isaac Sim twin renders
(docs/assets/topdown/real_vs_synth.jpg, bottom row).*

> The pipeline is fail-closed. It tracks each vial through the inspection lane from the top-down camera,
> measures the dispense bursts and the vial colour, compares that colour with the settled colour of the
> vial's own recipe, and returns keep, discard, or insufficient evidence, with the measurements attached.
> The learned parts are trained on a digital twin of the station in Isaac Sim, not on line footage. Can a
> model that has never seen the real line work on it?

**1:05 to 2:15, the measured result.** *Shot: demo_v4.mp4 full screen for 25 s (no narration over the first
5 s, let the three panels run), then the scoreboard's official-scores bars.*

> Left, the real camera; middle, the team's YOLO baseline; right, our U-Net. Both were trained only on
> synthetic twin renders. We scored both on all five real recordings, a hundred and five thousand frames,
> with a frozen harness and a confidence floor derived from synthetic validation only. No real frame was
> used for training, checkpoint selection or the threshold; but all five videos were scored while we
> searched for the augmentation recipe, so this is exposed-validation evidence, not a held-out test. Ours
> is ahead on per-frame detection, precision point nine five seven against point nine four five, recall
> point nine two against point eight seven three, and behind on track identity coverage, point six five
> against point seven eight. Liam, who also ran the iteration, reviewed the labels on videos two and four:
> one frame per run of disagreement plus a spot check every ninety frames, a hundred forty-seven of a
> hundred forty-eight items answered, one missing cap found. Videos one, three and five, which carry most
> of the score, remain unreviewed drafts.

**2:15 to 2:50, how it got there.** *Shot: real41_lane.jpg (model firing on fixtures), then aug53.jpg
(augmented training frames), then fp48_band.jpg.*

> It did not start there. The first synthetic model collapsed on the real lane, precision nine percent,
> because it had never seen holder fixtures, cables, or lit panels. Every fix was synthetic: structure
> pasted into the lane as background, drawn cables and panels, caps pasted onto that structure and onto
> dark squares shaped like the station holder. A confidence floor from synthetic validation removed the
> rest; its rule was written after that collapse was analysed and before any floored run was scored on
> real video. Twenty-five training runs, each changing one thing, each written down.

**2:50 to 3:15, the second camera.** *Shot: dispense-stream segmentation from docs/stream.md assets or the
vial_fill video with burst markers.*

> From the fill camera we measure the dispense itself. On video five, burst count matches the recipe in
> twenty-eight of the twenty-nine vials with a pre-pour frame, and the dispense order is recovered from
> timing in twenty-six of twenty-eight. The colour of the stream does not identify the tank, so tank
> identity has to come from timing.

**3:15 to 3:55, the decision, end to end.** *Shot: `changeover_vial22_recipe11to12_discard.mp4` full screen
(let the card flip from PENDING to DISCARD, about 8 s in, when the measurement window closes), then
`lanepass_vial25_discard.mp4` from about 5 s before its crossing (the flagged vial's camera ID reaching the
reference line), then `outputs/carryover-video5/attribution.png` (the two charts).*

> The whole chain on one real recording, three cameras in lockstep. Left, the tanks. Middle, the fill
> station: the programmed recipe, the measured colour, the settled colour of that recipe from its later
> vials, and a fixed rule that turns the difference into keep or discard. Right, the top-down tracker
> giving that vial its camera ID: twenty-seven of thirty decisions carry one, unverified, since no label
> checks the links, and eighteen of those have a recorded reference-line crossing, like this one. The
> first vial after every changeover sits a median twenty-three colour units from its recipe's settled
> colour; the second vial, two. That error points at the previous recipe, cosine point nine seven: a mix
> of the vial's own recipe and the one before it, larger for larger recipe changes in six of eight
> changeovers, gone two vials later. Carryover, measured and attributed to its cause. A stand-in for the
> line's own colour check passes this vial, because its wrong colour happens to be a known one.

**3:55 to 4:15, where it still fails.** *Shot: dropout53.jpg, then the exits row of the scoreboard table,
then the position-2 row of the carryover table.*

> Honest limits. The detector misses in short bursts, which splits tracks three times as often as the
> baseline and costs track identity and reference-line exits, twenty-nine of forty-nine against
> thirty-five. Epochs of the same run swing between recall point four three and point nine two; our
> selected epoch was a fortunate one, so we also report the reproducible variant. The carryover rule's
> threshold fell to its hand-chosen floor, so it flags half of the later vials too: an in-sample
> demonstration, not a validation. The Beer-Lambert fit is not used. The operator message is a
> deterministic template; an optional Cosmos Reason client exists but is not called by the pipeline and
> has not been run.

**4:15 to 4:45, why it matters elsewhere.** *Shot: title card with three lines, then the twin render beside
the real frame.*

> Three things transfer to any plant with a changeover problem. One: neither detector needed line footage;
> the difference was the augmentation recipe for the real world, the synthetic-derived floor and the
> reviewed evaluation, so a line can be instrumented before anyone labels footage. Two: carryover is a
> sequencing problem, and vial identity turns a colour reading into a flush decision with a name on it.
> Three: fail closed. Every flag carries its evidence or abstains, which is what makes an operator trust it.

**4:45 to 5:00, close.** *Shot: repository README on screen.*

> Everything except the data and the two 1.1 MB weights is in the repository: the code, the protocols and
> sha256 receipts, the harness reports, the review decisions, the carryover run and the scripts that
> reproduce every number; the weights are on the team's Spark. Team Mixture Of Experts.

---

## Recording checklist

- Screen-record the demo clips at full resolution; keep the first 5 s of demo_v4 silent. The carryover
  clip is 1936 x 550; letterbox it rather than crop the side panels.
- Read the numbers as written; they match the README tables and docs/carryover-video5.md exactly.
- Total narration is about 925 words, which runs about 6:30 at 140 words a minute; read briskly and trim
  the 0:00, 2:15 and 4:15 blocks first if a take exceeds 5:00. Do not trim the exposed-validation
  sentence, the track identity coverage line, the label-review sentence or the "Everything except" close.
- Export at 1080p; the five-minute limit is approximate per the organizers.
