# Five-minute video: script and shot list

Target 5:00. Narration is about 810 words: about 5:20 at 150 words a minute, so read briskly and use the trim guidance below if a take runs long.
Each block names the shot and where the asset is. Record narration over screen captures; no slides are
needed beyond the three title cards.

Assets on the DGX Spark (not in git): segmenter demo clips in
`/home/hackathon5/outputs/topdown-01a17f15-20261029/iteration-20261031-liam/demo/` (`demo_v4.mp4`, 60 s,
three panels: raw lane, the team's YOLO baseline, ours; `demo_v5_exit.mp4`, 43 s). Carryover clips in
`/home/hackathon5/liamreporoot/topdown-results-20261101/outputs/carryover-video5/demo/`: three-camera
changeover clips (`changeover_vial22_recipe11to12_discard.mp4` and others, 25 to 44 s) and lane-pass clips
(`lanepass_vial25_discard.mp4`, `lanepass_vial17_keep.mp4`). Figures and the scoreboard page are in
`docs/assets/topdown/`; the attribution figure is `docs/assets/carryover/attribution.png`. Regenerate any
clip with `scripts/topdown/demo_overlay.py VIDEO START END OUT.mp4`, a live run with
`scripts/topdown/run_topdown_segmenter.py`, or the README "Run it" chain for the carryover clips
(see [docs/carryover-video5.md](../carryover-video5.md)).

---

**0:00 to 0:30, the problem.** *Shot: the real top-down camera, raw, then the tank-level camera.*

> This is Station 3 on the UWM CSI line. Three tanks feed one nozzle, and every vial gets a recipe. When
> the recipe changes, residue in that shared path tints the first vials of the new batch. The line's colour
> check can say a vial looks wrong. It cannot say which recipe the vial belongs to, which vial came before
> it, or that the next recipe will be contaminated. So the plant scraps, flushes by hand, and keeps no
> record. We turn the three cameras into evidence.

**0:30 to 0:55, the approach.** *Shot: pipeline diagram (README table rendered), then Isaac Sim twin renders
(docs/assets/topdown/real_vs_synth.jpg, bottom row).*

> The pipeline is fail-closed: it tracks each vial from the top-down camera, measures the dispense and the
> vial colour, compares that colour with the settled colour of the vial's own recipe, and returns keep,
> discard or insufficient evidence with the measurements attached. Every part we trained was trained on an
> Isaac Sim twin of the station, never on line footage; the only pretrained model is SAM 2, used as is.

**0:55 to 1:55, the measured result.** *Shot: demo_v4.mp4 full screen for 20 s (no narration over the first
5 s, let the three panels run), then the scoreboard's official-scores bars.*

> Left, the real camera; middle, the team's YOLO baseline; right, our U-Net, both trained only on twin
> renders. We scored both on all five real recordings, a hundred and five thousand frames, with a frozen
> harness and a floor from synthetic validation. No real frame was used for
> training, checkpoint selection or the threshold, but all five videos were scored while we searched for
> the augmentation recipe: exposed validation, not a held-out test. Ours is ahead on
> per-frame detection, precision point nine five seven against point nine four five, recall point nine
> two against point eight seven three, and behind on track identity coverage, point six five against
> point seven eight. Liam, who also ran the iteration, reviewed the labels on videos two and four, one
> frame per run of disagreement plus a spot check: a hundred forty-eight items, one missing cap found.
> Videos one, three and five remain unreviewed drafts.

**1:55 to 2:25, how it got there.** *Shot: real41_lane.jpg (model firing on fixtures), then aug53.jpg
(augmented training frames), then fp48_band.jpg.*

> It did not start there. The first synthetic model collapsed on the real lane, precision nine percent,
> because it had never seen holder fixtures, cables or lit panels. Every fix was synthetic: structure and
> drawn cables pasted into the lane as background, caps pasted onto dark squares shaped like the holder,
> and a floor from synthetic validation, its rule written after that collapse was analysed and before any
> floored run was scored on real video. Twenty-five runs, each changing one thing, each written down.

**2:25 to 2:45, the second camera.** *Shot: dispense-stream segmentation from docs/stream.md assets or the
vial_fill video with burst markers.*

> From the fill camera we measure the dispense itself. On video five, burst count matches the recipe in
> twenty-eight of twenty-nine vials, and the dispense order is recovered from timing in twenty-six of
> twenty-eight. Stream colour does not identify the tank; timing does.

**2:45 to 3:40, the decision, end to end.** *Shot: `changeover_vial22_recipe11to12_discard.mp4` full screen
(let the card flip from PENDING to DISCARD, about 8 s in), then `lanepass_vial25_discard.mp4` from 5 s
before its crossing, then `docs/assets/carryover/attribution.png`.*

> The whole chain on one real recording, three cameras in lockstep. Left, the tanks. Middle, the fill
> station: the recipe, the measured colour, the recipe's settled colour, and a fixed rule that turns the
> difference into keep or discard. Right, the top-down tracker giving that vial its camera ID: twenty-seven
> of thirty decisions carry one, unverified by any label, eighteen followed to the line, like this one. The first vial after every changeover sits a median
> twenty-three colour units from its recipe's settled colour; the second vial, two point three. That error points at
> the previous recipe, cosine point nine seven: a mix of the vial's own recipe and the one before it,
> larger for larger recipe changes in six of eight changeovers, gone two vials later. Carryover, measured
> and attributed to its cause. A stand-in for the line's own colour check passes this vial, because its
> wrong colour happens to be a known one.

**3:40 to 4:05, where it still fails.** *Shot: dropout53.jpg, then the exits row of the scoreboard table,
then the position-2 row of the carryover table.*

> Honest limits. The detector misses in short bursts, which splits tracks three times as often as the
> baseline and costs track identity and reference-line exits, twenty-nine of forty-nine against
> thirty-five. Epochs of the same run swing between recall point four three and point nine two; ours was
> a fortunate one, so we also report the reproducible variant. The carryover threshold fell to its
> hand-chosen floor and flags half of the later vials too: an in-sample demonstration, not a validation.
> The operator message is a template; an optional Cosmos Reason client exists but is not called.

**4:05 to 4:40, why it matters elsewhere.** *Shot: title card with three lines, then the twin render beside
the real frame.*

> Three things transfer to any plant with a changeover problem. One: neither detector needed line footage,
> so a line can be instrumented before anyone labels it. Two: carryover is a sequencing problem, and vial
> identity turns a colour reading into a flush decision with a name on it. Three: fail closed. Every flag
> carries its evidence or abstains, which is what makes an operator trust it.

**4:40 to 5:00, close.** *Shot: repository README on screen.*

> The code, the protocols and sha256 receipts, the harness reports, the review decisions and the carryover
> scripts are in the repository, and the carryover numbers reproduce from a clean clone. The weights, the
> frozen scorer, the draft labels and the clips are on the team's Spark, named by path and hash. Team
> Mixture Of Experts.

---

## Recording checklist

- Screen-record the demo clips at full resolution; keep the first 5 s of demo_v4 silent. The carryover
  clip is 1936 x 550; letterbox it rather than crop the side panels.
- Read the numbers as written; they match the README tables and docs/carryover-video5.md exactly.
- Narration is about 810 words. If a take runs long, trim the 0:00 and 4:05 blocks first; keep the
  exposed-validation sentence, the coverage line, the label-review sentence and the close.
- Export at 1080p; the five-minute limit is approximate per the organizers.
