# Five-minute video: script and shot list

Target 5:00. Narration is written to be read at a natural pace (about 140 words a minute). Each block
names the shot and where the asset is. Record narration over screen captures; no slides are needed
beyond the three title cards.

Assets on the DGX Spark (not in git): demo clips in
`/home/hackathon5/outputs/topdown-01a17f15-20261029/iteration-20261031-liam/demo/` (`demo_v4.mp4`,
60 s, three panels: raw lane, YOLO baseline, ours; `demo_v5_exit.mp4`, 40 s, includes a reference-line
exit). End-to-end carryover clips (three panels: tank level, fill station with the decision card,
top-down tracker) in `/home/hackathon5/liamreporoot/topdown-results-20261101/outputs/carryover-video5/demo/`
(`changeover_vial07_recipe6to7_discard.mp4`, `changeover_vial22_recipe11to12_discard.mp4`,
`keep_vial17_recipe10_keep.mp4` and two more; 25 to 44 s each at 30 fps). Figures and the scoreboard page
are in `docs/assets/topdown/`. Regenerate any clip with `scripts/topdown/demo_overlay.py VIDEO START END OUT.mp4`
or a live run with `scripts/topdown/run_topdown_segmenter.py`; regenerate the carryover clips with
`scripts/carryover_video5.py` then `scripts/carryover_demo.py` (see [docs/carryover-video5.md](../carryover-video5.md)).

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
> measures the dispense bursts and the vial colour, predicts the expected colour from the recipe, and
> returns keep, discard, or insufficient evidence, with the measurements attached. The learned parts are
> trained on a digital twin of the station in Isaac Sim, not on line footage. That choice is the point of
> this video: can a model that has never seen the real line work on it?

**1:05 to 2:15, the measured result.** *Shot: demo_v4.mp4 full screen for 25 s (no narration over the first
5 s, let the three panels run), then the scoreboard's official-scores bars.*

> On the left, the real camera. In the middle, a YOLO detector trained on these very videos. On the right,
> our U-Net, trained only on synthetic renders. Both find the caps and both carry camera IDs up the lane.
> We scored both on all five real recordings, a hundred and five thousand frames, with a frozen harness and
> a threshold fixed on synthetic data before any real frame was scored. The synthetic-only model is ahead
> on both axes: precision point nine five seven against point nine four five, recall point nine two
> against point eight seven three. We then had a teammate hand-review every frame on two of the videos
> where the model and the labels disagreed. One hundred forty-seven decisions, one label error. The numbers
> did not move.

**2:15 to 2:50, how it got there.** *Shot: real41_lane.jpg (model firing on fixtures), then aug53.jpg
(augmented training frames), then fp48_band.jpg.*

> It did not start there. The first synthetic model collapsed on the real lane, precision nine percent,
> because it had never seen holder fixtures, cables, or lit panels. Every fix was synthetic: we pasted
> structure into the lane as background, drew cables and panels, and pasted caps onto that structure and
> onto dark squares shaped like the station holder. A confidence floor derived from synthetic validation
> removed the rest. Twenty-six training runs, each changing one thing, each written down.

**2:50 to 3:15, the second camera.** *Shot: dispense-stream segmentation from docs/stream.md assets or the
vial_fill video with burst markers.*

> From the fill camera we measure the dispense itself. Burst count matches the recipe in twenty-eight of
> twenty-nine vials and the dispense order is recovered in twenty-six of twenty-eight. The colour of the
> stream does not identify the tank, so tank identity has to come from timing.

**3:15 to 3:55, the decision, end to end.** *Shot: `changeover_vial22_recipe11to12_discard.mp4` full screen
(let the card flip from PENDING to DISCARD, about 8 s in, when the measurement window closes), then
`outputs/carryover-video5/attribution.png` (the two charts).*

> Here is the whole chain on one real recording, three cameras in lockstep. Left, the tanks. Middle, the
> fill station: the programmed recipe, the measured colour, the settled colour of that recipe, and a fixed
> rule that turns the difference into keep or discard with the evidence written underneath. Right, the
> top-down tracker giving that vial its camera ID; twenty-seven of thirty decisions carry one. Now the
> physics. The first vial after every changeover sits a median twenty-three colour units from its recipe's
> settled colour; the second vial, two. And the direction of that error points at the previous recipe,
> cosine point nine seven. The first vial is a mix of its own recipe and the one before it, the error
> grows with how different the two recipes are, and it is gone two vials later. That is carryover,
> measured, and attributed to its cause. The line's own colour check misses this vial, because its wrong
> colour happens to be a known one.

**3:55 to 4:15, where it still fails.** *Shot: dropout53.jpg, then the exits row of the scoreboard table,
then the position-2 row of the carryover table.*

> Honest limits. The detector misses in short bursts, which splits tracks three times as often as the
> baseline and costs reference-line exits: twenty-nine of forty-nine against thirty-five. The carryover
> rule's threshold fell to its hand-chosen floor, so it flags half of the later vials too, and that floor
> matches a statistic from the same video: an in-sample demonstration on one recording, not a validation.
> The Beer-Lambert fit returns negative absorbance and is not used. The operator message is a template;
> the Cosmos call is wired but has not run live.

**4:15 to 4:45, why it matters elsewhere.** *Shot: title card with three lines, then the twin render beside
the real frame.*

> Three things transfer to any plant with a changeover problem. One: the detector came from the twin, so a
> line can be instrumented before anyone labels footage. Two: carryover is a sequencing problem, and vial
> identity is what turns a colour reading into a flush decision with a name on it. Three: fail closed.
> Every flag carries its evidence or abstains, which is what makes an operator trust it.

**4:45 to 5:00, close.** *Shot: repository README on screen.*

> Everything is in the repository: the models, the harness, the twenty-six runs, the review decisions, the
> carryover run on video five, and the scripts that reproduce every number in this video. Team Mixture Of
> Experts.

---

## Recording checklist

- Screen-record the demo clips at full resolution; keep the first 5 s of demo_v4 silent. The carryover
  clip is 1936 x 550; letterbox it rather than crop the side panels.
- Read the numbers as written; they match the README tables and docs/carryover-video5.md exactly.
- Total narration is about 790 words, which runs long at 140 words a minute; trim the 2:15 and 4:15
  blocks first if a take exceeds 5:00.
- Export at 1080p; the five-minute limit is approximate per the organizers.
