# Contributor documentation

Start at [quickstart](quickstart.md) for CPU setup, checks and a synthetic smoke run.
Read [AGENTS.md](../AGENTS.md) for shared operating policy.

| Document | Read it when |
|---|---|
| [Proposal](proposal.md) | Checking team intent; source of truth since 2026-10-16 |
| [Plan](plan.md) | Checking current implementation and open decisions |
| [Architecture](architecture.md) | Connecting recipes, measurements, anomaly evidence and explanations |
| [Data](data.md) | Downloading inputs, tracing CSVs or aligning video |
| [Data inventory](data_inventory.md) | Counts, anomalies and missing labels in Dataset 2 |
| [Video timing](video_timing.md) | Mapping stills and labels to video frames |
| [Dispense stream](stream.md) | Reading tank timing or stream colour from the fill-station video (INN-52) |
| [SAM on the Spark](sam.md) | Setting up and running the SAM 2.1 segmentation check on the DGX Spark from a laptop |
| [Labels](labels.md) | Finding label sources, unknowns and which real frames to exclude from validation |
| [Absorbance](absorbance.md) | Using physics functions and judging calibration validity |
| [Evaluation](evaluation.md) | Freezing partitions, thresholds and baseline comparisons |
| [Twin](twin.md) | Making variants or assessing rendering limits |
| [Tank station](tank-station.md) | Building and rendering the optional four-tank USD assembly |
| [Operations](operations.md) | Configuring inputs, credentials, outputs and compute shutdown |
| [Skills](skills.md) | Choosing workflows and checking host discovery |
| [Glossary](glossary.md) | Looking up terminology |
| [Review evidence](review-inn-22-24.md) | Reviewing this batch and remaining acceptance checks |
| [Hackathon info](ground-truth-hackathon-info/hackathon-info.md) | Checking an organizer rule, deadline, judging criterion or dataset description. Large reference copies (~720 lines in 6 files): search with grep and read only the matching section |
| [Current acceptance audit](acceptance-inn-23-24.md) | Checking the Oct 16 host evidence, issue coverage and pending teammate trial |
| [Teammate receipt](teammate-quickstart.md) | Recording the required independent clean-checkout trial |
| [Draft feasibility protocol](feasibility-protocol.md) | Preparing INN-32 tolerances and an evidence-linked decision without claiming agreement |
| [INN-52 revised scope](inn-52-scope.md) | Reviewing the proposed INN-52 rewording before it goes to Linear |
| [Presentation](presentation/README.md) | Making slides, the video, the summary or a chart in the team design system |
| [Top-down cap segmenter](topdown-cap-segmenter.md) | The synthetic-only cap detector, its real-video scores against the YOLO baseline, failure modes and how to run it |
| [Top-down iteration log](topdown-iteration-log.md) | Every segmenter run from Run38 to dense64 with diagnostics; the evidence trail behind the scores |
| [Carryover on video 5](carryover-video5.md) | The end-to-end carryover decision on real video 5: measurement, fixed rule, attribution to the previous recipe, camera-ID links, demo clips, limits |
| [Submission](submission/one-page-summary.md) | The final one-page summary, [video script](submission/video-script.md) and [credits](submission/credits.md) |

Sources: [merged proposal PR #1](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/pull/1),
[draft scaffold PR #2](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/pull/2),
[Linear project](https://linear.app/innovation-lab-2026/project/mixture-of-experts-fluid-mixing-carryover-315d582ab3f1).
Keep detailed facts in one place and link to them. Label historical scaffold observations
as reported until rechecked; record input, command and limits for measurements.
Proposal estimates are not measured results. Final demo README/release work belongs to INN-19.
