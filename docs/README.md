# Contributor documentation

Start at [quickstart](quickstart.md) for CPU setup, checks and a synthetic smoke run.
Read [AGENTS.md](../AGENTS.md) for shared operating policy.

| Document | Read it when |
|---|---|
| [Proposal](proposal.md) | Checking team intent; source of truth since 2026-10-16 |
| [Plan](plan.md) | Checking current implementation and open decisions |
| [One-page summary](one-page-summary.md) | Reviewing the concise evidence-bounded project deliverable |
| [Architecture](architecture.md) | Connecting recipes, measurements, anomaly evidence and explanations |
| [Part ownership and read isolation](architecture.md#part-ownership-and-read-isolation) | Finding the authoritative Part A/Part B module and script inventory, dependency audit, exact-path read guard, evaluation join rule and WP7 registration gate |
| [`evidence-record-v1`](contracts/evidence-record-v1.md) | Reading the exact Part B per-stage record, phases, artifact verification, and production oracle exclusion |
| [`sim-metadata-v1`](contracts/sim-metadata-v1.md) | Reading the segregated Part A simulation/runtime record and post-inference case binding |
| [Data](data.md) | Downloading inputs, tracing CSVs or aligning video |
| [Data inventory](data_inventory.md) | Counts, anomalies and missing labels in Dataset 2 |
| [Video timing](video_timing.md) | Mapping stills and labels to video frames |
| [Dispense stream](stream.md) | Reading tank timing or stream colour from the fill-station video (INN-52) |
| [Labels](labels.md) | Finding label sources, unknowns and which real frames to exclude from validation |
| [Absorbance](absorbance.md) | Using physics functions and judging calibration validity |
| [Evaluation](evaluation.md) | Freezing partitions, thresholds and baseline comparisons |
| [WP6 model comparison](wp6-model-comparison.md) | Reading the no-run RCA-bench record and DP4 guarded-template fallback |
| [Twin](twin.md) | Making variants or assessing rendering limits |
| [Tank station](tank-station.md) | Building and rendering the optional four-tank USD assembly |
| [Operations](operations.md) | Configuring inputs, credentials, outputs and compute shutdown |
| [Skills](skills.md) | Choosing workflows and checking host discovery |
| [Glossary](glossary.md) | Looking up terminology |
| [Review evidence](review-inn-22-24.md) | Reviewing this batch and remaining acceptance checks |
| [Exploration final report](exploration-final-report.md) | Reviewing the 2026-10-24 Cosmos/camera/twin exploration outcome, limits and next experiments |
| [Next long-run agent brief](next-long-run-agent.md) | Continuing with Brev rendering, synthetic vision training, validity gates and a larger Cosmos run |
| [F2 composition-substitution cohort](physx-composition-cohort.md) | Integrating the train-only 120 Hz keep/discard schedules after the three-stage volume gate passes |
| [Hackathon info](ground-truth-hackathon-info/hackathon-info.md) | Checking an organizer rule, deadline, judging criterion or dataset description. Large reference copies (~720 lines in 6 files): search with grep and read only the matching section |
| [Current acceptance audit](acceptance-inn-23-24.md) | Checking the Oct 16 host evidence, issue coverage and pending teammate trial |
| [Teammate receipt](teammate-quickstart.md) | Recording the required independent clean-checkout trial |
| [Draft feasibility protocol](feasibility-protocol.md) | Preparing INN-32 tolerances and an evidence-linked decision without claiming agreement |
| [INN-52 revised scope](inn-52-scope.md) | Reviewing the proposed INN-52 rewording before it goes to Linear |

Sources: [merged proposal PR #1](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/pull/1),
[draft scaffold PR #2](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-liam-salas-s-team-98b842c559/pull/2),
[Linear project](https://linear.app/innovation-lab-2026/project/mixture-of-experts-fluid-mixing-carryover-315d582ab3f1).
Keep detailed facts in one place and link to them. Label historical scaffold observations
as reported until rechecked; record input, command and limits for measurements.
Proposal estimates are not measured results. Final demo README/release work belongs to INN-19.
