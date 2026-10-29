# Presentation design system and finals deck

Team design system for the Nov 5 finals deck, the Nov 1 video and one-page summary, and
every chart. Made 2026-10-29. It is a small first system with no earlier brand behind it.

- Design system page: https://claude.ai/artifact/Y7GUW3CRyHG9kqZaKMD7iy
- Finals deck scaffold (Slides): https://claude.ai/artifact/LD8MmWKmszxyeT1HnskPy1

Both pages are private until their owner shares them from the page's Share menu.
This folder is the source for the values. If the page and this folder differ, this folder wins.

| File | Use |
|---|---|
| [tokens.json](tokens.json) | Colours (light and dark), type styles, spacing, radius, each with a usage note |
| [tokens.css](tokens.css) | CSS variables, generated: `python scripts/presentation_tokens_css.py` |
| [moe.mplstyle](moe.mplstyle) | matplotlib style: `plt.style.use("docs/presentation/moe.mplstyle")` |

## Rules in short

- **Colour meaning is fixed.** `tank-blue` is our method, `baseline` grey is every baseline,
  `tank-orange` is carryover (the old recipe), `tank-clear` is clean, `flag` red is a miss or
  failure. The colours are the line's own fluids: T-400 blue, T-200 orange, T-100 clear.
  We use no Rockwell Automation or NVIDIA marks or brand colours: we have no permission.
- **Type.** IBM Plex Sans for words, IBM Plex Mono for numbers, tags and IDs (Google Fonts).
  Sizes in tokens.json are for the 1920 x 1080 slide canvas. Never under 24px on a slide.
- **Every number has a source line and a count** ("8 of 10", not "80%").
- **Evidence tags.** `MEASURED` (we ran it), `PROPOSED` (a plan), `UNCHECKED` (reported, not
  repeated). Follow AGENTS.md: never show a proposal or estimate as a result.
- **Charts.** Bars start at zero, y gridlines only, direct labels, a caption with source,
  vial count and date. No pies, no 3D, no dual axes.
- **Placeholders.** Yellow `placeholder-bg` box with a dashed `placeholder-ink` border.
  A yellow box left in the deck means the slide is not finished.

## Deck sections

The scaffold follows "What judges expect" in
[hackathon info](../ground-truth-hackathon-info/hackathon-info.md). Each slide has speaker
notes that say its purpose and which file to take numbers from.

| # | Slide | What goes there | Take it from |
|---|---|---|---|
| 1 | Cover | Team, dataset, one-line claim | [proposal](../proposal.md) |
| 2 | The problem | Carryover in the first vial; the 8 of 10 observation, tagged UNCHECKED | [plan](../plan.md), [labels](../labels.md) |
| 3 | Why it matters | Cost of a bad vial, how often recipes change, what the Teledyne check misses | [evaluation](../evaluation.md) |
| 4 | Business case | Cost today, what we save, what adoption takes | judges' list |
| 5 | Method | Five pipeline steps, each with a true status | [architecture](../architecture.md) |
| 6 | Data | Twin for training, real video for evaluation, disjoint splits | [evaluation](../evaluation.md) |
| 7 | Results | Our method against the first-vial, vision-only and drainage-only baselines | frozen test run |
| 8 | Limits | Real failure cases | test run, [data inventory](../data_inventory.md) |
| 9 | Live demo | Supervised run on the CSI line, trial log | [collecting your own data](../ground-truth-hackathon-info/collecting-your-own-data.md) |
| 10 | Operator message | Numbers in, a real Cosmos message out | [architecture](../architecture.md) |
| 11 | Next steps | Three steps, each closing a shown limit | [proposal](../proposal.md) |
| 12 | Credits | Data, models, code, tools, team | final submission rules |

The Teledyne colour check is context, not a baseline (decided 2026-10-16, see
[evaluation](../evaluation.md)).
