# Dataset 2 inventory (INN-7 T1)

Measured 2026-10-15 on one local copy at `external/csi-vial-fill-dataset/` with
`python scripts/inventory_data.py --write`, which writes
`data/dataset_inventory.csv` (one row per camera and video ID).
The manifest `data/dataset_manifest.csv` records path, size and SHA-256 for 4,688 files (4,673
dataset, 15 twin), and `scripts/check_data.py` passes on this machine. A second teammate reported a pass on
Windows 11 after downloading the whole Drive folder (relayed, 2026-10-16; see [data](data.md)).
The dataset-only zip omits three "(1)" duplicates, so use the whole folder. The bundle checksum
is recorded in [data](data.md).
"Source README" below means `csi-vial-fill-dataset/README.md` shipped with the download.

## What is there (measured)

- 4,673 files, 726,696,048 bytes (about 727 MB in decimal megabytes, or 693 MiB in binary
  mebibytes): 15 MP4 videos, 4,650 JPEG stills, 6 CSVs, 1 JSON, 1 README. Counted from the
  manifest and confirmed on disk 2026-10-16 (dotfiles excluded). The 4,650 stills include the
  three duplicates below.
- Three cameras (`cap_seal`, `tank_level`, `vial_fill`) × five video IDs (1–5). Every one of the
  15 videos exists, is H.264, 640×480, 30 fps. The three cameras of one video ID have identical
  frame counts and durations.

| Video | Duration (s) | Frames | Stills (tank, vial) | Stills (cap) | Vials (reported) | Recipes | Labeled |
|---|---|---|---|---|---|---|---|
| 1 | 1122.1 | 33,662 | 558 | 5 | 19 | unknown | no |
| 2 | 160.1 | 4,802 | 78 | 78 | 3 | unknown | no |
| 3 | 193.1 | 5,792 | 95 | 95 | 6 | unknown | no |
| 4 | 316.1 | 9,482 | 157 | 157 | 6 | unknown | no |
| 5 | 1698.1 | 50,942 | 845 | 845 | 30 | 5–14, 3 vials each | vial_fill only |

Vial counts for videos 1–4 come from the source `videos.csv`; nobody has counted them in video.
Video 5's recipe table (`vial_fill/labels/video5_vial_recipes.csv`) matches the committed
`data/video5_vial_recipes.csv` in its recipe columns.
Its 845 frame labels split 509 in a fill, 336 between fills, as the source README says.

## Anomalies found (measured)

1. **Path layout (resolved 2026-10-16).** The data sits at `external/csi-vial-fill-dataset/`, the
   layout the organizer bundle extracts to. The code first expected `external/ds/...`; `paths.py`,
   `check_data.py`, the tests and docs/data.md now use the archive layout.
2. **Other folders in `external/` (added after the first inventory).** `csi_digital_twin_v1/` is the
   twin, matching the path in docs/data.md; see [twin](twin.md). `csi_digital_twin_v1 2/` is a
   byte-identical copy. `fanuc_er4ia_sim/` is the starter for a separate FANUC cube-stacking
   challenge, not Dataset 2. The `Rockwell-NVIDIA-Fall-2026-Hackathon-*.zip` (1,864,155,302 bytes, about 1.86 GB or 1.74 GiB) is the organizer
   bundle: the dataset (extracted and as a nested zip), the twin and a FANUC starter zip.
3. **Three duplicate stills.** `tank_level_20260922_171745_728485(1).jpg`,
   `vial_fill_20260923_191129_974788(1).jpg` and `vial_fill_20260923_200146_560489(1).jpg` are
   byte-identical copies of indexed files and are not in any `image_index.csv`. That is why the
   folders hold 1,734/1,735 files while the README and indexes say 1,733. Exclude them so they
   cannot become duplicate frames across splits.
4. **One cap_seal still is `video_id=unassigned`** (`cap_seal_20260922_175155_329384.jpg`).
5. **cap_seal video 1 has only 5 stills**, with a 450 s gap. The source README notes this.
   Its timestamps do not match any vial_fill still, so cap_seal V1 cannot be joined by timestamp.
   For videos 2–5, tank_level, vial_fill and cap_seal share timestamps exactly.
6. **Video timing.** The burned-in clock runs 1.2157 s per video second in all 15 videos, so the
   source still-to-video mapping and the still coverage are both wrong. Measured and corrected in
   [video timing](video_timing.md).

## What is missing or unknown

| Item | Status | Why it matters |
|---|---|---|
| Defect labels, any video | none exist | Video 5 labels give recipe and fill window only. Defect labels come from the twin, which knows what it rendered. |
| Recipes, vial windows, videos 1–4 | unknown | The README says "label as you see fit". Not needed: test data comes from the twin. |
| Recipe before video 5 vial 1 | unknown | Recipe 5 is noted "swapped in". The first vial has no known carryover context. |
| Recipe IDs 1–4 | not in dataset | Recipes start at 5. Earlier recipes are not described anywhere. |
| Transition IDs | not in dataset | Derived from consecutive vials in `data/video5_transitions.csv`; see below. |
| Vial IDs, videos 1–4 | not in dataset | Only video 5 numbers its vials. |
| Stills for the last ~18% of each video | missing | Dropped by the 1:1 timing assumption (about 368 s of wall time in video 5); see [video timing](video_timing.md). |
| License | not stated | The source README gives an owner (NVIDIA contact) but no license. Ask the organizers before redistribution or publication. |
| Download version and checksum | recorded | Bundle SHA-256 in [data](data.md); per-file SHA-256 in `data/dataset_manifest.csv`. Second-machine pass reported 2026-10-16 (T0), relayed in a PR comment, not rerun by us. |
| T300 | out of service | The source README says it was intermittent and not used. This matches the T8 statement. |

## Previous-recipe context (T3)

`python3 scripts/build_transitions.py` reads `data/video5_vial_recipes.csv` and writes
`data/video5_transitions.csv`: one row per video-5 vial with the previous vial and recipe, whether
the recipe changed, the vial's position within its run of same-recipe vials, and an
`initial_batch` flag. A "transition" here is a consecutive vial pair, derived from vial order;
the dataset does not record transitions. `--check` fails if the committed CSV is stale.

- Measured from the committed CSV (2026-10-16): 30 vials, **29 transitions** with a known
  previous vial: **9 recipe changes** (5→6, 6→7, ... 13→14) and **20 repeats** of the same recipe.
  Every recipe runs exactly 3 vials, so each change vial has `position_in_run` = 1.
- Vial 1 has `prev_recipe_id` and `recipe_changed` = "unknown" and `initial_batch` = 1. Nothing
  records what ran before it; recipe 5 is noted "swapped in". Leave it out of carryover validation.
- Still coverage of the 9 change vials (aligned labels, `outputs/video5_frame_labels_aligned.csv`):
  vials 4, 7, 10, 13, 16, 19, 22 and 25 have 7–28 stills each (6–25 more than 2 s from a fill
  boundary); vial 28 has none, because it falls in the tail without stills. See [labels](labels.md).
- Videos 1–4 have no recipe data, so no transitions can be derived. They are out of scope,
  because test data comes from the twin.

## What this means for INN-7

The digital twin generates training and test data (project description: "look at the real
data, but train on the digital twin"). The real data is for understanding the station and
occasionally validating assumptions such as absorbance metrics. It is not a test set, and no
labels exist beyond the video-5 vial-fill recipes unless the team makes them.

- The only real validation reference is video 5: 30 vials, 10 recipes, fill windows in video
  seconds. T2 found its frame-label file misaligned (26 of 509 filling stills keep the right vial);
  use the aligned labels from [video timing](video_timing.md).
- Videos 1–4 have no recipes or vial windows. They can show the station and timing, not recipe
  ground truth.
- Never train on real frames, and keep any real frames used to check a metric out of tuning.

## Next actions

1. T2 is measured; see [video timing](video_timing.md) for what remains.
2. T0 second-machine check is reported passed (see [data](data.md)); optional follow-up: let `check_data.py` skip twin files for dataset-only downloads.
3. Ask the organizers about the license and the missing tail stills.
