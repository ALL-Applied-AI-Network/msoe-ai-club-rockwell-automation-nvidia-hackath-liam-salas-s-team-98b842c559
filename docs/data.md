# Data, provenance and timing

## Download and paths

The organizers' [Dataset 2 page](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing)
is linked from the event README and provides the download route.
Download the original dataset/twin through that source. The dataset README names the
owner/contact as tcellio@nvidia.com and states no license; ask before redistributing.

### Scope decision: the digital twin is in scope

Decided by Liam (the download holder) on 2026-10-15/16. Training and test data are generated with the
digital twin (csi_digital_twin_v1). The five real videos per camera are used only for
occasional validation, for example absorbance metrics on real vials. They are not training
data. This is why the manifest tracks both csi-vial-fill-dataset and csi_digital_twin_v1.
Caveat: twin material appearance is not physical absorption (see [twin](twin.md)), so a
model that scores well on twin test data still needs the real-video check.

### Source and version (manifest of 2026-10-16)

Measured on Liam's machine, which holds the original download:

| Item | Value |
|---|---|
| Bundle file | Rockwell-NVIDIA-Fall-2026-Hackathon-20261015T225158Z-1-001.zip (file timestamp 2026-10-15) |
| Bundle size | 1,864,155,302 bytes |
| Bundle SHA-256 | d67c0643fd55e11061bd90fb0ed7833f203fde7f6957975a886aacaec8314b76 |
| Bundle top folder | Rockwell-NVIDIA-Fall-2026-Hackathon/ |
| Extracted folders used | csi-vial-fill-dataset/ and csi_digital_twin_v1/ |
| Also in the bundle, not used | csi-vial-fill-dataset.zip, "Copy of csi_digital_twin.zip" (zipped copies of the two folders; the dataset zip is not a complete copy, see below), fanuc_er4ia_hackathon_starter_v05.zip (a different challenge) |
| Dataset file dates | videos 2026-10-01; image_index.csv and videos.csv 2026-10-05; dataset README 2026-10-05; twin README 2026-10-08 |
| Manifest | `data/dataset_manifest.csv`, 4,688 files (4,673 dataset, 15 twin), 1,475,294,702 bytes, written 2026-10-16 |

The earlier "~710 MB" and "~213 MB" figures are the sizes of the zipped copies. Extracted,
the dataset is 726,696,048 bytes and the twin is about 749 MB (most of it facility.usdc).

`check_data.py` passed on this copy (sizes and SHA-256) right after the manifest was written.
That only proves the manifest matches the copy it came from. **Second machine (2026-10-16):**
a teammate reported (relayed in a PR #5 comment and INN-7; not rerun by us) a pass on Windows 11,
Python 3.12.10, at 25bb3ed: `ok: 4688 files match the manifest (sizes and hashes)`, after
downloading from the organizers' Drive themselves, and a file-name comparison against the Drive
folder (same 4,688 files). So cross-machine identity of the manifest files is supported. It does
not cover timing or alignment, which were not rerun on that machine.

**Canonical source: the expanded whole Drive folder.** The manifest was written from it, and a
teammate's file-name comparison against the Drive folder found the same 4,688 files. Download the
whole folder, not just `csi-vial-fill-dataset.zip`. That ready-made zip omits the three "(1)"
duplicate stills listed below, so `check_data.py` exits 1 with 3 missing files until they are
fetched separately from Drive (they then match). The three are unindexed duplicates, so analysis
counts are unaffected. The manifest keeps them because they exist in the canonical folder.

For a dataset-only download, run `python scripts/check_data.py --only dataset`. It skips the 15
twin rows; the three "(1)" files still count as missing unless fetched. `--only twin` does the
reverse, and `--only` cannot be combined with `--write`. A full pass needs both folders.

Dataset zip checked separately (reported by Owen's independent download, 2026-10-16; not rerun by
us): file `csi-vial-fill-dataset.zip` from the organizer folder
(https://drive.google.com/drive/folders/1GWuHV2WUNLztq_Fga6xBH7qFzhxDaac8), 709,560,806 bytes,
SHA-256 `14130f55b0be4fbdf9b4d4b657b77f3f2b3cfbff3d9f51f877284cf1167bc77a`. Its other 4,670
dataset files matched the manifest by size and SHA-256. Those figures describe that download only;
a later re-upload could differ.

The manifest skips dotfiles (any name starting with "."), so macOS .DS_Store files never
cause a mismatch. It records files exactly as they exist, including three duplicates:

- tank_level/images/tank_level_20260922_171745_728485(1).jpg
- vial_fill/images/vial_fill_20260923_191129_974788(1).jpg
- vial_fill/images/vial_fill_20260923_200146_560489(1).jpg

Each is byte-identical to the file without "(1)" and is not listed in any image_index.csv.
Exclude them from analysis so a still is not counted twice.

Other folders that may sit next to the data are not tracked. "csi_digital_twin_v1 2" is a
byte-identical accidental copy of the twin (only a .DS_Store differed); delete it or ignore it.
fanuc_er4ia_sim is the unrelated starter for another challenge.

### Where the data lives on your machine

Raw inputs and generated files are separate. Scripts never write into the data root; they
write to FLUIDMIX_OUTPUTS (default outputs/, ignored by Git). Trained models, edits and
labels the team shares go in a shared-drive folder: point FLUIDMIX_OUTPUTS there, and
keep it different from FLUIDMIX_DATA.

The data is never committed. Each teammate keeps one data root and the code finds it
through src/fluidmix/paths.py, in this order: a path passed to the script, then the
FLUIDMIX_DATA environment variable (or the same key in an ignored .env, see .env.example),
then the default external/ inside the repo. Use the default unless the data must live on
another disk. Prefer the variable over a symlink: Windows symlinks need extra permissions.

Windows: set it per user (`setx FLUIDMIX_DATA "D:\\data\\innolab"`, then reopen the terminal)
or put `FLUIDMIX_DATA=D:/data/innolab` in .env. Forward slashes work in .env. Paths with
spaces need no quotes in .env. The code uses pathlib, so no other path edits are needed.

The download page is login-gated. One teammate with access downloads it, runs
`check_data.py --write`, and shares the folder (shared drive, not Git). Everyone else points
FLUIDMIX_DATA at their synced or copied folder and runs `check_data.py`. Sync clients can
leave half-downloaded files, so mark the folder "available offline" first.

Below the root, the layout is fixed so relative paths match on every machine. It is the
layout the organizer bundle produces: move the two folders out of
Rockwell-NVIDIA-Fall-2026-Hackathon/ into the data root without renaming them.

    external/
      csi-vial-fill-dataset/
        vial_fill/image_index.csv
        vial_fill/labels/video5_vial_recipes.csv
        vial_fill/videos/vial_fill_video_5.mp4
      csi_digital_twin_v1/
        stations/fill_station/fill_station.usda

Then check the copy and run from repo root:

    python scripts/check_data.py
    python scripts/measure_video_clock.py --check-stills 20
    python scripts/verify_still_alignment.py --per-video 12
    python scripts/align_still_labels.py
    python scripts/color.py

check_data.py compares file paths, sizes and SHA-256 hashes against the committed
data/dataset_manifest.csv, so a pass means you read the same bytes as your teammates.
It also flags names that differ only by case, which macOS and Windows hide but Linux
GPU hosts do not. The manifest is written once from a verified download with
check_data.py --write; regenerate it only when the team adopts a new dataset version,
and record that version and its source above.

The three timing scripts write clock_fits.csv, still_video_map.csv, still_alignment_check.csv
and video5_frame_labels_aligned.csv to FLUIDMIX_OUTPUTS; see [video timing](video_timing.md).
fixlabels.py is legacy (see the timing section below); do not run it to make labels.
color.py writes outputs/colors.json using fresh temporary frames. Failed extraction leaves
earlier results intact. Label outputs are generated before replacing each existing file.
color.py uses Pillow
and ffmpeg. It does not directly write the committed color CSV.
The tree above matches the extracted bundle listed in the source table (checked 2026-10-16).

## Committed small CSVs

- data/video5_vial_recipes.csv: 30 vial records, recipe IDs 5–14, volumes in mL,
  fill endpoints in video seconds/frames, dispense order and batch type.
- data/video5_vial_colors.csv: 30 rough sRGB R/G/B measurements and color-distance columns.
  The scaffold attributes these to color.py: median of the most saturated 0.7% of pixels
  in a fixed crop over three post-fill frames. CSV export was not automated in that script.
  Original extraction and source video have not been independently repeated here.

These are inherited derived artifacts, not independent defect/cause labels or valid
transmission calibration. The rough measurements lack matched I0/exposure control.
The fit reproduces negative absorbance; see [physics](absorbance.md).

## Timing and label caveats

Measured 2026-10-15 for all 15 videos; details in [video timing](video_timing.md). The
burned-in wall clock advances 1.2157 s per video second in every video and camera: the
cameras captured about 24.7 frames per real second, but the files play at 30 fps. The source
still labels assume 1 wall second per video second, so they are misaligned: of the 509 stills
the source marks as filling, 26 keep the same vial after alignment. Stills end at video
second 1395.5 in video 5, so vials 26–30 have no stills.

For video-5 still labels, use the output of `scripts/align_still_labels.py`
(video5_frame_labels_aligned.csv in FLUIDMIX_OUTPUTS), not the source frame-label file.
`scripts/fixlabels.py` is legacy: its embedded video-5 constants differ from the measurement
by 0.16–0.50 s and give a different vial for 8 stills, so do not use its output as labels.
Use video seconds (frame / 30) for frame seeking and the fitted wall time from clock_fits.csv
for still/video joins; do not reuse one video's constants for another.
Across cameras, timestamp equality alone does not prove synchronization: check clocks.

## Plant context and separation

**T-300 is unavailable for the whole hackathon** (organizer feedback, Owen, chat, 2026-10-14).
The dataset README agrees: T300 was intermittent/out of service during collection and was not
used, so every recipe draws only from T-100, T-200 and T-400. T-300 supplies no calibration,
validation or test data, and none should be planned around it. The code's tank list
(TANKS in src/fluidmix/absorbance.py) has no T-300 and rejects recipes that use it.

**Teledyne vision-system outputs are not in the dataset and are deferred** (team decision,
2026-10-16). The organizer page describes the system only as context: it passes any colour in
its reference database, so it does not test recipe match or carryover. Nothing in calibration,
evaluation or baselines depends on it; its outputs may be gathered later for context.

Reported scaffold constraints: max 80 mL per vial, max 50 mL from T-200;
T-100 clear, T-200 orange, T-400 blue. Confirm with current plant guidance before
operations. Recipe changes require plant-manager approval. These notes are not approval.
The twin lacks tank imagery according to inherited inspection; recheck the downloaded version.

Learned model training and test sets use synthetic twin data (scope decision above). Real
videos are used only for occasional validation; when they are, freeze real calibration and
validation manifests with disjoint vial/recipe groups. See [evaluation](evaluation.md) for
transition denominators, baselines, leakage boundaries and missing ground truth.
