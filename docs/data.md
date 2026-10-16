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
| Also in the bundle, not used | csi-vial-fill-dataset.zip, "Copy of csi_digital_twin.zip" (zipped copies of the two folders), fanuc_er4ia_hackathon_starter_v05.zip (a different challenge) |
| Dataset file dates | videos 2026-10-01; image_index.csv and videos.csv 2026-10-05; dataset README 2026-10-05; twin README 2026-10-08 |
| Manifest | `data/dataset_manifest.csv`, 4,688 files (4,673 dataset, 15 twin), 1,475,294,702 bytes, written 2026-10-16 |

The earlier "~710 MB" and "~213 MB" figures are the sizes of the zipped copies. Extracted,
the dataset is 726,696,048 bytes and the twin is about 749 MB (most of it facility.usdc).

`check_data.py` passed on this copy (sizes and SHA-256) right after the manifest was written.
That only proves the manifest matches the copy it came from. **Pending:** a second teammate,
on another operating system, downloads or copies the data and runs `check_data.py`. Until that
passes, cross-machine identity is not verified.

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
    python scripts/fixlabels.py
    python scripts/color.py

check_data.py compares file paths, sizes and SHA-256 hashes against the committed
data/dataset_manifest.csv, so a pass means you read the same bytes as your teammates.
It also flags names that differ only by case, which macOS and Windows hide but Linux
GPU hosts do not. The manifest is written once from a verified download with
check_data.py --write; regenerate it only when the team adopts a new dataset version,
and record that version and its source above.

fixlabels.py writes outputs/video5_frame_labels_corrected.csv and outputs/last_corrected.tsv.
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

fixlabels.py embeds video-5 clock observations: 19:47:11.161 at video second 0,
20:21:33.830 at video second 1697 (2026-09-23). Their ratio is about 1.2155 real
seconds per tagged video second, equivalent to about 24.7 real fps for a 30 fps file.
This batch verifies the code's calculation, not the original burned-in clocks.

Measured 2026-10-15 for all 15 videos: [video timing](video_timing.md) (1.2157 wall s per video
second everywhere; the source frame labels are misaligned). Use corrected labels after checking against video. Original still labels assume matching
wall/video time. Historical scaffold notes report about 95% filling-label errors and
stills ending near video t=1395 s (no stills for vials 26–30); those exact claims and
the claim that all 15 videos share this ratio require independent video checks.
For another video, measure its own start/late clocks; do not reuse video-5 constants.
Use tagged time for frame seeking and corrected wall time for still/video joins.
Across cameras, timestamp equality alone does not prove synchronization: check clocks.

## Plant context and separation

**T-300 is unavailable for the whole hackathon** (organizer feedback, Owen, chat, 2026-10-14).
The dataset README agrees: T300 was intermittent/out of service during collection and was not
used, so every recipe draws only from T-100, T-200 and T-400. T-300 supplies no calibration,
validation or test data, and none should be planned around it. The code's tank list
(TANKS in src/fluidmix/absorbance.py) has no T-300 and rejects recipes that use it.

Reported scaffold constraints: max 80 mL per vial, max 50 mL from T-200;
T-100 clear, T-200 orange, T-400 blue. Confirm with current plant guidance before
operations. Recipe changes require plant-manager approval. These notes are not approval.
The twin lacks tank imagery according to inherited inspection; recheck the downloaded version.

Learned model training and test sets use synthetic twin data (scope decision above). Real
videos are used only for occasional validation; when they are, freeze real calibration and
validation manifests with disjoint vial/recipe groups. See [evaluation](evaluation.md) for
transition denominators, baselines, leakage boundaries and missing ground truth.
