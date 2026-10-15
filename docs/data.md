# Data, provenance and timing

## Download and paths

The organizers' [Dataset 2 page](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing)
is linked from the event README and provides the download route.
Download the original dataset/twin through that source; record source URL, archive name,
download date, checksum and version locally. Downloads were not repeated for this batch.

Reported scaffold inventory: csi-vial-fill-dataset (~710 MB), three cameras
(cap_seal, tank_level, vial_fill), five videos each; csi_digital_twin_v1 (~213 MB).
Recheck the downloaded inventory before assuming completeness.

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

Below the root, the layout is fixed so relative paths match on every machine:

    external/
      ds/csi-vial-fill-dataset/
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
and record that version and its source below.

fixlabels.py writes outputs/video5_frame_labels_corrected.csv and outputs/last_corrected.tsv.
color.py writes outputs/colors.json using fresh temporary frames. Failed extraction leaves
earlier results intact. Label outputs are generated before replacing each existing file.
color.py uses Pillow
and ffmpeg. It does not directly write the committed color CSV.
The tree above is the code's path contract, not a verified archive extraction layout.

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

Use corrected labels after checking against video. Original still labels assume matching
wall/video time. Historical scaffold notes report about 95% filling-label errors and
stills ending near video t=1395 s (no stills for vials 26–30); those exact claims and
the claim that all 15 videos share this ratio require independent video checks.
For another video, measure its own start/late clocks; do not reuse video-5 constants.
Use tagged time for frame seeking and corrected wall time for still/video joins.
Across cameras, timestamp equality alone does not prove synchronization: check clocks.

## Plant context and separation

Reported scaffold constraints: max 80 mL per vial, max 50 mL from T-200, T-300 out of
service; T-100 clear, T-200 orange, T-400 blue. Confirm with current plant guidance before
operations. Recipe changes require plant-manager approval. These notes are not approval.
The twin lacks tank imagery according to inherited inspection; recheck the downloaded version.

Learned model training uses synthetic data only. Freeze real calibration, validation and
test manifests with disjoint vial/recipe groups. See [evaluation](evaluation.md) for
transition denominators, baselines, leakage boundaries and missing ground truth.
