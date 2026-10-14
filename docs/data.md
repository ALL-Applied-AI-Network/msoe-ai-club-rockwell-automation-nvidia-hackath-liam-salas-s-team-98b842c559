# Data, provenance and timing

## Download and paths

The organizers' [Dataset 2 page](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing)
is linked from the event README and provides the download route.
Download the original dataset/twin through that source; record source URL, archive name,
download date, checksum and version locally. Downloads were not repeated for this batch.

Reported scaffold inventory: csi-vial-fill-dataset (~710 MB), three cameras
(cap_seal, tank_level, vial_fill), five videos each; csi_digital_twin_v1 (~213 MB).
Recheck the downloaded inventory before assuming completeness.

The existing fixlabels.py and color.py append /ds/csi-vial-fill-dataset/vial_fill to
their positional root. Arrange files as:

    external/
      ds/csi-vial-fill-dataset/
        vial_fill/image_index.csv
        vial_fill/labels/video5_vial_recipes.csv
        vial_fill/videos/vial_fill_video_5.mp4
      csi_digital_twin_v1/
        stations/fill_station/fill_station.usda

Then run from repo root:

    python scripts/fixlabels.py external
    python scripts/color.py external

fixlabels.py writes external/video5_frame_labels_corrected.csv and external/last_corrected.tsv.
color.py writes external/colors.json using fresh temporary frames. Failed extraction leaves
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
