# Dataset 2: Fluid Mixing Station
### Authors: MSOE AI Club
### Published: October 2026
### Use synchronized views of tanks, the station, and finished vials to detect color and fill problems caused by recipes and fluid carryover.

# Dataset 2: Fluid Mixing Station
[Back to the Innovation Lab overview](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026)

# The problem
Four tanks feed the vial filling station, but T-300, the third tank from the left, was out of service during data collection. The other three were in use: two held colored water and one held clear diluent. A recipe tells the system how many milliliters to draw from each tank, but it does not store the actual color in a tank. The station mixes the fluids upstream of a single nozzle. Fluid left from the previous fill can carry into the next recipe, especially when the color changes.

Build a system that detects color or fill problems and, when possible, helps an operator understand the likely cause. Flagging that a vial looks different from what was expected is not enough. The system should connect what the cameras saw to the recipe order, tank state, or color carryover that may have produced the difference.

![Timestamped camera view of the four fluid tanks at the CSI testbed]()

# Carryover when the recipe changes
When the station switches recipes, residual liquid from the previous recipe's mixture in the shared mixing path carries over into the first vial of the new batch. That vial comes out discolored and does not match its labeled recipe.

The second and third vials of each batch come out clean as the old residue flushes through. The last vial of each recipe is therefore a reliable example of that recipe's true color.

![Three side-by-side station camera frames of vials 7, 8, and 9, all from Recipe 7 (20 mL clear from T-100 and 50 mL blue from T-400). Vial 7 is green; vials 8 and 9 are blue.]()

In this example, all three vials come from Recipe 7, which doses 20 mL from T-100 (clear) and 50 mL from T-400 (blue). Vial 7, the first of the batch, comes out green. Vials 8 and 9 come out blue.

The recipe that ran just before it, Recipe 6, was orange: 30 mL clear from T-100 and 10 mL orange from T-200. Blue fluid mixing with leftover orange fluid in the shared mixing path likely explains the green tint. Once that residue flushes out, the blue comes through as intended. This is the kind of link the system should make: the odd vial traces back to the recipe that ran before it.

# Equipment and recipes
The tanks are identified by position:
- T-100 is the leftmost tank.
- T-200 is next.
- T-300 is next.
- T-400 is the rightmost tank.

Tank colors can change, so the recipe system identifies tanks by number rather than by color. In this dataset, the labels identify T-100 as clear, T-200 as orange, and T-400 as blue.

Up to three recipes can be stored and queued in the Batch Scheduler. The labeled fills cover 10 recipes in total because recipes were swapped in and out of the queue during collection.

Each fill dispenses in a fixed order: T-200 first, then T-400, then T-100, skipping any tank the recipe does not use. For example, Recipe 7 dispenses blue from T-400 and then clear from T-100. The order determines which fluid passes through the shared mixing path last, so it is useful context when you explain carryover.

Important operating constraints:
- A vial should contain no more than 80 mL to avoid spilling.
- A component may be set from 0 to 80 mL, but the station should not use less than 10 mL of a selected component in practice.
- Do not put more than 50 mL from T-200 in a recipe. It can fault the system and require a plant-manager reset.
- Do not use T-300 in a recipe. It has a configuration issue that can also require a reset.
- Do not change recipes without the plant manager's guidance.

![Batch Scheduler interface used to queue fluid recipes]() 
![Recipe setup interface for the fluid station]()

# Why the current inspection is not enough
A Teledyne vision system checks vial color and weight before the final station. Because the control system does not know the colors currently in the tanks, the vision system compares the observed color with a reference database. It passes a known color and fails an unknown one.

That check does not reliably answer whether the vial matches the intended recipe. It also does not warn the operator that the next recipe is likely to be affected by fluid left in the shared mixing path.

# What is provided
- Timestamped images and video from three fixed cameras (640×480, video at 30 fps, stills about every 2 s):
  - `vial_fill`: the station's vial-fill camera;
  - `tank_level`: the reagent-tank camera, showing tanks T-100, T-200, T-300, and T-400;
  - `cap_seal`: the cap/seal station camera.
- Five synchronized videos (IDs 1–5). The three cameras were recorded together, so you can cross-reference frames by timestamp.
- Limited labeled data for 30 vials across 10 recipes (3 vials per recipe) on the `vial_fill` camera. Each vial has:
  - its recipe ID;
  - per-tank volumes (T-100 clear, T-200 orange, T-400 blue), total volume, and dispense order;
  - fill start and end times;
  - per-frame labels marking whether a vial is being filled.
- T-300 was out of service and is not used in any recipe.
- All other data, including the `tank_level` and `cap_seal` views, is unlabeled.
- A digital twin of the fill and cap/seal stations (USD, for Isaac Sim), with cameras matched to `vial_fill` and `cap_seal` and a script to render images from them. The tanks are not modeled, and the liquid color is an approximation.

The camera timestamps let you line up tank state (`tank_level`), the fill operation (`vial_fill`), and the cap/seal station (`cap_seal`). Barcodes and labels may or may not be present, and their orientation is inconsistent. The production system does not currently rely on those barcodes.

![Timestamped station camera frame of a vial being filled with blue fluid]() 
![Timestamped station camera frame of a vial filled with orange fluid]()

The clip below shows the real `vial_fill` camera (left) next to the digital twin (right) during a fill.

![Side-by-side clip of the real vial_fill camera (left) and the digital twin (right) as a vial fills with blue fluid]()

# Get the data and digital twin
*Digital twin (datasets 1 and 2):* [Download the digital twin](https://drive.google.com/file/d/1nXRN_VZ_DKGS33jtELwn6z8WR8dbQfKe/view?usp=sharing)

*Sample data from the line:* [Open the Fluid Mixing Station dataset](https://drive.google.com/drive/folders/1qwV2BEz3ODX17kwNCr9M15DfkY8U_blQ?usp=sharing)

*Everything in one place:* [Innovation Lab data folder](https://drive.google.com/drive/folders/1GWuHV2WUNLztq_Fga6xBH7qFzhxDaac8?usp=sharing)

*For MSOE students on Rosie:* use the download link above until an on-Rosie path is published.

### Look at the real data, but train on the digital twin
The real images, video, and labels come from the line. Look through them to understand the station, the recipes, and what your generated data needs to match, but *avoid training your models on them*. The goal of this lab is to generate your training data with the digital twin.

That is how this problem has to be solved on a high-mix, low-volume (HMLV) line like CSI's, which makes many product variants in small batches and changes over often. A new product or recipe has no real images until the line has been commissioned and is running it. Collecting and labeling real examples afterwards is costly: it takes line time and operators, faults have to be produced on purpose, and the work repeats at every changeover. A digital twin can produce labeled examples of a recipe, including rare problems, before it ever runs on the line.

Need more real data? See [Collect and Test Your Own Data](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Data-Collection). We may also release a held-out set of real data for testing if there is interest.

Start by checking that you can match a fill-station frame to the closest tank and cap/seal frames. Record any missing frames, clock offsets, or timestamp drift before building a model.

# Starter approaches
These are ideas to get you started, not a required plan. Combine them, change them, or take a different direction.

- *Start simple.* See how far basic color and fill-level measurements of the finished vial can go before reaching for a larger model.
- *Link the cameras.* The three cameras are synchronized, so one moment can be followed across stations: what was loaded in the tanks, how the vial filled, and how it looked at cap/seal. Linking these views lets a system reason about the cause of a problem, not just detect it.
- *Use the recipe sequence.* Each vial comes from a known recipe, and the recipe that ran before it can matter. Think about how the order of recipes and fluids might explain what the camera sees.
- *Reason with Cosmos Reason.* A vision-language model can look at images or video alongside the recipe and context, and explain whether a vial matches what was intended and why it might not. Examples of good and bad fills can guide it.
- *Generate synthetic data with the digital twin.* The twin's cameras match the real `vial_fill` and `cap_seal` views, so you can render new scenes with different fills, colors and lighting as your training data. Synthetic training data can cover recipes, defects and conditions that are rare or missing in the real data, and it comes with labels at no extra cost.
- *Grow the labeled test data.* Videos 1–4 and the `tank_level` and `cap_seal` views have no labels, but they hold many more fills and recipe changes that you can label to check your model against. You can also collect and label your own data on the line (see [Collect and Test Your Own Data](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Data-Collection)).

Questions that may help narrow the first experiment:
- Can a simple color measurement establish a useful baseline before using a vision-language model?
- Can you infer the tank colors from camera views, then compare the expected blend with the filled vial?
- How much does the prior recipe explain the next vial's color?
- Can the system flag risky recipe transitions before the vial is filled?
- How will lighting changes and transparent vial material affect color measurements?

Relevant tools include Cosmos Reason for multi-image or video context, in-context learning with example fills, and a time-series model if sensor data becomes available. See [Tools and Resources](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Resources).

# Testing
Split tests by production run or time window so nearly identical neighboring frames do not appear in both training and test sets.

Because the labeled data is limited to 30 vials across 10 recipes, we suggest leave-one-recipe-out evaluation: hold out all the vials of one recipe, use the other recipes as reference examples, and repeat for each recipe. This keeps near-identical vials of the same recipe out of both sets and tests whether your approach works on a recipe it has not seen.

Report false alarms and missed defects separately. If the solution claims a likely cause, evaluate that explanation separately from defect detection.

For additional collection or live-line testing, follow [Collect and Test Your Own Data](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Data-Collection).