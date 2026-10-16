# Contributor quickstart

Run from the repository root. Start with Python 3.11 or newer; [review evidence](review-inn-22-24.md)
records the version tested. GPU access and external data are unnecessary for the CPU checks.
There is no package installer: scripts and tests add src/ to their own import path.

## Environment

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-dev.txt
    python -m pytest tests -q
    python scripts/check_agent_docs.py

Pytest runs the five original synthetic math checks plus review regressions for input
validation, data-tool failures and skill metadata. These do not verify real camera accuracy
or physical rendering. The direct python tests/test_absorbance.py runner covers only the
five original math checks and is not a substitute for the full pytest suite.

## Small data and smoke run

The two small CSVs in data/ are committed. No download is needed.

    python scripts/smoke_absorbance.py

Expected: old-fluid fraction 0.150000, residual below 1e-6, and "synthetic smoke passed".
The tank values and reference are artificial test fixtures, not plant calibration.
The run creates no output files.

    python scripts/fit_absorbance_video5.py

Expected with committed CSVs: 20 presumed-clean vials, negative fitted absorbance,
leave-one-recipe-out median Delta E 12.6 and max 26.4.
This reproduces a measurement failure. Carryover percentages later printed by the script
come from that invalid fit; they are not valid estimates of real residue.
The diagnostic currently exits successfully despite the warning.

## Optional tools

Read [data setup](data.md) before real-video work. Color extraction needs Pillow and a
separately installed ffmpeg executable. Pillow is included in requirements-dev.txt:

    ffmpeg -version

USD variant creation needs a compatible usd-core build:

    python -m pip install usd-core
    python scripts/twin_liquid_variant.py --help

Rendering needs the downloaded twin's documented Isaac Sim/GPU environment.
See [twin](twin.md). This CPU quickstart does not establish render readiness.
Camera models, dynamic thresholds and Cosmos explanation are future work.

## Teammate check

From a clean checkout, record commit, OS, Python version, commands, actual outputs,
host skill discovery and corrections in the issue or review evidence.
A lead's own smoke run does not satisfy INN-24's teammate acceptance check.
Use the [teammate receipt](teammate-quickstart.md) to record the run without losing
failed steps or corrections. The template is unfilled; it is not validation evidence.
