"""One place that decides where the large, untracked inputs live.

The dataset (~0.9 GB of video and stills) and the digital twin are not in Git.
Each teammate keeps them wherever they like and points the code at that folder.
Lookup order, first match wins:

    1. an explicit path passed to the script (positional root or --flag)
    2. the FLUIDMIX_DATA environment variable (also read from .env at repo root)
    3. <repo>/external, the default documented in docs/data.md

Generated files are written to outputs_root() (FLUIDMIX_OUTPUTS, default outputs/), so the
raw data folder stays read-only and byte-identical to the manifest.

Everything below the root uses the fixed layout in docs/data.md, so a path built
here means the same file on every machine. pathlib keeps separators correct on
Windows, macOS and Linux.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV_VAR = "FLUIDMIX_DATA"
OUT_VAR = "FLUIDMIX_OUTPUTS"
VIAL_FILL = Path("csi-vial-fill-dataset/vial_fill")
TWIN = Path("csi_digital_twin_v1")


def _dotenv_value(name):
    env_file = REPO / ".env"
    if not env_file.is_file():
        return None
    for line in env_file.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            return value.strip().strip("'\"") or None
    return None


def data_root(explicit=None):
    """Return the absolute data root; a missing folder is reported by the caller."""
    raw = explicit or os.environ.get(ENV_VAR) or _dotenv_value(ENV_VAR)
    root = Path(raw).expanduser() if raw else REPO / "external"
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def outputs_root(explicit=None):
    """Where generated files go (corrected labels, colors.json). Never inside the raw data root.

    Default <repo>/outputs (ignored by Git). To share results, point FLUIDMIX_OUTPUTS at a
    shared-drive folder, but keep raw inputs and derived files in separate folders.
    """
    raw = explicit or os.environ.get(OUT_VAR) or _dotenv_value(OUT_VAR)
    root = Path(raw).expanduser() if raw else REPO / "outputs"
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def vial_fill_dir(explicit=None):
    return data_root(explicit) / VIAL_FILL


def twin_dir(explicit=None):
    return data_root(explicit) / TWIN


def require(path, what):
    """Exit with a setup hint instead of a bare FileNotFoundError."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"{what} not found at {path}\n"
            f"Set {ENV_VAR} to your data folder or place the download under "
            f"{REPO / 'external'}; see docs/data.md. "
            f"Run python scripts/check_data.py to check the layout."
        )
    return path
