"""Build a videos-1--4, hash-bound photometric manifest from organizer stills."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.paths import outputs_root
from fluidmix.real_photometric_manifest import build_photometric_frame_manifest
from fluidmix.real_photometric_range import write_no_clobber_json


def _output_below_root(root: Path, requested: str) -> Path:
    candidate = Path(requested)
    target = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("output must be below FLUIDMIX_OUTPUTS") from error
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-index", required=True, help="reviewed videos-1--4 image_index.csv subset")
    parser.add_argument("--images-root", required=True, help="organizer vial_fill/images directory")
    parser.add_argument("--stable-region-config", required=True, help="reviewed fixed-background JSON config")
    parser.add_argument("--output", default="real_photometric_frame_manifest.json", help="relative path below FLUIDMIX_OUTPUTS")
    args = parser.parse_args()
    manifest = build_photometric_frame_manifest(args.image_index, args.images_root, args.stable_region_config)
    output = _output_below_root(outputs_root(), args.output)
    write_no_clobber_json(output, manifest)
    print(f"built {len(manifest['frames'])} frame rows: {output}")


if __name__ == "__main__":
    main()
