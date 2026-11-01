"""Measure gated calibration and videos-1--4 appearance for renderer revision 2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from fluidmix.fill_level_real_appearance import (  # noqa: E402
    json_text,
    measure_real_appearance,
    write_json_new,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-root",
        required=True,
        type=Path,
        help="fresh extraction root containing the closed 98-frame index.json",
    )
    parser.add_argument(
        "--external-root",
        required=True,
        type=Path,
        help=(
            "organizer csi-vial-fill-dataset root; only bounded videos 1-4 "
            "vial_fill/tank_level candidates are decoded"
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="fresh deterministic JSON output path; overwrite is refused",
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a fresh path; refusing to overwrite evidence")
    report = measure_real_appearance(
        args.frames_root, external_root=args.external_root
    )
    write_json_new(args.out, report)
    print(json_text(report), end="")


if __name__ == "__main__":
    main()
