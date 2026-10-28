"""Measure configured videos-1--4 background photometric/shadow nuisance ranges."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluidmix.paths import outputs_root
from fluidmix.real_photometric_range import build_photometric_receipt, write_no_clobber_json


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
    parser.add_argument("--input", required=True, help="external JSON frame manifest; raw frames remain external")
    parser.add_argument("--output", default="real_photometric_ranges.json", help="relative receipt path below FLUIDMIX_OUTPUTS")
    args = parser.parse_args()
    receipt = build_photometric_receipt(args.input)
    output = _output_below_root(outputs_root(), args.output)
    write_no_clobber_json(output, receipt)
    print(f"{receipt['status']}: {output}")


if __name__ == "__main__":
    main()
