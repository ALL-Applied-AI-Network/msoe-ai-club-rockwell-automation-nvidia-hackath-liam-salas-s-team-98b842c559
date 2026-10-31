"""Write deterministic frame specifications for a FillCam render batch."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix.fill_level_variation import sample_spec  # noqa: E402


def parse_pair(text):
    seed_text, separator, level_text = text.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("pairs use SEED:LEVEL_ML")
    try:
        return int(seed_text), float(level_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed and level must be numeric") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("pilot", "train", "val", "test"), required=True)
    parser.add_argument(
        "--pair", type=parse_pair, action="append", default=[],
        metavar="SEED:LEVEL_ML",
        help="explicit seed and level; may be repeated",
    )
    parser.add_argument(
        "--seed", type=int, action="append", default=[],
        help="seed with a sampled level; may be repeated",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.pair and not args.seed:
        parser.error("provide at least one --pair or --seed")
    seeds = [seed for seed, _ in args.pair] + args.seed
    if len(seeds) != len(set(seeds)):
        parser.error("each seed may appear only once in a batch")
    output = args.out.expanduser().resolve()
    if output.exists():
        parser.error("--out must not already exist")
    specs = [sample_spec(seed, args.role, level) for seed, level in args.pair]
    specs.extend(sample_spec(seed, args.role) for seed in args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(output), "frames": len(specs), "role": args.role}))


if __name__ == "__main__":
    main()
