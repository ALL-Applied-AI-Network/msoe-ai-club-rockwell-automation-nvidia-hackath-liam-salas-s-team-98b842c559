"""Run or preflight one exact pinned model against a frozen bake-off plan."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.rca_bench.model_bakeoff_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
