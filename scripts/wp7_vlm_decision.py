"""Development-only WP7 VLM decision freeze CLI."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix.wp7_vlm_decision import main  # noqa: E402
if __name__ == "__main__":
    raise SystemExit(main())
