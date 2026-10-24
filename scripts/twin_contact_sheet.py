"""Tile rgb/*.png above colourised semantic/*.png for a run directory."""
import sys, json, cv2, numpy as np
from pathlib import Path
d = Path(sys.argv[1]); frames = sorted((d / "rgb").glob("*.png"))[: int(sys.argv[3]) if len(sys.argv) > 3 else None]
rng = np.random.default_rng(0); lut = rng.integers(40, 255, (65536, 3), dtype=np.uint8); lut[0] = 0
cols = []
for f in frames:
    rgb = cv2.imread(str(f)); m = cv2.imread(str(d / "semantic" / f.name), cv2.IMREAD_UNCHANGED)
    sem = lut[m] if m is not None else np.zeros_like(rgb)
    cv2.putText(rgb, f.stem, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cols.append(np.vstack([rgb, sem]))
sheet = np.hstack(cols); s = 1600 / sheet.shape[1]
cv2.imwrite(sys.argv[2], cv2.resize(sheet, None, fx=min(1, s), fy=min(1, s)))
