#!/usr/bin/env bash
# One-time SAM 2.1 setup on the DGX Spark / GB10 (ARM64 + CUDA). Run on the Spark, from the repo root:
#
#     bash scripts/setup_sam_spark.sh
#
# Creates .venv-sam (ignored by Git), installs CUDA PyTorch and Meta's sam2 package, downloads the
# SAM 2.1 weights to <data root>/models/sam2.1 (ignored by Git), records their SHA-256, then runs
# `scripts/sam_eval.py --check`. Safe to re-run: finished steps are skipped.
#
# Overrides (environment variables):
#   SAM_MODELS="tiny small"   checkpoints to download: tiny small base_plus large
#   TORCH_INDEX=URL           PyTorch wheel index (default: CUDA 13.0 wheels, which cover GB10)
#   USE_SYSTEM_TORCH=1        reuse a PyTorch already installed on the machine (e.g. an NGC container)
#   SAM2_REF=<commit or tag>  pin the sam2 source (default: its main branch; the run records the version)
#   FLUIDMIX_DATA=PATH        data root, as everywhere else in the repo (docs/data.md)
# Untested on the Spark when written; see docs/sam.md if a step fails.
set -euo pipefail
cd "$(dirname "$0")/.."

SAM_MODELS=${SAM_MODELS:-"tiny small"}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
SAM2_REF=${SAM2_REF:-main}
URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824

echo "== machine"
uname -m
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || { echo "nvidia-smi failed: is this the Spark, and is the GPU visible?" >&2; exit 1; }
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found; install it (sudo apt install ffmpeg) or ask an admin" >&2

echo "== python environment (.venv-sam)"
if [ ! -x .venv-sam/bin/python ]; then
  if [ "${USE_SYSTEM_TORCH:-0}" = 1 ]; then python3 -m venv --system-site-packages .venv-sam
  else python3 -m venv .venv-sam; fi
fi
PY=.venv-sam/bin/python
$PY -m pip install -q --upgrade pip
if ! $PY -c "import torch" 2>/dev/null; then
  $PY -m pip install torch torchvision --index-url "$TORCH_INDEX"
fi
$PY - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the GPU. Try another TORCH_INDEX, or USE_SYSTEM_TORCH=1 inside "
                     "an NVIDIA PyTorch container (docs/sam.md).")
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.ones(1024, 1024, device="cuda")
assert float((x @ x)[0, 0]) == 1024.0, "GPU matmul gave a wrong answer"
print("GPU matmul ok")
EOF
$PY -m pip install -q numpy opencv-python-headless
if ! $PY -c "import sam2" 2>/dev/null; then
  # The optional CUDA extension only fills small mask holes; skipping it avoids a compile step.
  SAM2_BUILD_CUDA=0 $PY -m pip install "git+https://github.com/facebookresearch/sam2.git@${SAM2_REF}"
fi

echo "== weights"
WEIGHTS=$($PY -c "import sys; sys.path.insert(0, 'src'); from fluidmix.paths import data_root; print(data_root() / 'models/sam2.1')")
mkdir -p "$WEIGHTS"
for m in $SAM_MODELS; do
  f="sam2.1_hiera_${m}.pt"
  if [ ! -s "$WEIGHTS/$f" ]; then
    curl -fL --retry 3 -o "$WEIGHTS/$f.part" "$URL/$f"
    mv "$WEIGHTS/$f.part" "$WEIGHTS/$f"
  fi
done
(cd "$WEIGHTS" && sha256sum sam2.1_hiera_*.pt | tee SHA256SUMS)

echo "== check"
for m in $SAM_MODELS; do $PY scripts/sam_eval.py --check --model "$m"; done
echo "SAM is ready. Run: .venv-sam/bin/python scripts/sam_eval.py   (docs/sam.md)"
