#!/usr/bin/env bash
# Install Isaac Sim 6.0.1 (pip) on a plain Brev GPU VM. Idempotent; logs to ~/isaac/setup.log.
set -euo pipefail
mkdir -p ~/isaac && cd ~/isaac
exec > >(tee -a setup.log) 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
[ -x .venv/bin/python ] || uv venv --python-preference only-managed --python 3.12 .venv
uv pip install --python .venv/bin/python "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
# Headless RTX on a bare VM needs the Vulkan loader and a few X/GL libs even without a display.
sudo apt-get update -qq && sudo apt-get install -y -qq libvulkan1 libglu1-mesa libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 >/dev/null || true
echo SETUP_DONE
