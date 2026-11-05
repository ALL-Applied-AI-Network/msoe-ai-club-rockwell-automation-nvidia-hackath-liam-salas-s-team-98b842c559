#!/usr/bin/env bash
# Run the SAM evaluation on the Spark from any laptop (macOS, Linux, or Git Bash on Windows),
# then copy the results back to outputs/sam/<tag>/ here. Run from the repo root:
#
#     SPARK=user@spark-host bash scripts/spark_sam.sh                 # full video-5 evaluation
#     SPARK=user@spark-host bash scripts/spark_sam.sh --check         # is the Spark ready?
#     SPARK=user@spark-host bash scripts/spark_sam.sh --model small --vials 8 20
#
# Arguments after the script name go to scripts/sam_eval.py. The Spark side uses its own clone
# (SPARK_REPO, default ~/fluidmix-sam) so it never touches a teammate's working copy. It updates
# that clone to BRANCH (default main) with a fast-forward only and stops if the clone has edits.
# Setup must have been run once on the Spark: bash scripts/setup_sam_spark.sh (docs/sam.md).
set -euo pipefail
: "${SPARK:?set SPARK=user@spark-host (docs/sam.md)}"
SPARK_REPO=${SPARK_REPO:-fluidmix-sam}
BRANCH=${BRANCH:-main}
TAG=${TAG:-$(date +%Y%m%d-%H%M%S)-$(whoami | tr -cd '[:alnum:]_-')}
ORIGIN=$(git remote get-url origin)

ssh "$SPARK" bash -s -- "$SPARK_REPO" "$BRANCH" "$ORIGIN" "$TAG" "${SPARK_DATA:-}" "$@" <<'REMOTE'
set -euo pipefail
dir=$1 branch=$2 origin=$3 tag=$4 data=$5
shift 5
if [ ! -d "$dir/.git" ]; then
  echo "cloning $origin into ~/$dir"
  git clone -q "$origin" "$dir"
fi
cd "$dir"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "~/$dir has local edits; commit or discard them on the Spark first" >&2
  exit 1
fi
git fetch -q origin "$branch"
git checkout -q "$branch"
git merge -q --ff-only "origin/$branch"
echo "Spark clone at $(git rev-parse --short HEAD) ($branch)"
if [ ! -x .venv-sam/bin/python ]; then
  echo "SAM is not set up in ~/$dir: run  bash scripts/setup_sam_spark.sh  there once (docs/sam.md)" >&2
  exit 1
fi
[ -n "$data" ] && export FLUIDMIX_DATA="$data"
.venv-sam/bin/python scripts/sam_eval.py --tag "$tag" "$@"
REMOTE

case " $* " in *" --check "*) exit 0 ;; esac
mkdir -p outputs/sam
scp -rq "$SPARK:$SPARK_REPO/outputs/sam/$TAG" outputs/sam/
echo "results copied to outputs/sam/$TAG (summary.txt, CSVs, *_overlays.png)"
