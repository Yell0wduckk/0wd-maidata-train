#!/usr/bin/env bash
# Fixed pilot261 assets from the completed audio256 experiment; isolated output.
set -euo pipefail
RUN_ROOT=${1:?Usage: bash scripts/launch-mmoe-a90-b50.sh /absolute/experiment-root}
[[ "$RUN_ROOT" = /* ]] || { echo 'experiment root must be absolute' >&2; exit 2; }
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -Sn 65536
PYTHON_BIN=${PYTHON:-python}
if [[ ! -f "$RUN_ROOT/a-plan/manifest.json" ]]; then
  "$PYTHON_BIN" scripts/prepare-query6-plan.py \
    --original-plan /root/autodl-tmp/maidata-pilot261-l3-v3-a90-20260927/plans \
    --output "$RUN_ROOT/a-plan"
fi
exec "$PYTHON_BIN" -u scripts/run-query6-ar.py \
  --dataset-root /root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911 \
  --epoch-plan-cache "$RUN_ROOT/a-plan" \
  --physical-interval-cache /root/autodl-tmp/maidata-full-interval-physical-v15-20260926/interval-cache \
  --song-chart-cache /root/autodl-tmp/maidata-shared1-full-train-20260924/run-input-v2/inputs/song-chart \
  --song-audio-cache /root/autodl-tmp/maidata-shared1-full-train-20260924/run/inputs/song-audio \
  --output "$RUN_ROOT/training" --execute
