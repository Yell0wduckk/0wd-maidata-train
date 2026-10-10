#!/usr/bin/env bash
# Eight chart decoder layers; Pilot261 song-first A4096 -> own caches -> unchanged B16384; isolated output.
set -euo pipefail
RUN_ROOT=${1:?Usage: bash scripts/launch-decoder8-a90-b50.sh /absolute/experiment-root}
[[ "$RUN_ROOT" = /* ]] || { echo 'experiment root must be absolute' >&2; exit 2; }
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -Sn 65536
PYTHON_BIN=${PYTHON:-python}
ORIGINAL_PLAN=${ORIGINAL_PLAN:-/root/autodl-tmp/maidata-pilot261-song-first-a90-b50-0c7b137-20260930/a-plan}
if [[ ! -f "$RUN_ROOT/a-plan/manifest.json" ]]; then
  "$PYTHON_BIN" scripts/prepare-query6-plan.py \
    --original-plan "$ORIGINAL_PLAN" \
    --dataset-root /root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911 \
    --output "$RUN_ROOT/a-plan"
fi
exec "$PYTHON_BIN" -u scripts/run-ar-decoder6.py \
  --dataset-root /root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911 \
  --epoch-plan-cache "$RUN_ROOT/a-plan" \
  --physical-interval-cache /root/autodl-tmp/maidata-full-interval-physical-v15-20260926/interval-cache \
  --song-chart-cache /root/autodl-tmp/maidata-shared1-full-train-20260924/run-input-v2/inputs/song-chart \
  --song-audio-cache /root/autodl-tmp/maidata-shared1-full-train-20260924/run/inputs/song-audio \
  --output "$RUN_ROOT/training" --execute
