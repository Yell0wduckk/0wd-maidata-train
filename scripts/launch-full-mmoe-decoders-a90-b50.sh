#!/usr/bin/env bash
# Full song-first A4096 -> own caches -> B16384; only launch when requested.
set -euo pipefail
RUN_ROOT=${1:?Usage: bash scripts/launch-full-mmoe-decoders-a90-b50.sh /absolute/experiment-root}
[[ "$RUN_ROOT" = /* ]] || { echo 'experiment root must be absolute' >&2; exit 2; }
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -Sn 65536
PYTHON_BIN=${PYTHON:-python}
DATASET_ROOT=${DATASET_ROOT:-/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911}
PHYSICAL_INTERVAL_CACHE=${PHYSICAL_INTERVAL_CACHE:-/root/autodl-tmp/maidata-full-interval-physical-v15-20260926/interval-cache}
SONG_CHART_CACHE=${SONG_CHART_CACHE:-/root/autodl-tmp/maidata-shared1-full-train-20260924/run-input-v2/inputs/song-chart}
SONG_AUDIO_CACHE=${SONG_AUDIO_CACHE:-/root/autodl-tmp/maidata-shared1-full-train-20260924/run/inputs/song-audio}
if [[ ! -f "$RUN_ROOT/a-plan/manifest.json" ]]; then
  "$PYTHON_BIN" scripts/prepare-full-mmoe-decoders-plan.py \
    --dataset-root "$DATASET_ROOT" --output "$RUN_ROOT/a-plan"
fi
exec "$PYTHON_BIN" -u scripts/run-full-mmoe-decoders.py \
  --dataset-root "$DATASET_ROOT" --epoch-plan-cache "$RUN_ROOT/a-plan" \
  --physical-interval-cache "$PHYSICAL_INTERVAL_CACHE" \
  --song-chart-cache "$SONG_CHART_CACHE" --song-audio-cache "$SONG_AUDIO_CACHE" \
  --output "$RUN_ROOT/training" --execute
