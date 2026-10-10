#!/usr/bin/env bash
# Run two independent fresh Stage-II experiments; input assets must already exist.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'Usage: bash %s BASE_CONFIG PREDICTED_OUTPUT GT_OUTPUT [LOG_DIR]\n' "$0" >&2
  exit 2
fi

BASE_CONFIG=$1
PREDICTED_OUTPUT=$2
GT_OUTPUT=$3
LOG_DIR=${4:-${PREDICTED_OUTPUT}.plan-comparison-logs}
if [[ "$PREDICTED_OUTPUT" == "$GT_OUTPUT" ]]; then
  printf 'predicted and gt must have independent output directories\n' >&2
  exit 2
fi

SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON:-python}
mkdir -p -- "$LOG_DIR"

printf 'Starting predicted plan training: %s\n' "$PREDICTED_OUTPUT"
"$PYTHON_BIN" -u -m maidata.decoder_training \
  --config "$BASE_CONFIG" --plan-source predicted --output "$PREDICTED_OUTPUT" \
  2>&1 | tee -a "$LOG_DIR/predicted.log"

printf 'Starting GT plan training: %s\n' "$GT_OUTPUT"
"$PYTHON_BIN" -u -m maidata.decoder_training \
  --config "$BASE_CONFIG" --plan-source gt --output "$GT_OUTPUT" \
  2>&1 | tee -a "$LOG_DIR/gt.log"

printf 'Both plan experiments completed. Logs: %s\n' "$LOG_DIR"
