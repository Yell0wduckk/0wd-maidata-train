#!/usr/bin/env bash
set -euo pipefail
# Start in the fixed uploaded source only when the experiment chat finds resources idle.
PYTHON=${PYTHON:-/opt/maimai-runtime/bin/python}
export PYTHONPATH="$PWD/src"
INPUTS=${INPUTS:-$PWD/configs/latent-ab-16079-inputs.json}
RUN=/root/autodl-tmp/main-new-encoder-audio-20261009/run
DATA=/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911
ROUTE="$DATA/slide-geometry/route-artifact"
A_PLAN=/root/autodl-tmp/maidata-decoder8-20261003/a-plan
EXTERNAL=/root/autodl-tmp/main-new-encoder-audio-20261009/external
TEACHER_SOURCE=${TEACHER_SOURCE:-$EXTERNAL/source-3d0bc68}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-$EXTERNAL/A-last.ckpt}
EXPORT_WRAPPER=${EXPORT_WRAPPER:-$EXTERNAL/export_native_audio.py}
if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite experiment output: $RUN" >&2
  exit 1
fi
mkdir -p "$RUN"
exec > >(tee -a "$RUN/queue.log") 2>&1

# Reuse a completed compatible prepared data manifest when explicitly supplied.
# Otherwise the producer generates only new window/index/plan assets.
if [[ -n "${PREPARED_DATA:-}" ]]; then
  PREP="$PREPARED_DATA"
else
  "$PYTHON" -m maimai_baseline prepare-encoder-audio-data \
    --dataset-manifest "$INPUTS" --epoch-plan-cache "$A_PLAN" --output "$RUN/data"
  PREP="$RUN/data/dataset.json"
fi
# Existing prepared profiles/plans remain in place. Only a missing export request
# list is generated, using the authoritative mapped paths in their manifest.
PREP_PATHS=$("$PYTHON" - "$PREP" "$RUN/export_requests.json" <<'PY'
import sys
from types import SimpleNamespace
from maimai_baseline.data.encoder_audio import resolve_paths, ensure_export_requests
paths = resolve_paths(SimpleNamespace(dataset_manifest=sys.argv[1]))
print(paths['epoch_plan_cache'])
print(paths['whole_song_plan'])
print(ensure_export_requests(sys.argv[1], sys.argv[2]))
PY
)
mapfile -t PREP_PATHS <<< "$PREP_PATHS"
A_DERIVED=${PREP_PATHS[0]}
B_DERIVED=${PREP_PATHS[1]}
EXPORT_REQUESTS=${PREP_PATHS[2]}
"$PYTHON" "$EXPORT_WRAPPER" --source-snapshot "$TEACHER_SOURCE" \
  --checkpoint "$TEACHER_CHECKPOINT" --route-artifact "$ROUTE" \
  --cache-writer "$PWD/src/maimai_baseline/data/teacher_audio.py" \
  --requests "$EXPORT_REQUESTS" --output "$RUN/teacher-audio" --device cuda

E_ARGS=(train-chart-encoder --experiment-config configs/train-chart-encoder.json \
  --dataset-manifest "$PREP" --epoch-plan-cache "$A_DERIVED" \
  --teacher-audio-cache "$RUN/teacher-audio" --num-workers 12 --device cuda)
"$PYTHON" -m maimai_baseline "${E_ARGS[@]}" --output "$RUN/encoder-cuda" --cuda-check
"$PYTHON" -m maimai_baseline "${E_ARGS[@]}" --output "$RUN/encoder"
"$PYTHON" -m maimai_baseline build-latent-cache --dataset-manifest "$PREP" \
  --epoch-plan-cache "$A_DERIVED" --chart-encoder-checkpoint "$RUN/encoder/encoder.pt" \
  --output "$RUN/mu" --device cuda

A_ARGS=(train --stage A --experiment-config configs/train-a.json \
  --chart-style native-audio-frozen-mu128-v1 --dataset-manifest "$PREP" \
  --epoch-plan-cache "$A_DERIVED" --latent-cache "$RUN/mu" \
  --chart-encoder-checkpoint "$RUN/encoder/encoder.pt" \
  --teacher-audio-cache "$RUN/teacher-audio" --num-workers 12 --device cuda)
"$PYTHON" -m maimai_baseline "${A_ARGS[@]}" --output "$RUN/A-cuda" --cuda-check
"$PYTHON" -m maimai_baseline "${A_ARGS[@]}" --output "$RUN/A"
"$PYTHON" -m maimai_baseline build-song-context --dataset-manifest "$PREP" \
  --stage-a-checkpoint "$RUN/A/last.ckpt" --latent-cache "$RUN/mu" \
  --route-artifact "$ROUTE" \
  --teacher-audio-cache "$RUN/teacher-audio" --output "$RUN/p0" --device cuda
"$PYTHON" -m maimai_baseline build-decoder-context-cache --dataset-manifest "$PREP" \
  --stage-a-checkpoint "$RUN/A/last.ckpt" --latent-cache "$RUN/mu" \
  --song-context-cache "$RUN/p0" --route-artifact "$ROUTE" \
  --teacher-audio-cache "$RUN/teacher-audio" --output "$RUN/candidate" --device cuda
B_ARGS=(train --stage B --experiment-config configs/train-b-decoder20.json \
  --chart-style native-audio-frozen-mu128-v1 --dataset-manifest "$PREP" \
  --stage-a-checkpoint "$RUN/A/last.ckpt" --latent-cache "$RUN/mu" \
  --song-context-cache "$RUN/p0" --decoder-context-cache "$RUN/candidate" \
  --teacher-audio-cache "$RUN/teacher-audio" --epoch-plan-cache "$B_DERIVED" \
  --route-artifact "$ROUTE" --num-workers 12 --device cuda)
"$PYTHON" -m maimai_baseline "${B_ARGS[@]}" --output "$RUN/B-cuda" --cuda-check
"$PYTHON" -m maimai_baseline "${B_ARGS[@]}" --output "$RUN/B"
"$PYTHON" scripts/diagnose-prior-decoder.py --checkpoint "$RUN/B/ar.ckpt" \
  --latent-cache "$RUN/mu" --song-context-cache "$RUN/p0" --route-artifact "$ROUTE" \
  --teacher-audio-cache "$RUN/teacher-audio" --bpm-audio-manifest "$DATA/audio/bpm-audio-manifest.json" \
  --physical-path-catalog "$DATA/physical/path-catalog" --output "$RUN/final-samples" --device cuda
echo 'Authorized encoder60/A90/B20 finished; no automatic continuation.'
