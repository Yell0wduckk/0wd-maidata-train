#!/usr/bin/env bash
set -euo pipefail
# Run from the uploaded independent source root, only when resources are idle.
# This executes only the explicitly authorized encoder60 -> A90 -> B50 work.
PYTHON=/opt/maimai-runtime/bin/python
export PYTHONPATH="$PWD/src"
INPUTS="$PWD/configs/latent-ab-16079-inputs.json"
RUN="$PWD/run"
DATA=/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911
GT=/root/autodl-tmp/maidata-shared1-full-train-20260924/run-input-v2/inputs/song-chart
SONG_AUDIO=/root/autodl-tmp/maidata-shared1-full-train-20260924/run/inputs/song-audio
A_PLAN=/root/autodl-tmp/maidata-decoder8-20261003/a-plan
ENCODER60=/root/autodl-tmp/main-new-encoder-pretrain-20261008/pretrain/checkpoints/epoch-060.ckpt
ROUTE="$DATA/slide-geometry/route-artifact"
if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite an existing experiment: $RUN" >&2
  exit 1
fi
mkdir -p "$RUN"
exec > >(tee -a "$RUN/queue.log") 2>&1

# Import validates the independent checkpoint and requires internal epoch=59.
"$PYTHON" -m maimai_baseline build-latent-cache \
  --dataset-manifest "$INPUTS" --epoch-plan-cache "$A_PLAN" \
  --chart-encoder-checkpoint "$ENCODER60" --output "$RUN/mu" --device cuda
"$PYTHON" -m maimai_baseline build-b-plan \
  --dataset-manifest "$INPUTS" --epoch-plan-cache "$A_PLAN" --output "$RUN/b-plan"

A_ARGS=(train --stage A --experiment-config configs/train-a.json \
  --dataset-manifest "$INPUTS" --epoch-plan-cache "$A_PLAN" --latent-cache "$RUN/mu")
"$PYTHON" -m maimai_baseline "${A_ARGS[@]}" --output "$RUN/A-cuda" --cuda-check
"$PYTHON" -m maimai_baseline "${A_ARGS[@]}" --output "$RUN/A"

"$PYTHON" -m maimai_baseline build-song-context \
  --stage-a-checkpoint "$RUN/A/last.ckpt" --latent-cache "$RUN/mu" \
  --song-chart-cache "$GT" --song-audio-cache "$SONG_AUDIO" \
  --bpm-audio-manifest "$DATA/audio/bpm-audio-manifest.json" \
  --interpolation-geometry "$DATA/geometry" --route-artifact "$ROUTE" \
  --output "$RUN/p0" --device cuda
B_ARGS=(train --stage B --experiment-config configs/train-b.json \
  --stage-a-checkpoint "$RUN/A/last.ckpt" --latent-cache "$RUN/mu" \
  --song-context-cache "$RUN/p0" --epoch-plan-cache "$RUN/b-plan" --route-artifact "$ROUTE")
"$PYTHON" -m maimai_baseline "${B_ARGS[@]}" --output "$RUN/B-cuda" --cuda-check
"$PYTHON" -m maimai_baseline "${B_ARGS[@]}" --output "$RUN/B"
echo 'Authorized encoder60 -> A90 -> B50 experiment finished.'
