#!/usr/bin/env bash
set -euo pipefail
# The experiment chat starts this only after resources are idle.
# Keep accepted A90/mu/P0 and any other experiment untouched.
PYTHON=/opt/maimai-runtime/bin/python
export PYTHONPATH="$PWD/src"
INPUTS="$PWD/configs/latent-ab-16079-inputs.json"
RUN=/root/autodl-tmp/main-new-prior-decoder-20261009/run
OLD_RUN=${OLD_RUN:-/root/autodl-tmp/main-new-latent-ab-20261008/source-5878f7a541f309d926254aa6461fb5234c41421c/run}
DATA=/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911
A_PLAN=/root/autodl-tmp/maidata-decoder8-20261003/a-plan
ROUTE="$DATA/slide-geometry/route-artifact"
ENCODER60=/root/autodl-tmp/main-new-encoder-pretrain-20261008/pretrain/checkpoints/epoch-060.ckpt
if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite experiment output: $RUN" >&2
  exit 1
fi
mkdir -p "$RUN"
exec > >(tee -a "$RUN/queue.log") 2>&1

# Each producer accepts its new assets through the actual consumer once.
# New complete-profile mu is encoded from encoder60; old audio/P0, Physical
# and geometry are reused without a scan or rebuild.
"$PYTHON" -m maimai_baseline prepare-prior-decoder-data \
  --dataset-manifest "$INPUTS" --epoch-plan-cache "$A_PLAN" \
  --chart-encoder-checkpoint "$ENCODER60" --route-artifact "$ROUTE" \
  --song-context-cache "$OLD_RUN/p0" --output "$RUN/data" --device cuda
"$PYTHON" -m maimai_baseline build-decoder-context-cache \
  --dataset-manifest "$RUN/data/dataset.json" \
  --stage-a-checkpoint "$OLD_RUN/A/last.ckpt" --latent-cache "$RUN/data/mu" \
  --song-context-cache "$RUN/data/p0" --route-artifact "$ROUTE" \
  --output "$RUN/candidate" --device cuda

B_ARGS=(train --stage B --experiment-config configs/train-b-decoder20.json \
  --dataset-manifest "$RUN/data/dataset.json" \
  --stage-a-checkpoint "$OLD_RUN/A/last.ckpt" --latent-cache "$RUN/data/mu" \
  --song-context-cache "$RUN/data/p0" --decoder-context-cache "$RUN/candidate" \
  --epoch-plan-cache "$RUN/data/b-plan" --route-artifact "$ROUTE" --num-workers 12 --device cuda)
"$PYTHON" -m maimai_baseline "${B_ARGS[@]}" --output "$RUN/B-cuda" --cuda-check
"$PYTHON" -m maimai_baseline "${B_ARGS[@]}" --output "$RUN/B"
"$PYTHON" scripts/diagnose-prior-decoder.py \
  --checkpoint "$RUN/B/ar.ckpt" --latent-cache "$RUN/data/mu" \
  --song-context-cache "$RUN/data/p0" --route-artifact "$ROUTE" \
  --bpm-audio-manifest "$DATA/audio/bpm-audio-manifest.json" \
  --physical-path-catalog "$DATA/physical/path-catalog" \
  --output "$RUN/final-samples" --device cuda
echo 'Authorized frozen-A90 Decoder-supervised B20 finished; no automatic continuation.'
