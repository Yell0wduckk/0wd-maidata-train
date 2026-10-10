#!/usr/bin/env bash
set -euo pipefail
# Invoke only after the user explicitly selects a draft checkpoint in the experiment chat.
if [[ $# != 1 || ! -f "$1" ]]; then
  echo 'Usage: continue-audio-draft-beat-A-16079.sh /explicitly-selected/draft-checkpoint' >&2
  exit 2
fi
SELECTED_DRAFT=$1
PYTHON=${PYTHON:-/opt/maimai-runtime/bin/python}
export PYTHONPATH="$PWD/src"
RUN=/root/autodl-tmp/main-new-audio-draft-beat-20261009/run
PREP="$RUN/data/dataset.json"
for target in "$RUN/draft-cache" "$RUN/A-cuda" "$RUN/A" "$RUN/final-samples"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite downstream output: $target" >&2
    exit 1
  fi
done
TEACHER_AUDIO_CACHE=${TEACHER_AUDIO_CACHE:-$("$PYTHON" - "$RUN/draft-inputs.json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text('utf8'))['teacher_audio_cache'])
PY
)}
PREP_PATHS=$("$PYTHON" - "$PREP" <<'PY'
import json, sys
from pathlib import Path
manifest = Path(sys.argv[1]).resolve()
document = json.loads(manifest.read_text('utf-8-sig'))
for name in ('epoch_plan_cache', 'route_artifact'):
    value = Path(document['inputs'][name])
    print((value if value.is_absolute() else manifest.parent / value).resolve())
PY
)
mapfile -t PREP_PATHS <<< "$PREP_PATHS"
A_PLAN=${PREP_PATHS[0]}
ROUTE=${PREP_PATHS[1]}
exec > >(tee -a "$RUN/selected-A.log") 2>&1
"$PYTHON" -m maimai_baseline build-draft-cache --dataset-manifest "$PREP" \
  --epoch-plan-cache "$A_PLAN" --draft-checkpoint "$SELECTED_DRAFT" \
  --teacher-audio-cache "$TEACHER_AUDIO_CACHE" --output "$RUN/draft-cache" --device cuda
A_ARGS=(train --stage A --experiment-config configs/train-a.json \
  --chart-style native-audio-draft128-beat-v1 --dataset-manifest "$PREP" \
  --epoch-plan-cache "$A_PLAN" --draft-cache "$RUN/draft-cache" \
  --draft-checkpoint "$SELECTED_DRAFT" --teacher-audio-cache "$TEACHER_AUDIO_CACHE" \
  --num-workers 12 --device cuda)
"$PYTHON" -m maimai_baseline "${A_ARGS[@]}" --output "$RUN/A-cuda" --cuda-check
"$PYTHON" -m maimai_baseline "${A_ARGS[@]}" --output "$RUN/A"
"$PYTHON" scripts/diagnose-audio-draft.py --dataset-manifest "$PREP" \
  --epoch-plan-cache "$A_PLAN" --checkpoint "$RUN/A/last.ckpt" \
  --draft-cache "$RUN/draft-cache" --teacher-audio-cache "$TEACHER_AUDIO_CACHE" \
  --route-artifact "$ROUTE" --output "$RUN/final-samples" --device cuda
echo 'Selected draft cache and A90 finished; no B or automatic continuation.'
