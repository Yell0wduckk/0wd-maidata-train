#!/usr/bin/env bash
set -euo pipefail
# Run from the fixed uploaded source after the experiment chat confirms resources idle.
PYTHON=${PYTHON:-/opt/maimai-runtime/bin/python}
export PYTHONPATH="$PWD/src"
INPUTS=${INPUTS:-$PWD/configs/latent-ab-16079-inputs.json}
RUN=/root/autodl-tmp/main-new-audio-draft-beat-20261009/run
EXTERNAL=${EXTERNAL:-/root/autodl-tmp/main-new-encoder-audio-20261009/external}
TEACHER_SOURCE=${TEACHER_SOURCE:-$EXTERNAL/source-3d0bc68}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-$EXTERNAL/A-last.ckpt}
if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite experiment output: $RUN" >&2
  exit 1
fi
mkdir -p "$RUN"
exec > >(tee -a "$RUN/queue.log") 2>&1

# A compatible prepared profile may be reused; only the canonical plan is new.
SOURCE_DATA=${PREPARED_DATA:-$INPUTS}
VALIDATION_SOURCE=${VALIDATION_SOURCE:-$INPUTS}
SOURCE_PLAN=${SOURCE_PLAN:-$("$PYTHON" - "$SOURCE_DATA" <<'PY'
import sys
from types import SimpleNamespace
from maimai_baseline.data.encoder_audio import resolve_paths
print(resolve_paths(SimpleNamespace(dataset_manifest=sys.argv[1]))['epoch_plan_cache'])
PY
)}
"$PYTHON" -m maimai_baseline prepare-audio-draft-data \
  --dataset-manifest "$SOURCE_DATA" --epoch-plan-cache "$SOURCE_PLAN" \
  --validation-source-manifest "$VALIDATION_SOURCE" --output "$RUN/data"
PREP="$RUN/data/dataset.json"
VALIDATION="$RUN/data/validation/dataset.json"
PREP_PATHS=$("$PYTHON" - "$PREP" <<'PY'
import json, sys
from pathlib import Path
manifest = Path(sys.argv[1]).resolve()
document = json.loads(manifest.read_text('utf-8-sig'))
for name in ('epoch_plan_cache', 'draft_epoch_plan_cache', 'route_artifact'):
    value = Path(document['inputs'][name])
    print((value if value.is_absolute() else manifest.parent / value).resolve())
print(document['export_requests'])
PY
)
mapfile -t PREP_PATHS <<< "$PREP_PATHS"
A_PLAN=${PREP_PATHS[0]}
DRAFT_PLAN=${PREP_PATHS[1]}
ROUTE=${PREP_PATHS[2]}
EXPORT_REQUESTS=${PREP_PATHS[3]}

# Metadata-only coverage check; the producer reads only changed audio variants.
if [[ -n "${TEACHER_AUDIO_CACHE:-}" ]]; then
  MISSING=$("$PYTHON" - "$EXPORT_REQUESTS" "$TEACHER_AUDIO_CACHE" <<'PY'
import sys
from maimai_baseline.data.audio_draft import missing_teacher_requests
request = missing_teacher_requests(sys.argv[1], sys.argv[2])
print(sum(len(row['starts']) for row in request['variants'].values()))
PY
)
else
  MISSING=1
fi
if [[ "$MISSING" != 0 ]]; then
  REUSE=()
  if [[ -n "${TEACHER_AUDIO_CACHE:-}" ]]; then REUSE=(--reuse-cache "$TEACHER_AUDIO_CACHE"); fi
  "$PYTHON" scripts/export-draft-native-audio.py --source-snapshot "$TEACHER_SOURCE" \
    --checkpoint "$TEACHER_CHECKPOINT" --route-artifact "$ROUTE" \
    --cache-writer "$PWD/src/maimai_baseline/data/teacher_audio.py" \
    --requests "$EXPORT_REQUESTS" --output "$RUN/teacher-audio" --device cuda \
    "${REUSE[@]}" | tee "$RUN/native-export.jsonl"
  TEACHER_AUDIO_CACHE=$("$PYTHON" - "$RUN/native-export.jsonl" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])['output'])
PY
)
fi

DRAFT_ARGS=(train-audio-draft --experiment-config configs/train-audio-draft.json \
  --dataset-manifest "$PREP" --epoch-plan-cache "$DRAFT_PLAN" \
  --validation-dataset-manifest "$VALIDATION" \
  --teacher-audio-cache "$TEACHER_AUDIO_CACHE" --num-workers 12 --device cuda)
"$PYTHON" -m maimai_baseline "${DRAFT_ARGS[@]}" --output "$RUN/draft-cuda" --cuda-check
"$PYTHON" -m maimai_baseline "${DRAFT_ARGS[@]}" --output "$RUN/draft"
# Save the actual cache location for the later explicit selection, never a choice.
"$PYTHON" - "$RUN" "$TEACHER_AUDIO_CACHE" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
(root / 'draft-inputs.json').write_text(json.dumps(
    {'teacher_audio_cache': sys.argv[2]}, indent=2), encoding='utf8')
PY
echo 'Draft60 and held-out validation finished. STOP: user must select a checkpoint before cache/A.'
