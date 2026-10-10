#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON_BIN:-python3}"
action="${1:-}"
if [[ -z "${action}" ]]; then
  printf 'usage: %s preflight|single-batch|smoke|benchmark|qualify|fit|train|resume PATH_ARGS...\n' "$0" >&2
  exit 2
fi
shift

export PYTHONPATH="${project}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

arguments=("$@")
common=()
schedule=()
output=""
resume=""
index=0
while (( index < ${#arguments[@]} )); do
  value="${arguments[index]}"
  case "${value}" in
    --output)
      (( index + 1 < ${#arguments[@]} )) || { printf '%s\n' 'missing value for --output' >&2; exit 2; }
      output="${arguments[index + 1]}"
      index=$((index + 2))
      ;;
    --resume)
      (( index + 1 < ${#arguments[@]} )) || { printf '%s\n' 'missing value for --resume' >&2; exit 2; }
      resume="${arguments[index + 1]}"
      index=$((index + 2))
      ;;
    --scheduler-epochs|--stop-after-epochs)
      (( index + 1 < ${#arguments[@]} )) || { printf 'missing value for %s\n' "${value}" >&2; exit 2; }
      schedule+=("${value}" "${arguments[index + 1]}")
      index=$((index + 2))
      ;;
    *)
      common+=("${value}")
      index=$((index + 1))
      ;;
  esac
done

[[ -n "${output}" ]] || { printf '%s\n' '--output is required' >&2; exit 2; }
if [[ "${action}" != "resume" && -n "${resume}" ]]; then
  printf '%s\n' '--resume is valid only with resume' >&2
  exit 2
fi

run_preflight() {
  "${python}" -m maimai_baseline preflight "${common[@]}" "${schedule[@]}" --output "${output}"
}

run_benchmark() {
  local target="$1"
  local steps="$2"
  local warmup="$3"
  "${python}" -m maimai_baseline benchmark "${common[@]}" \
    --output "${target}" --steps "${steps}" --benchmark-warmup-steps "${warmup}"
}

run_qualify() {
  local root="${output}.qualification"
  run_benchmark "${root}/single-batch-forward-backward" 1 0
  run_benchmark "${root}/short-smoke" 8 2
  run_benchmark "${root}/memory-throughput-benchmark" 50 10
}

case "${action}" in
  preflight)
    run_preflight
    ;;
  single-batch)
    run_benchmark "${output}" 1 0
    ;;
  smoke)
    run_benchmark "${output}" 8 2
    ;;
  benchmark)
    run_benchmark "${output}" 50 10
    ;;
  qualify)
    run_qualify
    ;;
  fit)
    "${python}" -m maimai_baseline fit "${common[@]}" "${schedule[@]}" --output "${output}"
    ;;
  train)
    "${python}" -m maimai_baseline fit "${common[@]}" "${schedule[@]}" --output "${output}"
    ;;
  resume)
    [[ -n "${resume}" ]] || { printf '%s\n' 'resume requires --resume CHECKPOINT' >&2; exit 2; }
    "${python}" -m maimai_baseline fit "${common[@]}" \
      "${schedule[@]}" --output "${output}" --resume "${resume}"
    ;;
  *)
    printf 'unknown action: %s\n' "${action}" >&2
    exit 2
    ;;
esac
