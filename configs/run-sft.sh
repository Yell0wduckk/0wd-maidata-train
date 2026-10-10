#!/usr/bin/env bash
set -euo pipefail
mode="${1:?usage: run-sft.sh teacher-forcing|moment-rollout [sft arguments]}"
shift
case "$mode" in teacher-forcing|moment-rollout) ;; *) printf '%s\n' 'invalid SFT mode' >&2; exit 2 ;; esac
for arg in "$@"; do
  case "$arg" in --sft-mode|--sft-mode=*) printf '%s\n' 'supply SFT mode only as first argument' >&2; exit 2 ;; esac
done
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" -m maimai_baseline sft --sft-mode "$mode" "$@"
