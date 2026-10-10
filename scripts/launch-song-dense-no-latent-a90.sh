#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=${1:?Usage: bash scripts/launch-song-dense-no-latent-a90.sh /absolute/experiment-root [--resume /absolute/last.ckpt]}
shift
[[ "$RUN_ROOT" = /* ]] || { echo 'experiment root must be absolute' >&2; exit 2; }
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -Sn 65536
exec "${PYTHON:-python}" -u scripts/run-song-dense-no-latent-a90.py "$RUN_ROOT" "$@"
