"""Launch the authorized dense queue using a specified isolated source directory."""
import argparse
import fcntl
import json
import os
import pathlib
import resource
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path('/root/autodl-tmp/maidata-decoder8-song-dense-20261003')
    python = '/opt/maimai-runtime/bin/python'
    command = [python, '-u', 'scripts/run-ar-decoder6.py',
        '--dataset-root', '/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911',
        '--epoch-plan-cache', '/root/autodl-tmp/maidata-decoder8-20261003/a-plan',
        '--physical-interval-cache', '/root/autodl-tmp/maidata-full-interval-physical-v15-20260926/interval-cache',
        '--song-chart-cache', '/root/autodl-tmp/maidata-shared1-full-train-20260924/run-input-v2/inputs/song-chart',
        '--song-audio-cache', '/root/autodl-tmp/maidata-shared1-full-train-20260924/run/inputs/song-audio',
        '--output', str(root / 'training'), '--a-cuda-checked', '--execute']
    env = dict(os.environ, PYTHONPATH=str(args.source / 'src'),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    with (root / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / 'launcher.pid').write_text(str(os.getpid()) + '\n')
        started = time.time()
        child = subprocess.Popen(command, cwd=args.source, env=env)
        result = dict(launcher_pid=os.getpid(), queue_pid=child.pid,
            source=str(args.source), command=command, started_unix=started)
        (root / 'launcher-start.json').write_text(json.dumps(result, indent=2))
        result.update(exit_code=child.wait(), ended_unix=time.time())
        (root / 'launcher-exit.json').write_text(json.dumps(result, indent=2))
        raise SystemExit(result['exit_code'])


if __name__ == '__main__':
    main()
