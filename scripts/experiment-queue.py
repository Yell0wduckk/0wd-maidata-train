"""Append independent experiment commands; failures do not block later jobs.

Linux runner: one flock-protected consumer, append is safe during a running job.
Each command owns its stage queue and full-epoch recovery. No data inspection.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time


@contextmanager
def locked(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read(path):
    return json.loads(path.read_text('utf8')) if path.exists() else {'schema': 1, 'jobs': []}


def write(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n', 'utf8')
    temporary.replace(path)


def append(path, job):
    if not isinstance(job.get('argv'), list) or not job['argv'] or not all(isinstance(v, str) for v in job['argv']):
        raise ValueError('argv must be a nonempty string list (no shell)')
    if not isinstance(job.get('id'), str) or not job['id'] or not Path(job['cwd']).is_absolute():
        raise ValueError('job requires an id and absolute cwd')
    if not Path(job['log']).is_absolute():
        raise ValueError('job requires an absolute log path')
    if not isinstance(job.get('env', {}), dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in job.get('env', {}).items()):
        raise ValueError('env must map strings to strings')
    # Preserve provenance supplied by the caller along with exact argv/cwd/env.
    job = dict(job)
    for key in ('status', 'attempts', 'history', 'exit_code'):
        if key in job:
            raise ValueError('job spec must not set runner state')
    with locked(path.with_suffix('.state.lock')):
        state = read(path)
        if any(j['id'] == job['id'] for j in state['jobs']):
            raise ValueError('duplicate job id; use run --retry-failed to resume')
        state['jobs'].append(dict(job, status='pending', attempts=0, history=[]))
        write(path, state)


def run(path, *, retry_failed=False):
    with locked(path.with_suffix('.runner.lock')):
        # A previous runner may have stopped. Its child must be gone before
        # operator restarts this queue; do not start a duplicate training job.
        with locked(path.with_suffix('.state.lock')):
            state = read(path)
            for job in state['jobs']:
                if job['status'] == 'running':
                    pid = job.get('pid')
                    if pid:
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            pass
                        else:
                            raise RuntimeError(f'job {job["id"]} may still be running: PID {pid}')
                    job['status'] = 'interrupted'
                if retry_failed and job['status'] in ('failed', 'interrupted'):
                    job['status'] = 'pending'
            write(path, state)
        while True:
            with locked(path.with_suffix('.state.lock')):
                state = read(path)
                job = next((j for j in state['jobs'] if j['status'] == 'pending'), None)
                if job is None:
                    return 1 if any(j['status'] != 'complete' for j in state['jobs']) else 0
                job.update(status='running', attempts=job['attempts'] + 1, pid=None)
                attempt = {'started': time.time()}
                job['history'].append(attempt)
                # Persist PID while holding state lock; append never races updates.
                try:
                    log = Path(job['log'])
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('a', encoding='utf8') as stream:
                        process = subprocess.Popen(job['argv'], cwd=job['cwd'],
                            env={**os.environ, **job.get('env', {})}, stdout=stream, stderr=subprocess.STDOUT)
                    job['pid'] = process.pid
                    write(path, state)
                except (OSError, ValueError) as error:
                    job.update(status='failed', exit_code=-1, error=str(error))
                    attempt.update(finished=time.time(), exit_code=-1, error=str(error))
                    write(path, state)
                    continue
            code = process.wait()
            with locked(path.with_suffix('.state.lock')):
                state = read(path)
                current = next(j for j in state['jobs'] if j['id'] == job['id'])
                current.update(status='complete' if code == 0 else 'failed', exit_code=code)
                current['history'][-1].update(finished=time.time(), exit_code=code)
                write(path, state)
            # Continue even after a nonzero exit. A failed A does not start its
            # dependent B; the next independent experiment can still run.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    sub = parser.add_subparsers(dest='action', required=True)
    add = sub.add_parser('append')
    add.add_argument('--job', type=Path, required=True, help='JSON: id, argv, cwd, log, optional env/provenance')
    execute = sub.add_parser('run')
    execute.add_argument('--retry-failed', action='store_true', help='one further attempt per failed/interrupted job, restoring its stage queue')
    args = parser.parse_args()
    if args.action == 'append':
        append(args.queue, json.loads(args.job.read_text('utf8')))
    else:
        raise SystemExit(run(args.queue, retry_failed=args.retry_failed))


if __name__ == '__main__':
    main()
