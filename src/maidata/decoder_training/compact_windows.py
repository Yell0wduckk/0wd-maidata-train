"""Legacy continuous-time v1→v2 compaction; never a tick384 CE migration."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import torch

from .intervals import compact_labels
from .window_cache import tensor_pack, tensor_unpack

SCHEMA = 'maidata.decoder-window-tensors.v2'


def initialize():
    torch.set_num_threads(1)


def convert_window(task):
    source, destination, key, row = task
    value = tensor_unpack(torch.load(Path(source)/row['path'], map_location='cpu', weights_only=True))
    before = sum(g['allowed'].numel() for g in value['labels']['groups'].values() if 'lower' in g)
    value['labels'] = compact_labels(value['labels'])
    after = sum(g['allowed'].numel() for g in value['labels']['groups'].values() if 'lower' in g)
    path = Path(destination)/row['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    payload = tensor_pack(value)
    torch.save(payload, temporary)
    restored = torch.load(temporary, map_location='cpu', weights_only=True)
    # Validate only the newly written asset, once, offline.
    if (payload['layout'] != restored['layout'] or
            any(not torch.equal(v, restored['buffers'][k]) for k, v in payload['buffers'].items())):
        raise ValueError('compact tensor storage roundtrip differs')
    temporary.replace(path)
    from hashlib import sha256
    return key, dict(row, bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest()), before, after


def convert(source, destination, *, workers=12):
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('compact output must be new or empty')
    original = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if original['schema'] not in ('maidata.decoder-window-tensors.v1', SCHEMA):
        raise ValueError('legacy compaction accepts only continuous-time v1/v2 windows')
    destination.mkdir(parents=True, exist_ok=True)
    result = dict(original, schema=SCHEMA, complete=False, windows={},
                  time_support='merged-equal-target-and-physical-intervals')
    manifest = destination/'manifest.json'
    manifest.write_text(json.dumps(result), encoding='utf-8')
    tasks = [(str(source), str(destination), key, row) for key, row in original['windows'].items()]
    before = after = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        for n, (key, row, old, new) in enumerate(pool.map(convert_window, tasks, chunksize=8), 1):
            result['windows'][key] = row
            before += old
            after += new
            if n % 256 == 0 or n == len(tasks):
                print(json.dumps(dict(windows=n, total=len(tasks))), flush=True)
    result.update(complete=True, time_cells_before=before, time_intervals_after=after)
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    result = convert(args.source, args.output, workers=args.workers)
    print(json.dumps(dict(complete=True, windows=len(result['windows']),
        bytes=sum(r['bytes'] for r in result['windows'].values()),
        time_cells_before=result['time_cells_before'], time_intervals_after=result['time_intervals_after'])), flush=True)


if __name__ == '__main__':
    main()
