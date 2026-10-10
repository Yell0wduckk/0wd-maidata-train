"""Read cached chart conditions; use profile metadata only when genuinely absent."""
from pathlib import Path
import json


def load_level_conditions(dataset_manifest, cache, stage_a=None):
    wanted = {str(row['record_id']) for row in cache.records}
    levels = {str(row['record_id']): float(row['level_value'])
              for row in cache.records if 'level_value' in row}
    source_kind = 'cache-records'
    if wanted - levels.keys():
        if dataset_manifest is None:
            raise ValueError('cached records lack levels; provide dataset metadata')
        path = Path(dataset_manifest).resolve()
        manifest = json.loads(path.read_text(encoding='utf-8-sig'))
        profile = (path.parent / manifest['inputs']['dataset']).resolve()
        with (profile / 'manifest.jsonl').open(encoding='utf-8-sig') as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    key = str(record['record_id'])
                    if key in wanted and key not in levels:
                        levels[key] = float(record['level_value'])
        source_kind = 'profile-metadata'
    if wanted - levels.keys():
        raise ValueError('level metadata is missing cached chart records')
    return levels, dict(schema='maimai.global_level_condition.v1', source_kind=source_kind)
