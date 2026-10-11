"""Select a natural multi-M-chunk epoch-one update from accepted metadata only.

Run from the source checkout, or supply --source-root. This never constructs a
Dataset, opens window tensors/audio/charts, or changes any batch budget.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, default=Path('configs/joint-z4-gt-linux.json'))
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root / 'src'))
    from maidata.entry import CONFIG
    from maidata.joint_training.recipe import resolve
    from maidata.joint_training.data import JointPlan
    from maidata.decoder_training.ordering import history_dropped
    from maidata.decoder_training.batching import window_chunks
    from maidata.training.batching import group_token_cost

    config_path = args.config if args.config.is_absolute() else root / args.config
    config = dict(CONFIG)
    config.update(json.loads(config_path.read_text(encoding='utf-8-sig')))
    config = resolve(config)
    plan = JointPlan(config['pretrain_train_plan'], config, training=True)
    manifest_path = Path(config['decoder_train_window_cache']) / 'manifest.json'
    windows = json.loads(manifest_path.read_text(encoding='utf-8-sig'))['windows']
    budget = config['decoder_train_chunk_token_budget']
    best = None
    for check_step, logical_update in enumerate(plan.batches(1, config)):
        physical, padded_rows, occurrences = [], [], 0
        for groups in logical_update:
            rows = sorted((row for variant in groups
                           for row in plan.decoder_plan.rows(variant, 1)),
                          key=lambda row: row['sequence_index'])
            model_rows = []
            for row in rows:
                target, history = windows[row['window_id']]['lengths']
                dropped = history_dropped(config, True, 1, row)
                model_rows.append({'length': target + (0 if dropped else history)})
            chunks = window_chunks(model_rows, budget)
            costs = [len(indices) * max(model_rows[index]['length'] for index in indices)
                     for indices in chunks]
            proxy = sum(group_token_cost(plan.groups[variant], config) for variant in groups)
            physical.append(dict(audio_variants=list(groups), groups=len(groups),
                                 original_proxy=proxy, decoder_occurrences=len(rows),
                                 m_chunks=len(chunks), chunk_padded_rows=costs))
            padded_rows.extend(costs)
            occurrences += len(rows)
        candidate = dict(check_step=check_step, epoch=1,
            groups=sum(len(groups) for groups in logical_update),
            original_proxy=sum(batch['original_proxy'] for batch in physical),
            decoder_occurrences=occurrences, m_chunks=len(padded_rows),
            total_padded_rows=sum(padded_rows), chunk_padded_rows=padded_rows,
            chunk_token_budget=budget, physical_batches=physical)
        # Stable tie-break: the first original update wins an otherwise exact tie.
        if best is None or (candidate['m_chunks'], candidate['total_padded_rows']) > (
                best['m_chunks'], best['total_padded_rows']):
            best = candidate
    if best is None:
        raise ValueError('the accepted epoch-one plan contains no logical updates')
    best['natural_multi_chunk'] = best['m_chunks'] > 1
    best['selection'] = 'maximum natural M chunks, then maximum total padded rows'
    best['config_path'] = str(config_path)
    best['window_manifest'] = str(manifest_path)
    print(json.dumps(best, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
