"""One formal A process; no diagnostics, learned cache producers or B."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from maimai_baseline.query6_parent import A_CONFIG
from maimai_baseline.experiment import load_config
from maimai_baseline.contract import DEFAULT_ENCODER_CAPACITY_PRESET


def command(root, *, resume=None):
    dataset = Path('/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911')
    inputs = dict(dataset='profile', chart_cache='chart-cache', note_view_manifest='note-view/manifest.json',
        interpolation_geometry='geometry', bpm_audio_manifest='audio/bpm-audio-manifest.json',
        spatial_view_cache='spatial-views', physical_path_catalog='physical/path-catalog',
        physical_source_rows='physical/source-rows', route_artifact='slide-geometry/route-artifact')
    manifest = dataset / 'dataset-v10.json'
    if not manifest.exists():
        manifest = dataset / 'dataset.json'
    if manifest.exists():
        inputs.update({k: v for k, v in json.loads(manifest.read_text('utf8')).get('inputs', {}).items() if k in inputs})
    spec = load_config(A_CONFIG)['stages']['A']
    values = dict(experiment_config=A_CONFIG, **{k: dataset/v for k, v in inputs.items()},
        epoch_plan_cache='/root/autodl-tmp/maidata-decoder8-20261003/a-plan',
        physical_interval_cache='/root/autodl-tmp/maidata-full-interval-physical-v15-20260926/interval-cache',
        song_chart_cache='/root/autodl-tmp/maidata-shared1-full-train-20260924/run-input-v2/inputs/song-chart',
        song_audio_cache='/root/autodl-tmp/maidata-shared1-full-train-20260924/run/inputs/song-audio',
        audio_policy='normalized-p50', validation_policy='disabled', decoder_rope_coordinate='window-beat',
        training_recipe=4096, learning_rate=spec['learning_rate'], encoder_capacity_preset=DEFAULT_ENCODER_CAPACITY_PRESET,
        audio_attention_preset='local-e012', difficulty_loss_preset='uniform', chart_style='full-song-z128-no-global-v1',
        scheduler_epochs=155, stop_after_epochs=90, device='cuda', num_workers=12,
        output=Path(root)/'training/A', resume=resume)
    return [sys.executable, '-u', '-m', 'maimai_baseline', 'fit',
        *[x for k, v in values.items() if v is not None for x in ('--'+k.replace('_', '-'), str(v))],
        '--no-specialized-supervision']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error('experiment root must be absolute')
    output = args.root/'training/A'
    if output.exists() and args.resume is None:
        parser.error('existing A output requires explicit complete-epoch --resume')
    cmd = command(args.root, resume=args.resume)
    args.root.mkdir(parents=True, exist_ok=True)
    print(json.dumps(cmd), flush=True)
    with (args.root/'A.log').open('a', encoding='utf8') as log:
        return subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)


if __name__ == '__main__':
    raise SystemExit(main())
