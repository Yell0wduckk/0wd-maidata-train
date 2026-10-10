"""Resolve dataset paths from a manifest and explicit CLI overrides once."""
import json
from pathlib import Path


DEFAULT_INPUTS = dict(dataset='profile', chart_cache='chart-cache',
    note_view_manifest='note-view/manifest.json', interpolation_geometry='geometry',
    bpm_audio_manifest='audio/bpm-audio-manifest.json', spatial_view_cache='spatial-views',
    physical_path_catalog='physical/path-catalog', physical_source_rows='physical/source-rows',
    physical_interval_cache='physical/interval-cache', route_artifact='slide-geometry/route-artifact',
    song_chart_cache='song-chart', song_audio_cache='song-audio')
PATH_KEYS = (*DEFAULT_INPUTS, 'majradar_cache', 'latent_cache', 'chart_encoder_checkpoint', 'epoch_plan_cache', 'stage_a_checkpoint', 'cache',
             'audio_condition_cache', 'song_context_cache', 'decoder_context_cache', 'teacher_audio_cache',
             'draft_cache', 'draft_checkpoint')


def resolve_training_paths(args, stage='A'):
    if getattr(args, '_training_paths_resolved', False):
        return args
    root = Path(args.dataset_root).expanduser().resolve() if getattr(args, 'dataset_root', None) else None
    manifest = getattr(args, 'dataset_manifest', None)
    if manifest is not None:
        manifest = Path(manifest).expanduser().resolve()
        if root is None:
            root = manifest.parent
    elif root is not None:
        manifest = next((root / name for name in ('dataset-v10.json', 'dataset.json') if (root / name).is_file()), None)
    inputs = dict(DEFAULT_INPUTS) if root is not None else {}
    if manifest is not None:
        document = json.loads(manifest.read_text(encoding='utf-8-sig'))
        inputs.update({k.replace('-', '_'): v for k, v in document.get('inputs', {}).items()})
        args.dataset_manifest = manifest
    for name in PATH_KEYS:
        explicit = getattr(args, name, None)
        value = explicit if explicit is not None else inputs.get(name)
        if value is not None:
            path = Path(value).expanduser()
            if explicit is None and root is not None and not path.is_absolute():
                path = root / path
            setattr(args, name, path.resolve())
    if stage == 'B' and getattr(args, 'latent_cache', None):
        if getattr(args, 'cache', None) and args.cache != args.latent_cache:
            raise ValueError('--cache and --latent-cache identify different representations')
        args.cache = args.latent_cache
    required = ('epoch_plan_cache', 'route_artifact')
    required += ('cache', 'song_context_cache') if stage == 'B' else tuple(DEFAULT_INPUTS)
    if stage == 'B' and not getattr(args, 'resume', None):
        required += ('stage_a_checkpoint',)
    if stage == 'B' and getattr(args, 'resolved_recipe', {}).get('objective') == 'nll+decoder-l3':
        required += ('decoder_context_cache',)
    optional = {'physical_source_rows', 'note_view_manifest', 'song_chart_cache'}
    missing = [name for name in required if name not in optional and not getattr(args, name, None)]
    if missing:
        raise ValueError('missing training paths: ' + ', '.join('--' + n.replace('_', '-') for n in missing))
    args._training_paths_resolved = True
    return args
