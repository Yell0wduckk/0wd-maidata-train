"""Shared arguments/input reading for the two joint precomputation commands."""
import argparse


def add_runtime(parser):
    from .progress import DEFAULT_INTERVAL
    parser.add_argument('--joint-checkpoint', required=True, help='Current joint E/D/R/P/M checkpoint')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--progress-interval', type=float, default=DEFAULT_INTERVAL)


def add_audio(parser):
    parser.add_argument('--audio', required=True)
    clock = parser.add_mutually_exclusive_group(required=True)
    clock.add_argument('--timing')
    clock.add_argument('--bpm', type=float)
    parser.add_argument('--first-seconds', type=float, default=0.)
    features = parser.add_mutually_exclusive_group(required=True)
    features.add_argument('--grid')
    features.add_argument('--mel-stats')
    parser.add_argument('--mel-root')
    parser.add_argument('--encoder-window-batch', type=int, default=8)


def read_timing(args):
    from .materialize import read_json
    from maidata.time import TimingMap
    document = read_json(args.timing) if args.timing else dict(
        first_seconds=args.first_seconds, bpm=[[0, 1, args.bpm]])
    return document, TimingMap.from_audio_timing(document)


def read_features(args, timing_data, config):
    from .conditioning import audio_grid
    from .materialize import read_json
    from maidata.preparation.cache import load_grid
    import soundfile as sf
    if args.grid:
        grid = load_grid(args.grid, mel_root=args.mel_root, validate_input=False)
        info = sf.info(args.audio)
        return grid, info.frames/info.samplerate
    return audio_grid(args.audio, timing_data, read_json(args.mel_stats),
                      samples_per_beat=config['samples_per_beat'])


def requested_audio_identity(checkpoint, args, timing, device):
    from .cache import audio_identity
    return audio_identity(checkpoint, args.audio, timing, grid=args.grid, mel_root=args.mel_root,
        mel_stats=args.mel_stats, device=device, precision=args.precision,
        encoder_window_batch=args.encoder_window_batch)
