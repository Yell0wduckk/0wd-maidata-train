"""python -m maidata.inference.predraft: D/P only, consuming preencoded audio."""
import argparse
import json
from pathlib import Path


def parser():
    from .precompute import add_runtime
    result = argparse.ArgumentParser(description='Predecode joint Draft/Plan from an audio cache without E/M.')
    add_runtime(result)
    result.add_argument('--audio-cache', required=True, help='The audio_cache.path returned by preencode')
    result.add_argument('--difficulty', type=int, choices=range(1, 8), required=True)
    result.add_argument('--level', type=float, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    import torch
    from .cache import JointCache, execution, load_audio
    from .checkpoint import JointCheckpoint
    from .progress import StderrProgress, stage
    progress = StderrProgress(interval=args.progress_interval) if args.progress else None
    device = torch.device(args.device)
    with stage(progress, 'models'):
        checkpoint = JointCheckpoint(args.joint_checkpoint)
    cache = JointCache(args.cache_dir, checkpoint, device=device, precision=args.precision, progress=progress)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
        with stage(progress, 'audio_cache_read'):
            encoded = load_audio(args.audio_cache, checkpoint.config, device=device)
            if (encoded.identity['checkpoint'] != checkpoint.identity
                    or encoded.identity['execution'] != execution(device, args.precision)):
                raise ValueError('audio cache checkpoint/device/precision differs from request')
        conditions = cache.draft(encoded, difficulty=args.difficulty, level=args.level)
    print(json.dumps(dict(audio_cache=str(Path(args.audio_cache).resolve()),
        draft_cache=cache.report['draft'], windows=len(conditions.starts)), ensure_ascii=False))


if __name__ == '__main__':
    main()
