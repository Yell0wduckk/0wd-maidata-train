"""python -m maidata.inference.preencode: joint E only, reusable local/center P0."""
import argparse
import json


def parser():
    from .precompute import add_runtime, add_audio
    result = argparse.ArgumentParser(description='Preencode joint-model audio without D/P/M or chart generation.')
    add_runtime(result)
    add_audio(result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    import torch
    from .cache import JointCache
    from .checkpoint import JointCheckpoint
    from .precompute import read_timing, read_features, requested_audio_identity
    from .progress import StderrProgress, stage
    progress = StderrProgress(interval=args.progress_interval) if args.progress else None
    device = torch.device(args.device)
    timing_data, timing = read_timing(args)
    with stage(progress, 'models'):
        checkpoint = JointCheckpoint(args.joint_checkpoint)
    cache = JointCache(args.cache_dir, checkpoint, device=device, precision=args.precision, progress=progress)
    identity = requested_audio_identity(checkpoint, args, timing, device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
        encoded = cache.find_audio(identity)
        if encoded is None:
            with stage(progress, 'audio'):
                grid, duration = read_features(args, timing_data, checkpoint.config)
            encoded = cache.encode(identity, grid, timing, duration)
    print(json.dumps(dict(audio_cache=cache.report['audio'], duration_seconds=encoded.duration,
                          windows=len(encoded.local.starts)), ensure_ascii=False))


if __name__ == '__main__':
    main()
