"""python -m maidata.inference --help"""
import argparse
from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path


def parser():
    from .options import GenerationOptions
    from .progress import DEFAULT_INTERVAL
    defaults = GenerationOptions()
    p = argparse.ArgumentParser(description='Current E/D/P + Decoder M full-song inference (new five-token fields).')
    p.add_argument('--encoder-draft-checkpoint', help='Required for Z or predicted plan; unused for Z-off GT/none')
    p.add_argument('--decoder-checkpoint', required=True)
    p.add_argument('--z-cross-attention', action=argparse.BooleanOptionalAction, default=None,
                   help='Inherit checkpoint by default; --no-z-cross-attention disables Z')
    p.add_argument('--plan-source', choices=('predicted', 'gt', 'none'), default='predicted')
    p.add_argument('--gt-plan', help='JSON windows: [{start_beat, plan12, valid}] for --plan-source gt')
    p.add_argument('--audio', required=True, help='Original audio; also defines the full-song duration')
    clock = p.add_mutually_exclusive_group(required=True)
    clock.add_argument('--timing', help='JSON: first_seconds and bpm=[[beat_num,beat_den,bpm], ...]')
    clock.add_argument('--bpm', type=float, help='Constant BPM; combine with --first-seconds')
    p.add_argument('--first-seconds', type=float, default=0.)
    features = p.add_mutually_exclusive_group(required=True)
    features.add_argument('--grid', help='Existing training-format audio grid NPZ; no feature recomputation')
    features.add_argument('--mel-stats', help='Fixed training Mel statistics JSON for extraction from audio')
    p.add_argument('--mel-root', help='Optional relocated referenced Mel directory for --grid')
    p.add_argument('--geometry', required=True, help='Physical path-catalog directory')
    p.add_argument('--slide-data', required=True, help='MiaCode slide_data.json')
    p.add_argument('--difficulty', type=int, required=True, choices=range(1, 8))
    p.add_argument('--level', type=float, required=True, help='Exact numeric target level, e.g. 14.5')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    p.add_argument('--temperature', type=float, default=defaults.temperature)
    p.add_argument('--seed', type=int, default=defaults.seed)
    p.add_argument('--encoder-window-batch', type=int, default=8)
    p.add_argument('--max-tokens-per-window', type=int, default=defaults.max_tokens_per_window)
    p.add_argument('--max-tokens-per-moment', type=int, default=defaults.max_tokens_per_moment,
                   help='Stop onset expansion at threshold; append required completion fields, then advance MOMENT')
    p.add_argument('--max-backtracks-per-window', type=int, default=defaults.max_backtracks_per_window)
    p.add_argument('--rollback-tokens', type=int, default=defaults.rollback_tokens, help='Search frontier, never a history token limit')
    p.add_argument('--physical-hard-mask', action=argparse.BooleanOptionalAction, default=defaults.physical_hard_mask)
    p.add_argument('--trace', action='store_true', default=defaults.trace, help='Explicit opt-in detailed generation trace')
    p.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True, help='Stage/window progress on stderr; stdout remains final JSON')
    p.add_argument('--progress-interval', type=float, default=DEFAULT_INTERVAL, help='Minimum seconds between encoder/token progress updates')
    p.add_argument('--force', action='store_true')
    return p


def ir_metadata(timing, duration):
    end = Fraction(math.ceil(timing.second_to_beat(duration)*96), 96)
    return dict(schema='maimai.chart_ir.v1', tick_rate=180, first_seconds=timing.first_seconds,
        default_meter=dict(numerator=4, denominator=4),
        controls=[dict(kind='bpm', bpm=bpm, position_beats=dict(num=b.numerator, den=b.denominator),
                       tick=math.floor(timing.beat_to_second(b)*180+.5)) for b, bpm in timing.points],
        render_end_beat=dict(num=end.numerator, den=end.denominator),
        eligibility=dict(training_eligible=True, isolation_reasons=[]))


def main(argv=None):
    args = parser().parse_args(argv)
    from .progress import StderrProgress, stage
    progress = StderrProgress(interval=args.progress_interval) if args.progress else None
    with stage(progress, 'runtime'):
        import torch
        import soundfile as sf
        from maidata.chart import decode_chart, token_to_dict
        from maidata.physical import Geometry
        from maidata.preparation.cache import load_grid
        from maidata.time import TimingMap
        from .checkpoint import load_models
        from .conditioning import audio_grid, prepare_conditions, prepare_decoder_conditions, with_plan_source
        from .generation import generate, GenerationOptions
        from .materialize import read_json, write_json

    output = Path(args.output)
    names = ('generation.json', 'model.tokens.json', 'model.ir.json', 'generation-error.json')
    if not args.force and any((output/n).exists() for n in names):
        raise FileExistsError(f'output already exists: {output}; use a new directory or --force')
    if args.encoder_window_batch < 1 or not math.isfinite(args.level):
        raise ValueError('encoder-window-batch must be positive and level finite')
    device = torch.device(args.device)
    timing_data = read_json(args.timing) if args.timing else dict(first_seconds=args.first_seconds, bpm=[[0, 1, args.bpm]])
    timing = TimingMap.from_audio_timing(timing_data)
    with stage(progress, 'models'):
        models = load_models(args.encoder_draft_checkpoint, args.decoder_checkpoint, device=device,
                             z_cross_attention=args.z_cross_attention, plan_source=args.plan_source)
    with stage(progress, 'audio'):
        if args.grid:
            grid = load_grid(args.grid, mel_root=args.mel_root, validate_input=False)
            info = sf.info(args.audio)
            duration = info.frames/info.samplerate
        else:
            grid, duration = audio_grid(args.audio, timing_data, read_json(args.mel_stats),
                                       samples_per_beat=models.config['samples_per_beat'])
    with stage(progress, 'geometry'):
        geometry = Geometry.from_resources(args.geometry, args.slide_data)
    options = GenerationOptions(temperature=args.temperature, seed=args.seed,
        max_tokens_per_window=args.max_tokens_per_window, max_tokens_per_moment=args.max_tokens_per_moment,
        max_backtracks_per_window=args.max_backtracks_per_window,
        rollback_tokens=args.rollback_tokens, physical_hard_mask=args.physical_hard_mask, trace=args.trace)
    metadata = ir_metadata(timing, duration)
    output.mkdir(parents=True, exist_ok=True)
    if args.force:
        # Remove only this command's previous products, so a failed replacement
        # cannot leave a stale success report/IR beside a new error report.
        for name in names:
            path = output/name
            if path.is_file():
                path.unlink()
    try:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
            if models.draft is None:
                conditions = prepare_decoder_conditions(models.decoder_encoder, models.config, grid, timing, duration,
                    device=device, encoder_window_batch=args.encoder_window_batch, progress=progress)
            else:
                conditions = prepare_conditions(models.encoder, models.draft, models.planner, models.config, grid,
                    timing, duration, difficulty=args.difficulty, level=args.level, device=device,
                    encoder_window_batch=args.encoder_window_batch, progress=progress,
                    decoder_encoder=models.decoder_encoder)
            conditions = with_plan_source(conditions, args.plan_source, read_json(args.gt_plan) if args.gt_plan else None)
            with stage(progress, 'generation'):
                encoded, windows = generate(models.decoder, conditions, timing, geometry, models.config, metadata,
                                             duration, device=device, options=options, progress=progress)
        with stage(progress, 'chart_ir'):
            ir = decode_chart(encoded, geometry)
    except Exception as exc:
        write_json(output/'generation-error.json', dict(error=str(exc), options=asdict(options),
                                                       checkpoints=models.provenance))
        raise
    bypass = sum(len(w['physical_bypass']) for w in windows)
    report = dict(schema='maidata.generation.v1', checkpoints=models.provenance,
        audio=str(Path(args.audio).resolve()), timing=timing_data, duration_seconds=duration,
        difficulty=args.difficulty, level=args.level, device=str(device), precision=args.precision,
        options=asdict(options), config=models.config, windows=windows, moments=len(ir['moments']),
        physical_bypass_count=bypass, complete=True)
    with stage(progress, 'write_outputs'):
        write_json(output/'model.tokens.json', dict(schema='maidata.generated-tokens.v1', start_tick=encoded.start_tick,
            metadata=encoded.metadata, tokens=[token_to_dict(t) for t in encoded.tokens]))
        write_json(output/'model.ir.json', ir)
        write_json(output/'generation.json', report)
        error_file = output/'generation-error.json'
        if error_file.exists():
            error_file.unlink()
    print(json.dumps(dict(output=str(output.resolve()), windows=len(windows), moments=len(ir['moments']),
                          physical_hard_mask=args.physical_hard_mask, physical_bypass_count=bypass), ensure_ascii=False))


if __name__ == '__main__':
    main()
