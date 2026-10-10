from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Sequence

from maimai_baseline.contract import SEED, ENCODER_CAPACITY_PRESETS, DEFAULT_ENCODER_CAPACITY_PRESET
from maimai_baseline.model.attention import DEFAULT_PRESET, PRESETS
from maimai_baseline.data.audio import TRAINING_POLICIES
from maimai_baseline.inference.evaluation import compare_manifest
from maimai_baseline.inference.api import generate_baseline_inference
from maimai_baseline.inference.materialize import materialize_baseline_inference
from maimai_baseline.training.runner import preflight
from maimai_baseline.training.runner import run_training


def _training_paths(parser: argparse.ArgumentParser, *, fine_tuning: bool = False) -> None:
    parser.set_defaults(latent_enabled=False)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, help="root for manifest-relative dataset inputs")
    parser.add_argument("--dataset-manifest", type=Path, help="dataset path map; explicit paths override it")
    parser.add_argument("--song-context-cache", help="whole-song frozen feature/audio mapping")
    parser.add_argument("--decoder-context-cache", help="frozen A90 candidate keys for Decoder-supervised B")
    parser.add_argument("--song-chart-cache", help="verified whole-song GT input cache")
    parser.add_argument("--majradar-cache", help="prebuilt per-sixteen-beat public MajRadar labels")
    parser.add_argument("--latent-cache", help="frozen encoder60 final mu cache shared by A and B")
    parser.add_argument("--draft-cache", help="clean canonical audio-draft tokens and train-only noise statistics")
    parser.add_argument("--draft-checkpoint", help="independently trained frozen audio draft export")
    parser.add_argument("--chart-encoder-checkpoint", help="independent epoch-060 checkpoint for fresh frozen-latent A")
    parser.add_argument("--dataset")
    parser.add_argument("--chart-cache")
    parser.add_argument("--note-view-manifest", help="verified v10 short-note view from prepare-note-view.py")
    parser.add_argument("--interpolation-geometry")
    parser.add_argument("--bpm-audio-manifest", help="verified BPM audio manifest; reuse v5 preparation artifacts")
    parser.add_argument("--spatial-view-cache")
    parser.add_argument("--epoch-plan-cache", required=True)
    parser.add_argument("--physical-path-catalog")
    parser.add_argument("--physical-source-rows", help="separate source-row mapping directory")
    parser.add_argument("--physical-interval-cache")
    parser.add_argument("--route-artifact")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--difficulty-loss-preset", choices=("uniform", "conservative", "standard", "aggressive"), default="uniform")
    parser.add_argument("--audio-attention-preset", choices=PRESETS, default=DEFAULT_PRESET,
                        help="E0-E2 beat radii 1/2/4; independent song audio attends the full song")
    if not fine_tuning:
        from maimai_baseline.style_contract import STYLE_ID, FROZEN_STYLE_ID, NATIVE_STYLE_ID, DRAFT_STYLE_ID
        from maimai_baseline.draft_contract import BEAT_STYLE_ID
        from maimai_baseline.radar_contract import RADAR_ID
        parser.add_argument("--chart-style", choices=("none", STYLE_ID, FROZEN_STYLE_ID, NATIVE_STYLE_ID, DRAFT_STYLE_ID, BEAT_STYLE_ID, RADAR_ID), default=BEAT_STYLE_ID)
        parser.add_argument("--decoder-rope-coordinate", choices=("window-beat",), default="window-beat")
    from maimai_baseline.route_contract import ROUTE_HEADS
    parser.add_argument("--route-head", choices=ROUTE_HEADS, default="prototype-distance")
    parser.add_argument("--song-audio-cache", help="immutable independent full-song beat mel input pack")
    parser.add_argument("--teacher-audio-cache", help="external full native22 P0/C0/C1 cache; required for native audio models")
    parser.add_argument("--audio-cross-attention", choices=("p0-p2-fused",), default="p0-p2-fused")
    parser.add_argument("--encoder-capacity-preset", choices=ENCODER_CAPACITY_PRESETS, default=DEFAULT_ENCODER_CAPACITY_PRESET)
    parser.add_argument("--specialized-supervision", action=argparse.BooleanOptionalAction, default=False,
                        help="apply Hold/Slide specialized loss multipliers")
    parser.add_argument("--hold-loss-multiplier", type=_positive_float, default=1.2)
    parser.add_argument("--slide-loss-multiplier", type=_positive_float, default=1.2)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--token-budget", type=_positive_integer)
    parser.add_argument("--plan-offset", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--physical-warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip", type=_positive_float)
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument("--training-recipe", type=_positive_integer, default=None, help="legacy token-budget alias")
    parser.add_argument("--learning-rate", type=_positive_float, default=None,
                        help=("SFT decoder/head learning rate (default 2e-5)" if fine_tuning else
                              "override recipe AdamW learning rate (default 4e-4 * sqrt(budget/2048))"))
    parser.add_argument("--audio-policy", choices=TRAINING_POLICIES, default=None)
    parser.add_argument("--normalized-audio-manifest")
    parser.add_argument(
        "--validation-policy",
        choices=("required", "disabled"),
        default=None,
        help="require the declared validation split, or explicitly run the train-only contract",
    )
    parser.add_argument("--debug", action="store_true")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _training_schedule_arguments(parser: argparse.ArgumentParser, *, fine_tuning: bool = False) -> None:
    parser.add_argument(
        "--scheduler-epochs",
        type=_positive_integer,
        default=None,
        help="SFT traversal horizon; must equal stop-after-epochs (default 3)" if fine_tuning else (
            "cosine scheduler horizon in epochs; independent of the stop epoch "
            "and not limited by epoch-plan cache capacity"
        ),
    )
    parser.add_argument(
        "--stop-after-epochs",
        type=_positive_integer,
        default=None,
        help="SFT absolute Master traversal count (default 3)" if fine_tuning else (
            "absolute epoch count at which training stops; the bound experiment determines the default and required plan capacity"
        ),
    )


def _generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route-artifact", required=True)
    parser.add_argument("--compact-profile", required=True)
    audio_input = parser
    audio_input.add_argument("--song-id")
    audio_input.add_argument("--audio", help="raw MP3/WAV; preprocessing comes from checkpoint")
    parser.add_argument("--audio-cache")
    parser.add_argument("--teacher-audio-cache", help="external native audio cache required by native audio checkpoints")
    parser.add_argument("--bpm-audio-manifest", help="verified prebuilt BPM96 features for v9 checkpoints")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--miacode-chart-ir-exe")
    parser.add_argument("--title")
    parser.add_argument("--artist", default="")
    parser.add_argument("--designer", default="maimai baseline")
    parser.add_argument("--timing", required=True)
    parser.add_argument("--difficulty", type=int, choices=(2, 3, 4, 5, 6), default=5)
    parser.add_argument("--level", type=float, required=True)
    parser.add_argument("--cabinet", choices=("SD", "DX"), default="DX")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--local-seed", type=int, default=None)
    parser.add_argument("--style-mode", choices=("ar", "draft", "disabled"), default=None)
    parser.add_argument("--style-reference")
    parser.add_argument("--radar-temperature", "--latent-temperature", dest="latent_temperature", type=float, default=None,
                        help="radar Gaussian sampling temperature (default 1)")
    parser.add_argument("--mixture-temperature", type=float, default=None, help="radar GMM component sampling temperature (default 1)")
    parser.add_argument("--trace", action="store_true", help="write detailed per-row inference trace and probability statistics")
    parser.add_argument("--physical-hard-mask", action=argparse.BooleanOptionalAction, default=True,
                        help="constrain free-run choices with the generated-prefix Physical v5 policy")
    parser.add_argument("--miacode-slide-check", action=argparse.BooleanOptionalAction, default=True,
                        help="MiaCode slide judgments with bounded retries; requires --physical-hard-mask")
    parser.add_argument("--physical-path-catalog", help="static Physical paths; required with --physical-hard-mask")
    parser.add_argument("--max-rows-per-window", type=int, default=512)
    parser.add_argument("--truncation-policy", choices=("fail", "close"), default="fail")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="maimai-baseline")
    subparsers = result.add_subparsers(dest="operation", required=True)

    for command, recipe, description in (
        ("train-chart-encoder", "configs/train-chart-encoder.json", "independently pretrain the chart encoder on a supplied fixed plan"),
        ("train-audio-draft", "configs/train-audio-draft.json", "train a non-AR full-song audio draft with latent-only reconstruction"),
    ):
        encoder = subparsers.add_parser(command, help=description)
        encoder.add_argument("--experiment-config", type=Path, default=Path(recipe))
        encoder.add_argument("--dataset-root", type=Path)
        encoder.add_argument("--dataset-manifest", type=Path)
        encoder.add_argument("--teacher-audio-cache", type=Path)
        if command == "train-audio-draft":
            encoder.add_argument("--validation-dataset-manifest", type=Path,
                                 help="fixed canonical held-out 32-song draft validation inputs")
        for name in ("dataset", "chart-cache", "spatial-view-cache", "song-chart-cache", "route-artifact"):
            encoder.add_argument("--" + name)
        encoder.add_argument("--epoch-plan-cache", required=True)
        encoder.add_argument("--output", required=True)
        encoder.add_argument("--resume")
        encoder.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        encoder.add_argument("--precision", choices=("bf16", "fp32"))
        encoder.add_argument("--num-workers", type=int)
        encoder.add_argument("--seed", type=int)
        encoder.add_argument("--token-budget", type=_positive_integer)
        encoder.add_argument("--plan-offset", type=int)
        encoder.add_argument("--warmup-steps", type=int)
        encoder.add_argument("--learning-rate", type=_positive_float)
        encoder.add_argument("--weight-decay", type=float)
        encoder.add_argument("--gradient-clip", type=_positive_float)
        encoder.add_argument("--scheduler-epochs", type=_positive_integer,
                             help="scheduler horizon in complete epochs; supplied fixed plan must cover this horizon")
        encoder.add_argument("--stop-after-epochs", type=_positive_integer,
                             help="absolute complete epoch count at which encoder pretraining stops")
        encoder.add_argument("--cuda-check", action="store_true", help="check one original-plan batch and record its CUDA cost without training epochs")
        encoder.add_argument("--debug", action="store_true")


    preflight_parser = subparsers.add_parser("preflight")
    _training_paths(preflight_parser)
    _training_schedule_arguments(preflight_parser)
    preflight_parser.set_defaults(command="fit")
    preflight_parser.set_defaults(resume=None, steps=None, benchmark_warmup_steps=0)

    verify = subparsers.add_parser("verify-inputs", help="verify portable input files once after transfer; CPU only")
    _training_paths(verify)
    _training_schedule_arguments(verify)
    verify.set_defaults(command="fit", resume=None, steps=None, benchmark_warmup_steps=0)

    train = subparsers.add_parser("train", help="train one explicit A, B or sft stage on a supplied final plan")
    _training_paths(train)
    _training_schedule_arguments(train)
    train.add_argument("--stage", choices=("A", "B", "sft"), required=True)
    train.add_argument("--cuda-check", action="store_true", help="one original-plan batch update and cost record, without training epochs")
    train.set_defaults(command="fit", steps=None, benchmark_warmup_steps=0)
    source = train.add_mutually_exclusive_group()
    source.add_argument("--resume")
    source.add_argument("--init-checkpoint")
    train.add_argument("--stage-a-checkpoint")
    train.add_argument("--cache", help="frozen chart representation cache for B")
    train.add_argument("--audio-condition-cache")
    train.add_argument("--sft-mode", choices=("teacher-forcing", "moment-rollout"))
    train.add_argument("--rollout-fraction", type=float, default=.25)
    train.add_argument("--max-rollout-notes", type=_positive_integer, default=33)
    train.add_argument("--encoder-learning-rate", type=_positive_float, default=None)

    fit = subparsers.add_parser("fit")
    _training_paths(fit)
    _training_schedule_arguments(fit)
    fit.set_defaults(command="fit")
    fit.add_argument("--resume")
    fit.set_defaults(steps=None, benchmark_warmup_steps=0)

    sft = subparsers.add_parser("sft", help="Master-only supervised single-moment fine-tuning")
    _training_paths(sft, fine_tuning=True)
    _training_schedule_arguments(sft, fine_tuning=True)
    sft.set_defaults(command="fit", steps=None, benchmark_warmup_steps=0,
                     learning_rate=None, scheduler_epochs=None, stop_after_epochs=None)
    source = sft.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint")
    source.add_argument("--resume")
    sft.add_argument("--sft-mode", choices=("teacher-forcing", "moment-rollout"), required=True)
    sft.add_argument("--rollout-fraction", type=float, default=.25)
    sft.add_argument("--max-rollout-notes", type=_positive_integer, default=33)
    sft.add_argument("--encoder-learning-rate", type=_positive_float, default=None)

    benchmark = subparsers.add_parser("benchmark")
    _training_paths(benchmark)
    benchmark.set_defaults(command="benchmark")
    benchmark.add_argument("--steps", type=int, default=50)
    benchmark.add_argument("--benchmark-warmup-steps", type=int, default=10)
    benchmark.set_defaults(resume=None)

    versions = subparsers.add_parser("runtime-versions")
    radar = subparsers.add_parser('build-majradar-cache', help='CPU-only per-sixteen-beat radar labels; no training')
    radar.add_argument('--profile', required=True)
    radar.add_argument('--subset-config', required=True)
    radar.add_argument('--output', required=True)
    radar.add_argument('--miacode-exe', required=True)
    radar.add_argument('--majradar-project')
    radar.add_argument('--host-project')
    radar.add_argument('--host-dll')
    radar.add_argument('--build-root')
    radar.add_argument('--debug', action='store_true')
    latent = subparsers.add_parser('build-latent-cache', help='encode frozen encoder60 mu once per complete chart/view')
    for name in ('dataset-root', 'dataset-manifest', 'dataset', 'song-chart-cache', 'route-artifact'):
        latent.add_argument('--' + name, type=Path)
    for name in ('epoch-plan-cache', 'chart-encoder-checkpoint', 'output'):
        latent.add_argument('--' + name, type=Path, required=True)
    latent.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    latent.add_argument('--debug', action='store_true')
    prepare = subparsers.add_parser('prepare-prior-decoder-data', help='build complete-tail windows/plans, full encoder60 mu and reused P0 mappings')
    for name in ('dataset-root', 'dataset-manifest', 'dataset', 'chart-cache',
                 'spatial-view-cache', 'song-chart-cache', 'bpm-audio-manifest',
                 'route-artifact', 'chart-encoder-checkpoint', 'latent-cache'):
        prepare.add_argument('--' + name, type=Path)
    for name in ('epoch-plan-cache', 'song-context-cache', 'output'):
        prepare.add_argument('--' + name, type=Path, required=True)
    prepare.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    prepare.add_argument('--debug', action='store_true')
    native_data = subparsers.add_parser('prepare-encoder-audio-data', help='prepare complete-tail windows and fixed A/B plans without old models or latent/audio caches')
    for name in ('dataset-root', 'dataset-manifest', 'dataset', 'chart-cache', 'spatial-view-cache', 'song-chart-cache', 'bpm-audio-manifest', 'route-artifact'):
        native_data.add_argument('--' + name, type=Path)
    for name in ('epoch-plan-cache', 'output'):
        native_data.add_argument('--' + name, type=Path, required=True)
    native_data.add_argument('--debug', action='store_true')
    draft_data = subparsers.add_parser('prepare-audio-draft-data', help='prepare complete-tail windows, canonical draft and original-view A plans')
    draft_data.add_argument('--validation-source-manifest', type=Path,
                            help='accepted full inputs outside the training-song selection')
    draft_data.add_argument('--validation-song-count', type=int, default=32)
    draft_data.add_argument('--validation-from-training', action='store_true',
                            help='hold out songs from draft only; preserve the full A plan')
    for name in ('dataset-root', 'dataset-manifest', 'dataset', 'chart-cache', 'spatial-view-cache', 'song-chart-cache', 'bpm-audio-manifest', 'route-artifact'):
        draft_data.add_argument('--' + name, type=Path)
    for name in ('epoch-plan-cache', 'output'):
        draft_data.add_argument('--' + name, type=Path, required=True)
    draft_data.add_argument('--debug', action='store_true')
    draft_cache = subparsers.add_parser('build-draft-cache', help='cache deterministic canonical audio draft tokens and noise statistics')
    for name in ('dataset-root', 'dataset-manifest', 'dataset', 'song-chart-cache', 'route-artifact', 'teacher-audio-cache'):
        draft_cache.add_argument('--' + name, type=Path)
    for name in ('epoch-plan-cache', 'draft-checkpoint', 'output'):
        draft_cache.add_argument('--' + name, type=Path, required=True)
    draft_cache.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    draft_cache.add_argument('--debug', action='store_true')
    decoder_cache = subparsers.add_parser('build-decoder-context-cache', help='cache frozen A90 candidate keys once per unique audio')
    for name in ('dataset-root', 'dataset-manifest', 'dataset', 'song-chart-cache',
                 'song-audio-cache', 'bpm-audio-manifest', 'interpolation-geometry'):
        decoder_cache.add_argument('--' + name, type=Path)
    for name in ('stage-a-checkpoint', 'latent-cache', 'song-context-cache', 'route-artifact', 'output'):
        decoder_cache.add_argument('--' + name, type=Path, required=True)
    decoder_cache.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    decoder_cache.add_argument('--debug', action='store_true')
    decoder_cache.add_argument('--teacher-audio-cache', type=Path)
    song = subparsers.add_parser('build-song-context', help='bind native stitched P0 or cache continuous P0 for the frozen A')
    for name in ('stage-a-checkpoint', 'latent-cache', 'route-artifact', 'output'):
        song.add_argument('--' + name, type=Path, required=True)
    for name in ('dataset-root', 'dataset-manifest', 'dataset', 'song-chart-cache', 'song-audio-cache',
                 'bpm-audio-manifest', 'interpolation-geometry'):
        song.add_argument('--' + name, type=Path)
    song.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    song.add_argument('--debug', action='store_true')
    song.add_argument('--teacher-audio-cache', type=Path)
    bp = subparsers.add_parser('build-b-plan', help='derive final whole-song B boundaries without changing source references')
    for name in ('dataset-root', 'dataset-manifest', 'dataset'):
        bp.add_argument('--' + name, type=Path)
    for name in ('epoch-plan-cache', 'output'):
        bp.add_argument('--' + name, type=Path, required=True)
    bp.add_argument('--debug', action='store_true')
    infer = subparsers.add_parser("infer")
    infer.add_argument("--stream-events", action="store_true")
    _generation_arguments(infer)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--stream-input", action="store_true")
    materialize.add_argument("--preview-dir")
    materialize.add_argument("--inference-dir", required=True)
    materialize.add_argument("--timing", required=True)
    materialize.add_argument("--audio", required=True)
    materialize.add_argument("--miacode-chart-ir-exe", required=True)
    materialize.add_argument("--title")
    materialize.add_argument("--artist", default="")
    materialize.add_argument("--designer", default="maimai baseline")
    materialize.add_argument("--force", action="store_true")
    materialize.add_argument("--debug", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--debug", action="store_true")
    from maimai_baseline.style_contract import STYLE_ID
    cache = subparsers.add_parser("style-cache")
    _training_paths(cache)
    _training_schedule_arguments(cache)
    cache.add_argument("--stage-a-checkpoint", required=True)
    cache.add_argument("--cache-batch-size", type=_positive_integer, default=16)
    cache.add_argument("--dry-run", action="store_true")
    cache.set_defaults(command="fit", chart_style=STYLE_ID, resume=None)
    verify_cache = subparsers.add_parser("style-cache-verify")
    verify_cache.add_argument("--cache", required=True)
    p0 = subparsers.add_parser("style-p0-cache")
    _training_paths(p0)
    _training_schedule_arguments(p0)
    for name in ("stage-a-checkpoint", "cache"):
        p0.add_argument("--" + name, required=True)
    p0.add_argument("--dry-run", action="store_true")
    p0.set_defaults(command="fit", chart_style=STYLE_ID)
    export = subparsers.add_parser("style-export")
    export.add_argument("--cache", required=True)
    export.add_argument("--stage-a-checkpoint", required=True)
    export.add_argument("--route-artifact", required=True)
    export.add_argument("--song-id", required=True)
    export.add_argument("--difficulty", type=int, required=True)
    export.add_argument("--output", required=True)
    return result


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _run(args: argparse.Namespace) -> object:
    if args.operation == 'prepare-audio-draft-data':
        from maimai_baseline.data.audio_draft import prepare_data
        return prepare_data(args)
    if args.operation == 'build-draft-cache':
        from maimai_baseline.data.draft_cache import build_cache
        return build_cache(args)
    if args.operation == 'train-audio-draft':
        from maimai_baseline.training.audio_draft import run
        result = run(args)
        return result if isinstance(result, dict) else {'exit_code': 0, 'output': str(result)}
    if args.operation == 'prepare-encoder-audio-data':
        from maimai_baseline.data.encoder_audio import prepare_data
        return prepare_data(args)
    if args.operation == 'prepare-prior-decoder-data':
        from maimai_baseline.data.prior_decoder import prepare_data
        return prepare_data(args)
    if args.operation == 'build-decoder-context-cache':
        from maimai_baseline.training.decoder_context import build_decoder_context_cache
        return build_decoder_context_cache(args)
    if args.operation == 'build-latent-cache':
        from maimai_baseline.data.frozen_latent import build_cache
        return build_cache(args)
    if args.operation == 'build-song-context':
        from maimai_baseline.training.frozen_context import build_song_context
        return build_song_context(args)
    if args.operation == 'build-b-plan':
        from maimai_baseline.whole_song_training import build_frozen_plan
        return build_frozen_plan(args)
    if args.operation == "train-chart-encoder":
        from maimai_baseline.training.chart_encoder_pretrain import train
        result = train(args)
        return result if isinstance(result, dict) else {"exit_code": 0, "output": str(result)}
    if args.operation == 'build-majradar-cache':
        from maimai_baseline.data.majradar import build_cache
        return build_cache(profile=args.profile, subset_config=args.subset_config,
            output=args.output, miacode_exe=args.miacode_exe,
            majradar_project=args.majradar_project, host_project=args.host_project,
            host_dll=args.host_dll, build_root=args.build_root)
    if args.operation in {"train", "fit", "sft", "benchmark", "preflight", "style-cache"}:
        from maimai_baseline.training.recipe import bind_stage_recipe
        from maimai_baseline.training.paths import resolve_training_paths
        stage = args.stage if args.operation == "train" else ("sft" if args.operation == "sft" else "A")
        from maimai_baseline.style_contract import is_draft
        if is_draft(getattr(args, 'chart_style', None)) and stage != 'A':
            raise ValueError('audio draft models train stage A only; no B or sft stage')
        from maimai_baseline.style_contract import is_frozen
        args.latent_enabled = is_frozen(getattr(args, 'chart_style', None))
        bind_stage_recipe(args, stage)
        resolve_training_paths(args, stage)
        if args.operation == "train":
            if stage == "B":
                from maimai_baseline.training.ar_runner import train, cuda_check
                args.ar_config = args.experiment_config
                args.whole_song_plan = args.epoch_plan_cache
                args.execute = True
                if args.device == "auto":
                    from maimai_baseline.training.steps import _device
                    args.device = str(_device("auto"))
                return cuda_check(args) if args.cuda_check else train(args)
            if getattr(args, 'cuda_check', False):
                if stage != 'A':
                    raise ValueError('this CUDA check supports only A or B')
                from maimai_baseline.training.latent_cuda import cuda_check
                return cuda_check(args)
            args.operation = "sft" if stage == "sft" else "fit"
        if stage == "sft" and (not args.sft_mode or not (args.init_checkpoint or args.resume)):
            raise ValueError("sft requires --sft-mode and --init-checkpoint or --resume")
    if args.operation == "style-p0-cache":
        raise ValueError("prepare P0 with scripts/build-global-local-style-cache.py audio-condition-cache")
    if args.operation == "style-cache":
        from maimai_baseline.training.style_prepare import prepare_cache
        return prepare_cache(args)
    if args.operation == "style-cache-verify":
        from maimai_baseline.data.style_cache import StyleLatentCache
        cache = StyleLatentCache(args.cache, verify_payloads=True)
        return {"verified": True, "identity": cache.manifest["manifest_identity_sha256"]}
    if args.operation == "style-export":
        from maimai_baseline.training.style_prepare import export_reference
        return export_reference(args)
    if args.operation == "sft":
        from maimai_baseline.training.sft import run_sft
        return {"exit_code": run_sft(args)}
    if args.operation == "verify-inputs":
        from maimai_baseline.training.verification import verify_inputs
        return verify_inputs(args)
    if args.operation == "preflight":
        return preflight(args)
    if args.operation in {"fit", "benchmark"}:
        output = _path(args.output)
        if args.resume is None and output.exists():
            raise ValueError(f"refusing to overwrite baseline output: {output}")
        return {"exit_code": run_training(args)}
    if args.operation == "runtime-versions":
        from maimai_baseline.runtime_versions import policy
        return policy()
    if args.operation == "infer":
        if args.materialize and (not args.audio or not args.miacode_chart_ir_exe):
            raise ValueError("infer --materialize requires --audio and --miacode-chart-ir-exe")
        if args.audio_cache and not args.audio:
            raise ValueError("--audio-cache requires --audio")
        if args.audio and args.bpm_audio_manifest:
            raise ValueError("raw audio and BPM feature manifest are mutually exclusive")
        if args.materialize:
            from maimai_baseline.inference.materialize import resolve_chart_ir_executable
            resolve_chart_ir_executable(_path(args.miacode_chart_ir_exe))
        from maimai_baseline.inference.streaming import StreamEmitter
        stream = StreamEmitter() if args.stream_events else None
        result = generate_baseline_inference(
            stream=stream,
            checkpoint=_path(args.checkpoint),
            route_artifact=_path(args.route_artifact),
            profile_root=_path(args.compact_profile),
            song_id=str(args.song_id or _path(args.audio).stem),
            timing_path=_path(args.timing),
            output=_path(args.output),
            level=float(args.level),
            cabinet=str(args.cabinet),
            difficulty_id=int(args.difficulty),
            force=bool(args.force),
            device=str(args.device),
            seed=int(args.seed),
            temperature=float(args.temperature),
            local_seed=args.local_seed, style_mode=args.style_mode,
            style_reference=_path(args.style_reference) if args.style_reference else None,
            latent_temperature=args.latent_temperature, mixture_temperature=args.mixture_temperature,
            max_rows_per_window=int(args.max_rows_per_window),
            allow_truncated=args.truncation_policy == "close",
            audio_path=_path(args.audio) if args.audio else None,
            audio_cache=_path(args.audio_cache) if args.audio_cache else None,
            bpm_audio_manifest=_path(args.bpm_audio_manifest) if args.bpm_audio_manifest else None,
            physical_hard_mask=bool(args.physical_hard_mask),
            miacode_slide_check=bool(args.miacode_slide_check),
            trace=bool(args.trace),
            physical_path_catalog=_path(args.physical_path_catalog) if args.physical_path_catalog else None,
        )
        if stream: stream("inference_complete")
        if args.materialize:
            from maimai_baseline.inference.materialize import _json
            status = _path(args.output) / "pipeline-status.json"
            _json(status, {"status": "materializing", "complete": False})
            try:
                package = materialize_baseline_inference(
                    inference_dir=_path(args.output), timing_path=_path(args.timing), audio_path=_path(args.audio),
                    miacode_exe=_path(args.miacode_chart_ir_exe), title=args.title, artist=args.artist,
                    designer=args.designer, force=args.force,
                )
            except Exception as error:
                _json(status, {"status": "failed", "complete": False, "error": str(error)})
                raise
            _json(status, {"status": "complete", "complete": True})
            result["materialization"] = package
        return result
    if args.operation == "materialize":
        if args.stream_input:
            from maimai_baseline.inference.streaming import materialize_stream
            return materialize_stream(args)
        return materialize_baseline_inference(
            inference_dir=_path(args.inference_dir),
            timing_path=_path(args.timing),
            audio_path=_path(args.audio),
            miacode_exe=_path(args.miacode_chart_ir_exe),
            title=args.title,
            artist=str(args.artist),
            designer=str(args.designer),
            force=bool(args.force),
        )
    value = compare_manifest(_path(args.manifest))
    output = _path(args.output)
    if output.exists():
        raise ValueError(f"refusing to overwrite comparison output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        value = _run(args)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return int(value.get("exit_code", 0)) if isinstance(value, dict) else 0
    except Exception as error:
        print(
            json.dumps(
                {"schema": "maimai.baseline_error.v1", "message": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        if bool(getattr(args, "debug", False)):
            traceback.print_exc()
        return 2


__all__ = ["main", "parser"]
