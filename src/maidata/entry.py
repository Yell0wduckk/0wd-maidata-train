"""Single editable architecture configuration and component entry point."""

import argparse
from pprint import pprint

from .model import build_model, build_draft, build_planner, build_reconstruction, build_decoder

CONFIG = {
    "mel_bins": 128,
    "samples_per_beat": 24,
    "d_model": 256,
    "heads": 8,
    "ffn_dim": 1024,
    "stage_depths": (1, 1, 2),
    "stage_attention_radii": (1, 2, 4),
    "temporal_kernel_size": 15,
    "stem_channels": 32,
    "frequency_dim": 64,
    "frequency_queries": 4,
    "fpn_rates": (8, 4, 2),
    "target_beats": 16,
    "target_stride_beats": 16,
    "audio_before_beats": 4,
    "audio_after_beats": 4,
    "chart_history_beats": 4,
    "complex_rhythm_min_kinds": 3,
    "complex_rhythm_multiplier": 2.0,
    "dropout": 0.1,
    "rope_base": 10000,
    "rope_position_scale": 1.0,
    "draft_layers": 8,
    "draft_dim": 256,
    "draft_heads": 8,
    "draft_experts": 4,
    "draft_expert_ffn_dim": 512,
    "draft_latent_dim": 256,  # Must equal draft_dim; no latent bottleneck.
    "draft_z_stride_beats": 1,  # One Z per block; divisors of 16 only.
    "draft_dropout": 0.0,
    "draft_norm_eps": 1e-5,
    "draft_difficulty_vocab_size": 8,
    "draft_level_center": 10.0,
    "draft_level_scale": 5.0,
    "draft_span_scale": 16.0,
    "planner_dim": 128,
    "planner_heads": 4,
    "planner_mlp_dim": 256,
    "planner_dropout": 0.1,
    "planner_norm_eps": 1e-5,
    "reconstruction_ffn_dim": 512,
    "reconstruction_dropout": 0.1,
    "reconstruction_norm_eps": 1e-5,
    "pretrain_batch_reference_rows": 207,  # Existing pilot scheduling proxy, never a query capacity.
    "reconstruction_position_base": 10000.0,
    "reconstruction_position_scale": 1.0,
    "reconstruction_label_smoothing": 0.05,
    "reconstruction_classification_weight": 1.0,
    "reconstruction_moment_weight": 1.0,
    "reconstruction_empty_weight": 1.0,
    "reconstruction_positive_count_weight": 1.0,
    "planner_continuous_weight": 1.0,
    "planner_category_weight": 0.25,
    "pretrain_optimizer": "adamw",
    "pretrain_learning_rate": 2e-4,
    "pretrain_weight_decay": 0.01,
    "pretrain_gradient_clip": 1.0,
    "pretrain_warmup_steps": 200,  # Optimizer updates, not microbatches or windows.
    "pretrain_scheduler_epochs": 100,
    "pretrain_stop_after_epochs": 50,  # Absolute completed epoch, including after resume.
    "pretrain_precision": "bf16",
    "pretrain_initialization": "scratch",
    "pretrain_audio_mode": "joint",  # joint: update E/D/R/P; cached: fixed P0, update D/R/P.
    "pretrain_p0_cache": None,  # New public P0 export plus its frozen Encoder snapshot.
    "pretrain_seed": 20261010,
    "pretrain_audio_groups_per_batch": 2,
    "pretrain_gradient_accumulation_steps": 1,
    "pretrain_token_budget": 45056,
    "pretrain_audio_token_weight": 1.0,
    "pretrain_row_token_weight": 1.0,
    "pretrain_workers": 12,
    "pretrain_device": "cuda",
    "pretrain_train_plan": "../artifacts/pilot320-encoder-draft-sampling-20261010/train",
    "pretrain_validation_plan": "../artifacts/pilot320-encoder-draft-sampling-20261010/validation",
    "pretrain_chart_cache": "../artifacts/chart-cache-original-v2-master-remaster-fields-20261010",
    "pretrain_audio_cache": "../artifacts/encoder-audio-cache-all-official-20261010",
    "pretrain_mel_root": "../../repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired/audio/features",
    "pretrain_source_root": "../../data/extracted",
    "pretrain_output": "../artifacts/pilot320-encoder-draft-training-20261010",
    "decoder_dim": 256,
    "decoder_layers": 6,
    "decoder_router_layer": 2,
    "decoder_heads": 8,
    "decoder_shared_ffn_dim": 1024,
    "decoder_expert_ffn_dim": 512,
    "decoder_experts": 4,
    "decoder_readout_ffn_dim": 256,
    "decoder_dropout": 0.1,
    "decoder_norm_eps": 1e-5,
    "decoder_rope_base": 10000.0,
    "decoder_rope_position_scale": 1.0,
    "decoder_condition_gate_init": 1e-3,
    "decoder_z_cross_attention": True,
    "decoder_note_dims": (64, 128, 16, 16, 16, 8, 8),
    "decoder_route_dims": (240, 16),
    "decoder_time_base": 10000.0,
    "decoder_time_scale": 1.0,
    "decoder_z_before_beats": 4,
    "decoder_z_after_beats": 4,
    "decoder_history_dropout": 0.15,
    "decoder_train_checkpoint": None,  # Offline-export producer; optional in training, never initializes stage-II E.
    "decoder_train_p0_cache": None,  # Frozen public P0 for cached adaptation; unused by joint training.
    "decoder_train_local_p0_cache": None,  # Exact per-window P0 (including context/tail) for M adaptation.
    "decoder_train_audio_mode": "joint",  # joint: fresh E/M; cached: frozen E, warm-start M.
    "decoder_train_init_checkpoint": None,  # Weight initialization only; never an optimizer/epoch resume.
    "decoder_train_dp_cache": None,  # Matching frozen Z and predicted plan12 export.
    "decoder_train_plan_source": "gt",  # Match GT charts in training; inference reads predicted plan12.
    "decoder_train_gt_plan_cache": None,  # Required only for gt; offline MusicPlanTargets export.
    "decoder_train_optimizer": "adamw",
    "decoder_train_fused_optimizer": True,  # CUDA backend; optimizer moments are preserved on resume.
    "decoder_train_learning_rate": None,
    "decoder_train_weight_decay": None,
    "decoder_train_gradient_clip": None,
    "decoder_train_warmup_steps": 200,  # Optimizer updates.
    "decoder_train_scheduler_epochs": None,
    "decoder_train_stop_after_epochs": None,
    "decoder_train_precision": None,
    "decoder_train_gradient_accumulation_steps": 1,  # One batch per optimizer update.
    "decoder_train_row_budget": 8192,  # Padded M inputs, including history and boundary.
    "decoder_train_chunk_token_budget": 8192,  # Independent M computation cap; never crop history.
    "decoder_train_audio_scope": "window",  # Selected local P0, including its audio context.
    "decoder_train_window_cache": None,  # Offline tensorized window inputs and supervision.
    "decoder_train_physical_weight": None,
    "decoder_train_judge_weight": None,
    "decoder_train_trajectory_weight": None,
    "decoder_train_seed": 20261010,
    "decoder_train_workers": 12,
    "decoder_train_device": "cuda",
    "decoder_train_train_plan": "../artifacts/pilot320-sampling-20261010/train",
    "decoder_train_validation_plan": "../artifacts/pilot320-sampling-20261010/validation",
    "decoder_train_chart_cache": "../artifacts/chart-cache-original-v2-master-remaster-fields-20261010",
    "decoder_train_physical_cache": "../artifacts/training-physical-v2-original-pilot320-20261010",
    "decoder_train_audio_cache": "../artifacts/encoder-audio-cache-all-official-20261010",
    "decoder_train_mel_root": "../../repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired/audio/features",
    "decoder_train_source_root": "../../data/extracted",
    "decoder_train_geometry": "../../repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired/physical/path-catalog",
    "decoder_train_slide_data": "../../tools/MiaCode/assets/reference/slide_data.json",
    "decoder_train_route_distances": None,
    "decoder_train_output": None,
    "joint_train_initialization": "scratch",
    "joint_train_z_stride_beats": 4,
    "joint_train_plan_source": "gt",
    "joint_train_optimizer": "adamw",
    "joint_train_fused_optimizer": True,
    "joint_train_learning_rate": 4e-4,
    "joint_train_weight_decay": 0.01,
    "joint_train_gradient_clip": 1.0,
    "joint_train_warmup_steps": 200,
    "joint_train_scheduler_epochs": 150,
    "joint_train_stop_after_epochs": 90,
    "joint_train_precision": "bf16",
    "joint_train_workers": 12,
    "joint_train_seed": 20261010,
    "joint_train_device": "cuda",
    "joint_train_decoder_weight": 1.0,
    "joint_train_reconstruction_weight": 1.0,
    "joint_train_planner_weight": 1.0,
    "joint_train_physical_weight": 0.8,
    "joint_train_judge_weight": 0.5,
    "joint_train_trajectory_weight": 0.5,
    "joint_train_output": "../artifacts/joint-z4-gt-20261011/training",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m maidata",
        description="Maidata components; Encoder/Draft training: python -m maidata.training.",
    )
    parser.add_argument("--show-config", action="store_true", help="show the configuration in entry.py")
    parser.add_argument("--component", choices=("encoder", "draft", "planner", "reconstruction", "decoder"),
                        default="encoder", help="build one independent component")
    parser.add_argument('--z-stride-beats', type=int, choices=(1, 2, 4, 8, 16),
                        help='beats per Z; default from CONFIG')
    args = parser.parse_args(argv)
    config = dict(CONFIG)
    if args.z_stride_beats is not None:
        config['draft_z_stride_beats'] = args.z_stride_beats
    if args.show_config:
        pprint(config, sort_dicts=False)
        return 0
    builders = {"encoder": build_model, "draft": build_draft,
                "planner": build_planner, "reconstruction": build_reconstruction, "decoder": build_decoder}
    try:
        module = builders[args.component](config)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{type(module).__name__} built. Encoder/Draft training: python -m maidata.training. "
          "Stage-II training: python -m maidata.decoder_training; generation: python -m maidata.inference.")
    return 0
