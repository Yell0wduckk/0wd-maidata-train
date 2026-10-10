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
    "audio_before_beats": 4,
    "audio_after_beats": 4,
    "chart_history_beats": 4,
    "dropout": 0.1,
    "rope_base": 10000,
    "rope_position_scale": 1.0,
    "draft_layers": 8,
    "draft_dim": 256,
    "draft_heads": 8,
    "draft_experts": 4,
    "draft_expert_ffn_dim": 512,
    "draft_latent_dim": 128,
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
    "reconstruction_layers": 2,
    "reconstruction_dim": 128,
    "reconstruction_heads": 4,
    "reconstruction_ffn_dim": 512,
    "reconstruction_dropout": 0.1,
    "reconstruction_norm_eps": 1e-5,
    "reconstruction_query_rows": 207,  # Per target window (pilot320), including SEGMENT_END.
    "reconstruction_before_beats": 16,
    "reconstruction_after_beats": 16,
    "reconstruction_position_base": 10000.0,
    "reconstruction_position_scale": 1.0,
    "reconstruction_label_smoothing": 0.05,
    "reconstruction_smooth_l1_beta": 1.0,
    "reconstruction_classification_weight": 1.0,
    "reconstruction_time_weight": 1.0,
    "reconstruction_count_weight": 1.0,
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
    "pretrain_seed": 20261010,
    "pretrain_audio_groups_per_batch": 2,
    "pretrain_gradient_accumulation_steps": 1,
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
    "decoder_note_dims": (64, 128, 16, 16, 16, 8, 8),
    "decoder_route_dims": (240, 16),
    "decoder_time_base": 10000.0,
    "decoder_time_scale": 1.0,
    "decoder_time_components": 5,
    "decoder_z_before_beats": 4,
    "decoder_z_after_beats": 4,
    "decoder_history_dropout": 0.15,
    "decoder_time_distributions": {
        "MOMENT": {"reference": 1000., "minimum_scale": 2.5,
                   "initial_centers": (0., 100., 200., 400., 800.),
                   "initial_scales": (50., 50., 200/3, 100., 200.)},
        "HOLD_DURATION": {"reference": 1000., "minimum_scale": 2.5,
                          "initial_centers": (100., 200., 400., 800., 1600.),
                          "initial_scales": (100., 100., 400/3, 200., 400.)},
        "SLIDE_DURATION": {"reference": 96., "minimum_scale": .25,
                           "initial_centers": (0., 24., 48., 96., 192.),
                           "initial_scales": (12., 12., 16., 24., 48.)},
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m maidata",
        description="Maidata Encoder, Draft and Decoder components; training is not implemented.",
    )
    parser.add_argument("--show-config", action="store_true", help="show the configuration in entry.py")
    parser.add_argument("--component", choices=("encoder", "draft", "planner", "reconstruction", "decoder"),
                        default="encoder", help="build one independent component")
    args = parser.parse_args(argv)
    if args.show_config:
        pprint(CONFIG, sort_dicts=False)
        return 0
    builders = {"encoder": build_model, "draft": build_draft,
                "planner": build_planner, "reconstruction": build_reconstruction, "decoder": build_decoder}
    try:
        module = builders[args.component](CONFIG)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{type(module).__name__} built. End-to-end generation and training are not implemented.")
    return 0
