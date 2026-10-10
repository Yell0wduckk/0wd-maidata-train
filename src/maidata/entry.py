"""Single editable architecture configuration and component entry point."""

import argparse
from pprint import pprint

from .model import build_model, build_draft, build_planner, build_reconstruction

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
    "reconstruction_query_rows": None,  # Set from new-field lengths including SEGMENT_END.
    "reconstruction_before_beats": 16,
    "reconstruction_after_beats": 16,
    "reconstruction_position_base": 10000.0,
    "reconstruction_position_scale": 1.0,
    "reconstruction_label_smoothing": 0.05,
    "reconstruction_smooth_l1_beta": 1.0,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m maidata",
        description="Maidata Encoder and Draft components; training is not implemented.",
    )
    parser.add_argument("--show-config", action="store_true", help="show the configuration in entry.py")
    parser.add_argument("--component", choices=("encoder", "draft", "planner", "reconstruction"),
                        default="encoder", help="build one independent component")
    args = parser.parse_args(argv)
    if args.show_config:
        pprint(CONFIG, sort_dicts=False)
        return 0
    builders = {"encoder": build_model, "draft": build_draft,
                "planner": build_planner, "reconstruction": build_reconstruction}
    try:
        module = builders[args.component](CONFIG)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{type(module).__name__} built. Decoder and training are not implemented.")
    return 0
