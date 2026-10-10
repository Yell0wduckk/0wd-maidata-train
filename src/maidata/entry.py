"""Editable encoder configuration and command-line entry point."""

import argparse
from pprint import pprint

from .model import build_model

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
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m maidata",
        description="Maidata audio encoder; training is not implemented.",
    )
    parser.add_argument("--show-config", action="store_true", help="show the configuration in entry.py")
    args = parser.parse_args(argv)
    if args.show_config:
        pprint(CONFIG, sort_dicts=False)
        return 0
    build_model(CONFIG)
    print("AudioEncoder built. Decoder and training are not implemented.")
    return 0
