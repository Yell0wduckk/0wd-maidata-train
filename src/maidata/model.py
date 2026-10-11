"""Independent component factories. Defaults live only in entry.CONFIG."""

ENCODER_KEYS = (
    'mel_bins', 'samples_per_beat', 'd_model', 'heads', 'ffn_dim', 'stage_depths',
    'stage_attention_radii', 'temporal_kernel_size', 'stem_channels', 'frequency_dim',
    'frequency_queries', 'fpn_rates', 'target_beats', 'audio_before_beats',
    'audio_after_beats', 'dropout', 'rope_base', 'rope_position_scale',
)


def build_model(config):
    from .audio.encoder import AudioEncoder

    return AudioEncoder(**{key: config[key] for key in ENCODER_KEYS})


def build_draft(config):
    from .draft.generator import DraftGenerator
    keys = ('layers', 'dim', 'heads', 'experts', 'expert_ffn_dim', 'latent_dim',
            'dropout', 'norm_eps', 'difficulty_vocab_size', 'level_center', 'level_scale', 'span_scale')
    return DraftGenerator(audio_dim=config['d_model'],
        **{key: config['draft_' + key] for key in keys},
        rope_base=config['rope_base'], rope_position_scale=config['rope_position_scale'])


def build_planner(config):
    from .draft.planner import PlannerHead
    keys = ('dim', 'heads', 'mlp_dim', 'dropout', 'norm_eps')
    return PlannerHead(latent_dim=config['draft_latent_dim'],
        **{key: config['planner_' + key] for key in keys},
        rope_base=config['rope_base'], rope_position_scale=config['rope_position_scale'])


def build_reconstruction(config):
    from .draft.reconstruction import ReconstructionHead
    keys = ('ffn_dim', 'dropout', 'norm_eps',
            'position_base', 'position_scale')
    return ReconstructionHead(latent_dim=config['draft_latent_dim'], target_beats=config['target_beats'],
                               **{key: config['reconstruction_' + key] for key in keys})


def decoder_time_distributions(config):
    """385-way categorical heads have no continuous mixture configuration."""
    return {name: 'relative-tick384-continue' for name in ('MOMENT', 'HOLD_DURATION', 'SLIDE_DURATION')}


def build_decoder(config):
    from .decoder.network import Decoder
    keys = ('dim', 'layers', 'router_layer', 'heads', 'shared_ffn_dim', 'expert_ffn_dim',
            'experts', 'readout_ffn_dim', 'dropout', 'norm_eps', 'rope_base', 'rope_position_scale',
            'note_dims', 'route_dims', 'time_base', 'time_scale', 'condition_gate_init', 'z_cross_attention')
    model = Decoder(audio_dim=config['d_model'], latent_dim=config['draft_latent_dim'],
                    **{key: config['decoder_' + key] for key in keys})
    return model
