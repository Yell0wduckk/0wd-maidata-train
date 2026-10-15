"""Independent component factories. Defaults live only in entry.CONFIG."""

ENCODER_KEYS = (
    'mel_bins', 'samples_per_beat', 'd_model', 'heads', 'ffn_dim', 'stage_depths',
    'stage_attention_radii', 'temporal_kernel_size', 'stem_channels', 'frequency_dim',
    'frequency_queries', 'fpn_rates', 'target_beats', 'audio_before_beats',
    'audio_after_beats', 'dropout', 'rope_base', 'rope_position_scale', 'fpn_p0_normalization',
)


def encoder_config(config):
    """Old saved recipes predate P0 normalization and retain their old math."""
    return {key: config.get(key, False) if key == 'fpn_p0_normalization' else config[key]
            for key in ENCODER_KEYS}


def build_model(config):
    from .audio.encoder import AudioEncoder

    return AudioEncoder(**encoder_config(config))


def build_draft(config):
    from .draft.generator import DraftGenerator
    from .draft.spacing import z_stride
    z_stride(config['draft_z_stride_beats'], target_beats=config['target_beats'])
    keys = ('layers', 'dim', 'heads', 'experts', 'expert_ffn_dim', 'latent_dim',
            'dropout', 'norm_eps', 'difficulty_vocab_size', 'level_center', 'level_scale', 'span_scale')
    return DraftGenerator(audio_dim=config['d_model'],
        **{key: config['draft_' + key] for key in keys},
        rope_base=config['rope_base'], rope_position_scale=config['rope_position_scale'])


def build_planner(config):
    from .draft.planner import PlannerHead
    keys = ('dim', 'heads', 'mlp_dim', 'dropout', 'norm_eps')
    return PlannerHead(latent_dim=config['draft_latent_dim'], span_beats=plan_span(config),
        **{key: config['planner_' + key] for key in keys},
        rope_base=config['rope_base'], rope_position_scale=config['rope_position_scale'])


def plan_span(config):
    """Saved recipes without this field retain the original sixteen-beat Plan."""
    span = config.get('planner_span_beats', 16)
    if type(span) is not int or span not in (4, 16):
        raise ValueError('planner_span_beats must be 4 or 16')
    if span == 4 and config['draft_z_stride_beats'] != 4:
        raise ValueError('four-beat Plan requires aligned four-beat Z')
    if span == 4 and (config['target_beats'] != 16 or
            config['decoder_z_before_beats'] != 4 or config['decoder_z_after_beats'] != 4):
        raise ValueError('six aligned Plan4 tokens require a sixteen-beat target and four-beat context on each side')
    return span


def build_reconstruction(config):
    from .draft.reconstruction import ReconstructionHead
    keys = ('ffn_dim', 'dropout', 'norm_eps',
            'position_base', 'position_scale')
    return ReconstructionHead(latent_dim=config['draft_latent_dim'], target_beats=config['target_beats'],
                               z_stride_beats=config['draft_z_stride_beats'],
                               count_loss=config.get('reconstruction_count_loss', 'ce'),
                               route_vocabulary=config.get('reconstruction_route_vocabulary', 'route71'),
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
                    z_normalization=config.get('decoder_z_normalization', False),
                    plan_span_beats=plan_span(config),
                    **{key: config['decoder_' + key] for key in keys})
    return model
