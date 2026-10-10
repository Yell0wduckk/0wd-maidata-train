"""Public radar conditioning and retained historical chart-style contracts."""
from copy import deepcopy
from maimai_baseline.architecture import DECODER_LAYERS
from maimai_baseline.radar_contract import RADAR_ID, DIMENSION, SCALES, identity as radar_identity
from maimai_baseline.draft_contract import BEAT_STYLE_ID, LEGACY_STYLE_ID, geometry

STYLE_ID = "full-song-z128-no-global-v1"
FROZEN_STYLE_ID = "full-song-frozen-mu128-v1"
NATIVE_STYLE_ID = "native-audio-frozen-mu128-v1"
DRAFT_STYLE_ID = LEGACY_STYLE_ID
BEAT_DRAFT_STYLE_ID = BEAT_STYLE_ID
STYLE_BEATS = 4
STYLE_SLOTS = 4
STYLE_DIM = 128
FEATURE_DIM = 128
GLOBAL_REFERENCE_BEATS = 4
GLOBAL_REFERENCE_SLOTS = 4
STYLE_SEED_XOR = 0x5354594C
STYLE_PREFIXES = ("chart_style_encoder.", "style_conditioner.", "style_prior.")
PRIOR_PREFIXES = ("style_prior.",)
STYLE_CONTRACT = {
    "schema": "maimai.chart_style.v3", "name": STYLE_ID,
    "beats_per_token": STYLE_BEATS, "tokens": STYLE_SLOTS, "dimension": STYLE_DIM,
    "boundaries": "half-open-target-window;audio-last-onset-center-v2",
    "modifiers": "role-bits-short-hold-short-touch-hold-v1",
    "global": None,
    "encoder": {"width": 128, "layers": 2, "heads": 4, "ffn": 512, "dropout": 0.1,
                "aggregation": "branch-to-note-to-moment-sum-v1", "position": "whole-song-absolute-beats-and-moment-delta", "scope": "full-song-moments-then-four-beat-pooling"},
    "decoder": {"layers": DECODER_LAYERS, "width": 256, "heads": 8, "initial_gate": 1e-3,
                "attention": "full-style-memory-beat-rope"},
    "posterior": {"logvar_bounds": [-6., 2.], "free_bits": 0.25,
                  "beta": 0.001, "warmup_fraction": 0.1, "local_kl_reduction": "valid-coordinate-mean-times128"},
    "prior": {"type": "native-song-ar-gmm4-moe", "dimension": 128, "layers": 8},
}

FROZEN_STYLE_CONTRACT = deepcopy(STYLE_CONTRACT)
FROZEN_STYLE_CONTRACT.update(schema="maimai.chart_style.frozen_mu.v1", name=FROZEN_STYLE_ID,
    representation="deterministic-mu", posterior=None,
    prior={"type": "native-song-ar-gmm4-moe", "dimension": 128, "layers": 8,
           "target": "deterministic-mu", "normalization": "mu-only-variance"})
FROZEN_STYLE_CONTRACT["encoder"].update(frozen=True, training="independent-pretraining",
    execution="cached-final-mu;no-posterior-or-sampling")


def is_draft(value):
    return (value if isinstance(value, str) else getattr(value, "chart_style", None)) in (DRAFT_STYLE_ID, BEAT_DRAFT_STYLE_ID)


def is_native(value):
    return (value if isinstance(value, str) else getattr(value, "chart_style", None)) in (NATIVE_STYLE_ID, DRAFT_STYLE_ID, BEAT_DRAFT_STYLE_ID)


def is_frozen(value):
    return (value if isinstance(value, str) else getattr(value, "chart_style", None)) in (FROZEN_STYLE_ID, NATIVE_STYLE_ID, DRAFT_STYLE_ID, BEAT_DRAFT_STYLE_ID)


def style_identity(name="none"):
    if name == "none":
        return {"schema": "maimai.chart_style.v3", "name": "none"}
    if name == RADAR_ID:
        return radar_identity()
    if is_draft(name):
        from maimai_baseline.model.audio_draft import CONTRACT, LEGACY_CONTRACT
        shape = geometry(name)
        value = dict(schema='maimai.chart_style.audio_draft.v1', name=name,
            beats_per_token=shape.beats_per_token, tokens=shape.condition_slots, dimension=128, posterior=None, encoder=None, prior=None,
            generator=deepcopy(CONTRACT if name == BEAT_DRAFT_STYLE_ID else LEGACY_CONTRACT), audio='external-full-native22-center-stitched-p0-c0-c1-v1',
            decoder=deepcopy(STYLE_CONTRACT['decoder']),
            condition_noise=dict(probability=0.5, std=0.05, scale='train-only-per-coordinate-std', scope='window-view-occurrence'),
            condition_view='learned-identity-mirror-embedding')
        if name == BEAT_DRAFT_STYLE_ID:
            value.update(schema='maimai.chart_style.audio_draft.beat.v1', target_tokens=shape.target_slots,
                condition_first_slot=shape.first_slot, condition_windows='previous-current-next')
        return value
    if is_frozen(name):
        value = deepcopy(FROZEN_STYLE_CONTRACT)
        if is_native(name):
            value.update(schema="maimai.chart_style.native_audio_mu.v1", name=NATIVE_STYLE_ID,
                         representation="layernorm-mu-eps1e-5",
                         audio="external-full-native22-center-stitched-p0-c0-c1-v1")
            value["encoder"].update(output="non-affine-layernorm-eps1e-5", training="reconstruction+0.25-audio-nll+variance")
            value["prior"].update(target="layernorm-mu-eps1e-5")
            value["condition_noise"] = dict(probability=0.5, std=0.05, scope="window-view-occurrence")
        return value
    if name != STYLE_ID:
        raise ValueError(f"unsupported chart style: {name}")
    return deepcopy(STYLE_CONTRACT)


def style_beta(progress):
    return STYLE_CONTRACT["posterior"]["beta"] * min(1., max(0., float(progress)) / 0.1)


def style_parameter_count(vocabulary, name=RADAR_ID):
    from maimai_baseline.contract import REMOVED_FIELDS
    from maimai_dataset.modifiers import LEGACY_FIELDS
    sizes = sum(len(v.values) for n, v in vocabulary.fields.items()
                if n not in (*REMOVED_FIELDS, *LEGACY_FIELDS, 'route'))
    original = (546433 + 70 * 128 + sizes * 128 + (128 * 256 + 256)
                + DECODER_LAYERS * (263168 + 512 + 1) + 14063780)
    if is_draft(name):
        import torch
        from maimai_baseline.model.audio_draft import AudioDraftGenerator, CONTRACT, LEGACY_CONTRACT
        with torch.random.fork_rng(devices=[]):
            count = sum(p.numel() for p in AudioDraftGenerator(structure=CONTRACT if name == BEAT_DRAFT_STYLE_ID else LEGACY_CONTRACT).parameters())
        return (128 * 256 + 256) + DECODER_LAYERS * (263168 + 512 + 1) + count + 2 * 128
    if name in (STYLE_ID, FROZEN_STYLE_ID, NATIVE_STYLE_ID):
        return original
    if name != RADAR_ID:
        raise ValueError('unsupported condition parameter inventory')
    # Seven-coordinate AR input/BOS and four-component mean/logvar heads;
    # retain the chart encoder and decoder conditioner, add semantic axes.
    removed = (128-DIMENSION)*(256+1+8*(256+1))
    return original-removed+DIMENSION*128+256



def style_model_contract(config, vocabulary):
    from maimai_baseline.contract import baseline_contract
    from maimai_baseline.dual_contract import style_parameter_delta
    value = baseline_contract(config.encoder_capacity, config.decoder_rope_coordinate, config.route_head)
    if config.chart_style != "none":
        value["latent_enabled"] = config.latent_enabled
        value["chart_style"] = style_identity(config.chart_style)
        value["parameter_count"] += style_parameter_count(vocabulary, config.chart_style) + style_parameter_delta()
    return value


def representation_identity(identity):
    """The frozen A representation is independent of its untrained prior."""
    if not isinstance(identity, dict):
        raise ValueError("missing style representation identity")
    return {k: deepcopy(v) for k, v in identity.items() if k != "prior"}


def validate_checkpoint_style(payload):
    config = payload.get("model_config", {})
    if is_draft(config.get('chart_style')):
        import torch
        identity = style_identity(config['chart_style'])
        state = payload.get('model_state', {})
        if (config.get('latent_enabled') is not True or payload.get('chart_style') != identity
                or payload.get('baseline_manifest', {}).get('chart_style') != identity
                or payload.get('baseline_contract', {}).get('chart_style') != identity
                or payload.get('ar_prior') or payload.get('radar_prior') or payload.get('style_diffusion')
                or any(k.startswith(('style_prior.', 'chart_style_encoder.')) for k in state)):
            raise ValueError('audio draft checkpoint condition contract mismatch')
        std = state.get('draft_condition_std')
        if not isinstance(std, torch.Tensor) or std.shape != (128,) or not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
            raise ValueError('audio draft checkpoint lacks finite training coordinate scales')
        metadata = payload.get('draft_generator')
        if not isinstance(metadata, dict) or metadata.get('structure') != identity['generator']:
            raise ValueError('audio draft checkpoint lacks its embedded generator structure')
        if payload.get('checkpoint_kind') == 'training' and (payload.get('training_config', {}).get('chart_style') != config['chart_style'] or payload.get('training_config', {}).get('latent_enabled') is not True):
            raise ValueError('audio draft training configuration mismatch')
        return
    if is_frozen(config.get('chart_style')):
        if (config.get('latent_enabled') is not True
                or payload.get('baseline_manifest', {}).get('latent_enabled') is not True
                or (payload.get('checkpoint_kind') == 'training'
                    and payload.get('training_config', {}).get('latent_enabled') is not True)):
            raise ValueError('frozen-mu checkpoint model/manifest/training state mismatch')
        if payload.get('radar_prior'):
            raise ValueError('frozen-mu checkpoint cannot contain a radar prior')
        _validate_checkpoint_style(payload, style_identity(config['chart_style']))
        return
    if config.get('chart_style') == RADAR_ID:
        if (config.get('latent_enabled') is not False
                or payload.get('baseline_manifest', {}).get('latent_enabled') is not False
                or (payload.get('checkpoint_kind') == 'training'
                    and payload.get('training_config', {}).get('latent_enabled') is not False)):
            raise ValueError('radar checkpoint must keep the chart posterior disabled')
        _validate_checkpoint_style(payload, radar_identity())
        return
    if (config.get('chart_style') != STYLE_ID or config.get('latent_enabled') is not False
            or payload.get('baseline_manifest', {}).get('latent_enabled') is not False
            or (payload.get('checkpoint_kind') == 'training'
                and payload.get('training_config', {}).get('latent_enabled') is not False)):
        raise ValueError('no-latent checkpoint model/manifest/training state mismatch')
    if payload.get('ar_prior') or bool(payload.get('model_state', {}).get('style_prior.ready', False)):
        raise ValueError('no-latent checkpoint cannot contain a trained latent prior')
    if "chart_style" not in config:
        raise ValueError("checkpoint lacks explicit chart style identity")
    _validate_checkpoint_style(payload, style_identity(config["chart_style"]))


def _validate_checkpoint_style(payload, identity):
    config = payload.get("model_config", {})
    if "chart_style" not in config:
        raise ValueError("checkpoint lacks explicit chart style identity")
    if payload.get("chart_style") != identity or payload.get("baseline_manifest", {}).get("chart_style") != identity:
        raise ValueError("checkpoint chart style identity mismatch")
    if payload.get("checkpoint_kind") == "training" and payload.get("training_config", {}).get("chart_style") != config["chart_style"]:
        raise ValueError("checkpoint chart style training configuration mismatch")
    state = payload.get("model_state", {})
    if config["chart_style"] != "none":
        import torch
        if payload.get("baseline_contract", {}).get("chart_style") != identity:
            raise ValueError("checkpoint chart style contract mismatch")
        ready = state.get("style_prior.ready")
        if any(k.startswith(('global_style_prior.','global_style_encoder.','global_style_conditioner.')) for k in state):
            raise ValueError('G modules are forbidden in this experiment')
        if not isinstance(ready, torch.Tensor) or ready.shape != () or ready.dtype != torch.bool:
            raise ValueError("checkpoint lacks style prior readiness")
        if config['chart_style'] == RADAR_ID:
            prior = payload.get('radar_prior')
            if payload.get('ar_prior') or payload.get('style_diffusion'):
                raise ValueError('radar checkpoint cannot contain retired latent prior metadata')
            if bool(ready) != bool(prior):
                raise ValueError('radar prior readiness metadata mismatch')
            if prior and (not isinstance(prior, dict) or prior.get('stage') != 'A' or prior.get('contract') != radar_identity()):
                raise ValueError('radar prior must use its A-stage public-score contract')
            scales = state.get('style_prior.radar_scales')
            if not isinstance(scales, torch.Tensor) or not torch.equal(scales.cpu(), torch.tensor(SCALES)):
                raise ValueError('radar prior public-score scales mismatch')
            return
        prior = payload.get("ar_prior")
        if bool(ready) != bool(prior):
            raise ValueError("AR readiness/provenance mismatch")
        if prior:
            from maimai_baseline.ar_contract import validate_identity
            validate_identity(prior, operation="inference")
            norm = prior["normalization"]["local"]
            for key in ("mean", "std"):
                expected = torch.tensor(norm[key], dtype=torch.float32)
                if not torch.equal(state["style_prior.latent_" + key].cpu(), expected):
                    raise ValueError("AR normalization state/provenance mismatch")
        if payload.get("style_diffusion"):
            raise ValueError("retired prior format")
