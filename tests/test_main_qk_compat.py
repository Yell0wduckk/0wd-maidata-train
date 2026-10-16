"""Main keeps its original Z4/Plan16 semantics while saving Q/K mathematics."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from maidata.entry import CONFIG
from maidata.model import ENCODER_KEYS, QK_KEYS, encoder_config, with_legacy_defaults
from maidata.training.frozen_p0 import FrozenP0
from maidata.training.checkpoint import SCHEMA as TEACHER_SCHEMA
from maidata.decoder_training import checkpoint
from maidata.decoder_training.producer import teacher_configuration
from maidata.decoder_training.local_p0_cache import LocalP0, SCHEMA as LOCAL_SCHEMA
from maidata.decoder_training.cache_schema import P0_SCHEMA
from maidata.inference.cache import encoder_recipe, draft_identity, identity_key


def without_qk(config):
    return {key: value for key, value in config.items() if key not in QK_KEYS}


def test_disabled_and_missing_flags_preserve_original_cache_identity():
    old = without_qk(CONFIG)
    disabled = dict(old, **{key: False for key in QK_KEYS})
    legacy_encoder = {key: old[key] for key in ENCODER_KEYS if key != 'qk_normalization'}
    assert encoder_recipe(old) == encoder_recipe(disabled) == legacy_encoder
    assert identity_key(encoder_recipe(dict(disabled, qk_normalization=True))) != identity_key(legacy_encoder)
    metadata = dict(path='unchanged.ckpt', size=123, mtime_ns=456)
    arguments = dict(difficulty=5, level=14.5, device='cpu', precision='fp32')
    legacy = draft_identity(SimpleNamespace(config=old, identity=metadata), 'same-audio', **arguments)
    off = draft_identity(SimpleNamespace(config=disabled, identity=metadata), 'same-audio', **arguments)
    assert legacy == off and identity_key(legacy) == identity_key(off)
    for key in ('draft_qk_normalization', 'planner_qk_normalization'):
        enabled = draft_identity(SimpleNamespace(config=dict(disabled, **{key: True}), identity=metadata),
                                 'same-audio', **arguments)
        assert identity_key(enabled) != identity_key(legacy)
    historical = with_legacy_defaults({})
    assert historical['reconstruction_onset_presence_supervision'] is False
    assert historical['reconstruction_onset_presence_weight'] == 1.0
    assert not any('anchor_mode' in key or key == 'planner_layout' for key in historical)


def test_teacher_missing_flags_keep_old_math_even_if_live_defaults_change(monkeypatch):
    old = without_qk(CONFIG)
    for key in QK_KEYS:
        monkeypatch.setitem(CONFIG, key, True)
    selected = teacher_configuration(dict(schema=TEACHER_SCHEMA, recipe=old))
    assert all(selected[key] is False for key in QK_KEYS)
    enabled = teacher_configuration(dict(schema=TEACHER_SCHEMA,
        recipe=dict(old, **{key: True for key in QK_KEYS})))
    assert all(enabled[key] is True for key in QK_KEYS)


@pytest.mark.parametrize('key', QK_KEYS)
def test_stage_two_rejects_parameter_free_math_change_before_weights_load(key):
    old = without_qk(CONFIG)
    payload = dict(schema=checkpoint.SCHEMA, architecture=checkpoint.ARCHITECTURE, recipe=old)
    current = dict(with_legacy_defaults(old), **{key: True})
    with patch.object(checkpoint, '_load', side_effect=AssertionError('weights loaded')):
        with pytest.raises(ValueError, match=key):
            checkpoint.restore(payload, SimpleNamespace(cached=False), None, None,
                config=current, binding={}, source={}, producer=None, steps=[], loaders=[])


@pytest.mark.parametrize('key', ['qk_normalization', 'draft_qk_normalization', 'planner_qk_normalization'])
def test_frozen_producer_recipe_compares_missing_false_and_true(key):
    old = without_qk(CONFIG)
    saved = json.loads(json.dumps(old))
    checkpoint._validate_producer(dict(recipe=saved), old, {})
    checkpoint._validate_producer(dict(recipe=saved), with_legacy_defaults(old), {})
    with pytest.raises(ValueError, match='representation'):
        checkpoint._validate_producer(dict(recipe=saved), dict(old, **{key: True}), {})


@pytest.mark.parametrize('missing_p0_normalization', [False, True])
def test_original_frozen_p0_and_local_metadata_load_without_new_keys(tmp_path, missing_p0_normalization):
    old_encoder = encoder_config(CONFIG)
    old_encoder.pop('qk_normalization')
    if missing_p0_normalization:
        old_encoder.pop('fpn_p0_normalization')
    frozen = tmp_path/'frozen'
    frozen.mkdir()
    manifest = dict(schema=P0_SCHEMA, complete=True, encoder_config=old_encoder,
        encoder=dict(sha256='encoder', path='encoder.pt'), producer=dict(sha256='producer'),
        p0_rate=old_encoder['fpn_rates'][0], precision='fp32', groups={})
    (frozen/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    source = FrozenP0(frozen)
    assert source.encoder_config['qk_normalization'] is False
    assert source.encoder_config['fpn_p0_normalization'] is (not missing_p0_normalization)
    local = tmp_path/'local'; local.mkdir()
    local_manifest = dict(schema=LOCAL_SCHEMA, complete=True, encoder_config=old_encoder,
                          audio_producer=source.identity, precision='fp32', groups={})
    (local/'manifest.json').write_text(json.dumps(local_manifest), encoding='utf-8')
    assert LocalP0(local, source).identity['audio_producer'] == source.identity
    local_manifest['encoder_config'] = dict(old_encoder, qk_normalization=True)
    (local/'manifest.json').write_text(json.dumps(local_manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen Encoder'):
        LocalP0(local, source)
