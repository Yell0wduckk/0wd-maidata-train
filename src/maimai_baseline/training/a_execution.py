"""Stage A execution identity; no historical checkpoint migration."""
from copy import deepcopy

POLICY = dict(schema='maimai.dual_decoder.p2_song_audio_a.v1',
              charts='metadata-only-role-record-view;no-posterior-tensors',
              attention='decoder-causal-song-history;full-p0;local-router',
              pooling='disabled',
              audio='unique-continuous-song-variants;local-router-slices;target-delta-candidates',
              song_audio='source-timing-preprocessing-unique;one-padded-complete-song-batch',
              activation_checkpointing=False, posterior='disabled;no-encoder-head-sampling-or-kv',
              checkpoints='complete-epochs-only')


def identity(inputs):
    source = getattr(inputs, 'source_identity', inputs.manifest.get('source_identity'))
    policy = deepcopy(POLICY)
    if inputs.manifest.get('teacher_audio') is not None:
        policy.update(schema='maimai.dual_decoder.native_audio_a.v1',
            audio='external-native22-window-p0;center-stitched-song-p0;cached-c0-c1-trainable-candidates',
            song_audio='external-frozen-full-a90;zero-training-encoder-calls',
            posterior='frozen-layernorm-mu;window-shared-probability0.5-std0.05-noise')
    if inputs.manifest.get('draft_cache') is not None:
        policy.update(schema='maimai.dual_decoder.audio_draft_a.v1',
            posterior='absent;frozen-audio-only-draft;train-std-scaled-window-noise',
            draft_generator='embedded-frozen;zero-training-forward-calls')
    return dict(policy=policy, source=deepcopy(source),
                inputs=deepcopy(inputs.manifest), plan=deepcopy(inputs.manifest.get('data_strategy', {}).get('plan_cache')))


def validate_resume(payload, inputs):
    if payload.get('checkpoint_kind') != 'training' or payload.get('style_diffusion') or payload.get('diffusion_training'):
        raise ValueError('batched A requires its own training checkpoint')
    offset = int(payload.get('training_config', {}).get('plan_offset', 0))
    epoch = payload.get('epoch')
    if type(epoch) is not int or epoch < 0 or offset + epoch >= inputs.plan_store.available_epochs:
        raise ValueError('invalid completed A epoch')
    expected = sum(inputs.plan_store.batch_count(e) for e in range(offset, offset + epoch + 1))
    if (payload.get('step') != expected or payload.get('sampler_epoch') != epoch
            or payload.get('training_config', {}).get('schedule_progress') != expected):
        raise ValueError('A checkpoint must be at the end of a complete epoch')
