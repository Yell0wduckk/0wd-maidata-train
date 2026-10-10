"""Import one completed draft model; embed its weights and training scales in A."""
from copy import deepcopy
import torch


def import_draft(model, checkpoint, cache_path, *, teacher_audio):
    from maimai_baseline.training.audio_draft_checkpoint import load_draft
    from maimai_baseline.data.draft_cache import AudioDraftCache
    from maimai_baseline.data.teacher_audio import draft_producer_identity
    from maimai_baseline.runtime_versions import operational
    from maimai_baseline.draft_contract import geometry
    if checkpoint is None:
        raise ValueError('fresh draft A requires --draft-checkpoint')
    generator, payload = load_draft(checkpoint, device='cpu')
    cache = AudioDraftCache(cache_path)
    if (geometry(model.config) != geometry(payload['structure'])
            or geometry(model.config) != cache.geometry
            or model.draft_generator.structure != payload['structure']):
        raise ValueError('selected A, imported draft, and draft cache geometry contracts differ')
    if cache.producer.get('generator_structure') != payload['structure']:
        raise ValueError('draft cache and imported generator structure differ')
    binding = payload.get('data_binding') or {}
    expected_teacher = binding.get('native_audio', binding.get('teacher_audio'))
    if expected_teacher is not None and draft_producer_identity(expected_teacher) != draft_producer_identity(teacher_audio):
        raise ValueError('draft and A frozen audio producers differ')
    if draft_producer_identity(cache.producer['native_audio']) != draft_producer_identity(teacher_audio):
        raise ValueError('draft cache and A frozen audio producers differ')
    if cache.producer.get('completed_epochs') != payload.get('completed_epochs'):
        raise ValueError('draft cache and imported completed generator differ')
    if (cache.producer.get('seed') != generator.seed
            or operational(cache.producer.get('recipe')) != operational(payload.get('recipe'))
            or operational({key: value for key, value in (cache.producer.get('data_binding') or {}).items() if key not in ('native_audio', 'teacher_audio')}) != operational({key: value for key, value in binding.items() if key not in ('native_audio', 'teacher_audio')})):
        raise ValueError('draft cache and exported generator training contract differ')
    # Actual frozen tensors are embedded; no training-cache dependency at inference.
    model.draft_generator.load_state_dict(generator.state_dict(), strict=True)
    model.draft_generator.requires_grad_(False).eval()
    std = torch.as_tensor(cache.normalization.get('local', cache.normalization)['std'], dtype=torch.float32)
    if std.shape != (128,) or not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise ValueError('draft training coordinate scales are invalid')
    model.draft_condition_std.copy_(std)
    metadata = dict(structure=deepcopy(payload['structure']), data_binding=deepcopy(binding),
        completed_epochs=payload['completed_epochs'], seed=generator.seed, recipe=deepcopy(payload.get('recipe')))
    model.draft_generator_metadata = metadata
    return deepcopy(metadata)
