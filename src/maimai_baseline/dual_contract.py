"""Output-family routing is distinct from the Route-71 geometry vocabulary."""
from copy import deepcopy
from maimai_baseline.architecture import DECODER_LAYERS
import torch

TIME, NOTE, INVALID = 0, 1, -1
POLICY = dict(schema='maimai.mmoe_decoder_router.v1', layers=DECODER_LAYERS,
    shared_layers=list(range(DECODER_LAYERS)), expert_layers=list(range(DECODER_LAYERS)),
    shared_block='attention-norms-and-latent-gate', width=256, heads=8,
    feedforward=512, experts_per_layer=4, task_gates=2, mixing='dense-softmax',
    router=dict(layers=1, width=128, heads=4, feedforward=512, seed_xor=0x524F5554),
    routing='next-output-family', history='full-causal-selected-task-kv',
    condition_kv='shared-per-layer', initialization='fresh-independent-ffn-uniform-gates',
    router_input='embedding-independent-of-decoder', control_family=TIME,
    event_loss='sum-of-legal-family-and-conditional-event-ce', smoothing=.05,
    physical='whole-family-router-otherwise-selected-note-event-v1')


def identity():
    from maimai_baseline.row_field_contract import IDENTITY
    return dict(deepcopy(POLICY),row_fields=deepcopy(IDENTITY))


def parameter_delta(vocabulary=None, style=False):
    # Replace each dense FFN with 4x512 experts and two gates per layer.
    events = 8 if vocabulary is None else len(vocabulary.fields['event'].values)
    from maimai_baseline.row_field_contract import parameter_count
    return DECODER_LAYERS * (4 * 262912 + 2 * 1028 - 525568) + 512 + (4 - events) * 257 + 397059 + style_parameter_delta(style) + parameter_count(vocabulary)


def style_parameter_delta(enabled=True):
    return 0  # Shared latent attentions and gates match the dense initializer.


def event_family(events, vocabulary):
    result = torch.full_like(events, TIME)
    result.masked_fill_(events.eq(vocabulary.event_id('NOTE')) |
                        events.eq(vocabulary.event_id('SLIDE_BRANCH')), NOTE)
    return result


def prefix_families(events, padding, vocabulary, *, targets=None):
    """Known next input labels the history; the last next row is not yet known."""
    result = torch.full_like(events, INVALID)
    if events.shape[1] > 1:
        result[:, :-1] = event_family(events[:, 1:], vocabulary).masked_fill(padding[:, 1:], INVALID)
    if targets is not None:
        active = targets.ne(-100) & ~padding
        result = torch.where(active, event_family(targets, vocabulary), result)
    return result.masked_fill(padding, INVALID)


def family_support(event_allowed, vocabulary):
    return torch.stack([event_allowed[..., [vocabulary.event_id(e) for e in names]].any(-1)
        for names in (('MOMENT', 'SEGMENT_END'), ('NOTE', 'SLIDE_BRANCH'))], -1)
