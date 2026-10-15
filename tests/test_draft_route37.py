"""Draft-only mirror grouping, real CE/gradient/diagnostics and M isolation."""
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

from maidata.entry import CONFIG
from maidata.chart import Note, Route, SlideDuration
from maidata.chart.routes import catalog
from maidata.draft.chart_data import ChartData, OnsetGroup
from maidata.draft.targets import reconstruction_target, collate_reconstruction
from maidata.draft.types import BeatMemory, WindowLatents
from maidata.draft.losses import reconstruction_loss
from maidata.draft.route_vocabulary import AXIS15_GROUPS, ROUTE71_TO_AXIS15, route_targets
from maidata.model import build_reconstruction, build_decoder
from maidata.training.runner import Diagnostics
from maidata.time import TimingMap


def test_all_routes_group_by_start_opposite_axis_for_every_start():
    rows = catalog()
    assert len(AXIS15_GROUPS) == 37
    assert sorted(i for group in AXIS15_GROUPS for i in group) == list(range(71))
    assert [group for group in AXIS15_GROUPS if len(group) == 1] == [(2,), (55,), (70,)]
    for start in range(1, 9):
        inverse = {v['paths_by_start'][start-1]: v['variant_id'] for v in rows}
        for i, row in enumerate(rows):
            path = row['paths_by_start'][start-1].translate(str.maketrans('<>pqsz', '><qpzs'))
            reflected = ''.join(str((2*start-int(ch)-1) % 8+1) if ch.isdigit() else ch for ch in path)
            assert ROUTE71_TO_AXIS15[i] == ROUTE71_TO_AXIS15[inverse[reflected]]


def example(vocabulary='axis15-37'):
    torch.set_num_threads(1)
    config = dict(CONFIG, draft_dim=8, draft_latent_dim=8, target_beats=4,
                  draft_z_stride_beats=1, reconstruction_ffn_dim=12,
                  reconstruction_dropout=0., reconstruction_route_vocabulary=vocabulary)
    tokens = []
    for route in range(71):
        tokens.extend((Note.make('slide', 'A1'), Route(route, False), SlideDuration(96)))
    chart = ChartData((OnsetGroup(0, tuple(tokens)),), (), TimingMap([(0, 120)]), Fraction(4))
    target = collate_reconstruction([reconstruction_target(chart, start_tick=0, target_beats=4)],
                                    target_beats=4, device='cpu')
    memory = WindowLatents(BeatMemory(torch.randn(1, 4, 8, requires_grad=True),
        torch.arange(4)[None].float()+.5, torch.ones(1, 4, dtype=torch.bool)),
        torch.tensor([0.]), 4, torch.tensor([True]))
    head = build_reconstruction(config).eval()
    return head, target, memory


@pytest.mark.parametrize('vocabulary,width', [('axis15-37', 37), ('route71', 71)])
@pytest.mark.parametrize('prepared', [False, True])
def test_real_readout_loss_gradients_and_diagnostics(vocabulary, width, prepared):
    head, target, memory = example(vocabulary)
    output = head(memory, target.queries)
    assert output.field_logits['route_id'].shape[-1] == width
    assert output.route_vocabulary == vocabulary
    raw_labels = target.fields['route_id'].clone()
    selectors = dict(indices={
        **{'field/'+name: (labels != -100).flatten().nonzero().flatten()
           for name, labels in target.fields.items()},
        'moment96': (target.moment_tick != -100).flatten().nonzero().flatten()})
    terms = reconstruction_loss(output, target, label_smoothing=.05, count_supervision=False,
                                prepared=selectors if prepared else None)
    mask = raw_labels != -100
    labels = route_targets(raw_labels, vocabulary)
    expected = F.cross_entropy(output.field_logits['route_id'][mask].float(), labels[mask],
                               label_smoothing=.05, reduction='sum')
    torch.testing.assert_close(terms['field/route_id'].total, expected)
    assert terms['field/route_id'].count == 71
    assert torch.equal(target.fields['route_id'], raw_labels)
    assert target.queries.current_route_ids.max() == 70  # Known prefixes remain exact Route-71.
    expected.backward()
    assert torch.isfinite(head.field_heads['field_route_id'].weight.grad).all()
    assert head.field_heads['field_route_id'].weight.grad.abs().sum() > 0
    assert memory.memory.values.grad.abs().sum() > 0
    diagnostics = Diagnostics()
    diagnostics.add(output, target)
    report = diagnostics.report()
    assert report['route_vocabulary'] == vocabulary
    assert len(report['fields']['route_id']['confusion']) == width
    assert report['fields']['route_id']['count'] == 71
    assert report['fields']['route_id']['class_support'] == (
        [len(group) for group in AXIS15_GROUPS] if width == 37 else [1]*71)


def test_merged_mirror_labels_have_identical_ce_and_ignored_slots_stay_ignored():
    raw = torch.tensor([[0, 4, 2, 55, 70, -100]])
    assert route_targets(raw, 'axis15-37').tolist() == [[0, 0, 2, 28, 36, -100]]
    assert route_targets(raw, 'route71') is raw
    with pytest.raises(ValueError, match='vocabulary'):
        route_targets(raw, 'unknown')


def test_legacy_recipe_builds_71_head_and_rejects_loading_into_37_head():
    old = dict(CONFIG)
    old.pop('reconstruction_route_vocabulary')
    legacy = build_reconstruction(old)
    merged = build_reconstruction(CONFIG)
    assert legacy.field_heads['field_route_id'].out_features == 71
    assert merged.field_heads['field_route_id'].out_features == 37
    with pytest.raises(RuntimeError, match='size mismatch'):
        merged.load_state_dict(legacy.state_dict(), strict=True)
    from maidata.training.checkpoint import recipe as stage_recipe
    from maidata.joint_training.checkpoint import recipe as joint_recipe
    assert stage_recipe(CONFIG)['reconstruction_route_vocabulary'] == 'axis15-37'
    assert joint_recipe(CONFIG)['reconstruction_route_vocabulary'] == 'axis15-37'


def test_main_decoder_weights_and_71_way_route_head_are_unchanged():
    from test_decoder_network import config
    a = dict(config(), reconstruction_route_vocabulary='route71')
    b = dict(a, reconstruction_route_vocabulary='axis15-37')
    torch.manual_seed(44)
    original = build_decoder(a)
    torch.manual_seed(44)
    merged = build_decoder(b)
    assert merged.heads.route.out_features == 71
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, merged.state_dict()[name], atol=0, rtol=0)
