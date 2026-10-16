"""Shared small joint-training fixtures and a one-graph gradient oracle."""
from fractions import Fraction
import random

import pytest
import torch

from maidata.chart import Moment, Note, Route, SlideDuration
from maidata.decoder.windows import build_window
from maidata.decoder_training.runner import chunk_loss
from maidata.decoder_training.supervision import KEYS, objective
from maidata.decoder_training.support_cache import ChartSupports
from maidata.draft.chart_data import from_record
from maidata.draft.plan_targets import MusicPlanTargets
from maidata.draft.targets import planner_target, reconstruction_target
from maidata.draft.windows import song_coverage
from maidata.joint_training import recipe, update as updates
from maidata.time import TimingMap
from maidata.training.runner import supports

from test_decoder_training import fields, tensor_window, distances
from test_pretrain_runtime import item as auxiliary_item


@pytest.fixture
def shared_joint_config(config, tmp_path):
    return recipe.resolve(dict(config, target_beats=16, target_stride_beats=16,
        audio_before_beats=4, audio_after_beats=4, decoder_z_before_beats=4,
        decoder_z_after_beats=4, draft_z_stride_beats=4, joint_train_z_stride_beats=4,
        planner_span_beats=16, joint_train_device='cpu', joint_train_workers=0,
        joint_train_precision='fp32', joint_train_scheduler_epochs=6,
        joint_train_stop_after_epochs=6, joint_train_output=str(tmp_path/'unused'),
        dropout=0., draft_dropout=0., planner_dropout=0., reconstruction_dropout=0.,
        decoder_dropout=0., decoder_history_dropout=0., decoder_train_chunk_token_budget=10000))


def shared_joint_item(c, monkeypatch):
    """Exact chart/Physical windows plus a small legacy Plan16 GT bank."""
    base = auxiliary_item(0, c)
    coverage = song_coverage(TimingMap([(0, 120)]), 1.2, first_beat=0, target_beats=16)
    auxiliary, decoder, ids, chart_ids, plans = [], [], [], [], []
    for key in base['conditions']:
        tokens = [Moment(0), Note.make('slide', 'A2'), Route(0, 1), SlideDuration(24),
                  Moment(500), Note.make('tap', 'A1')]
        record, physical, geo = fields(monkeypatch, tokens, key=key)
        chart = from_record(record, source_end_beat=Fraction(16))
        planner = MusicPlanTargets(chart)
        physical_chart = ChartSupports(record, physical, geo)
        for start in coverage.starts(include_tail=True):
            identity = f'{key}:{start}'
            target = planner_target(planner, start*96, (start+16)*96, span_beats=16)
            auxiliary.append((identity, key, start,
                reconstruction_target(chart, start_tick=start*96, target_beats=16), target))
            ids.append(identity)
            chart_ids.append(key)
            plans.append([*target.continuous, *[float(i == target.category) for i in range(4)]])
            for occurrence in range(2 if start == 0 else 1):
                window = build_window(physical_chart.tokens, physical_chart.timing, sequence_start_tick=0,
                    start_tick=start*96, end_tick=(start+16)*96, audio_after_beats=4,
                    audio_end_seconds=1.2, history_beats=4, history_dropout=0.,
                    training=True, rng=random.Random(0))
                row = dict(identity=(identity, occurrence), chart_id=key, start=start, example=window,
                    stages=physical_chart.window(window)[0], timing=physical_chart.timing, audio_end=1.2)
                decoder.append(dict(tensor_window(row), sequence_index=len(decoder)))
    bank = dict(plan_span_beats=16, window_ids=ids, chart_ids=chart_ids,
        window_index={key: i for i, key in enumerate(ids)},
        plan_values=torch.tensor(plans).float(), valid=torch.ones(len(ids), dtype=torch.bool))
    return dict(base, coverage=coverage, windows=auxiliary, m_windows=decoder, gt_plan=bank)


def direct_update(model, batch, opt, sched, c):
    """Oracle: retain every real M graph and backpropagate one scalar sum."""
    opt.zero_grad(set_to_none=True)
    raw = batch.to('cpu')
    counts = supports(raw)
    decoder_counts = {key: sum(chunk.labels.counts[key] for chunk in raw.chunks) for key in KEYS}
    terms, conditions, _ = model.forward_shared(raw)
    values = [updates.auxiliary_objective(terms, counts, c)]
    metrics = updates.JointMetrics()
    metrics.add_auxiliary(terms, counts)
    audio = model.decoder.prepare_audio(conditions.audio, indices=conditions.audio_indices)
    for chunk in raw.chunks:
        terms, _ = chunk_loss(model, conditions, audio, chunk.to('cpu'), c, distances())
        values.append(objective(terms, decoder_counts, c)*c['joint_train_decoder_weight'])
        metrics.add_decoder(terms)
    torch.stack(values).sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), c['pretrain_gradient_clip'], error_if_nonfinite=True)
    opt.step()
    sched.step()
    return metrics
