"""Offline loss-equivalent partitions of time support; no model or rule queries."""
import torch
from torch.nn.utils.rnn import pad_sequence


def compact_time_group(group):
    """Merge touching cells only when GT membership and all penalties agree.

    Logistic interval mass is additive. This preserves the basic normalizer,
    GT mass and each Physical complement, including disconnected supports and
    the censored infinite tail. Original decision counts/scales remain intact.
    These intervals are loss integration regions, not generation candidates.
    """
    rows = {key: [] for key in ('lower', 'upper', 'target', 'penalty', 'allowed')}
    for row, valid in enumerate(group['allowed']):
        selected = valid.nonzero().flatten()
        lower, upper = (group[key][row, selected] for key in ('lower', 'upper'))
        target, penalty = (group[key][row, selected] for key in ('target', 'penalty'))
        breaks = ((lower[1:] != upper[:-1]) | (target[1:] != target[:-1])
                  | (penalty[1:] != penalty[:-1]).any(-1))
        starts = torch.cat((selected.new_zeros(1), breaks.nonzero().flatten()+1))
        stops = torch.cat((starts[1:]-1, selected.new_full((1,), len(selected)-1)))
        rows['lower'].append(lower[starts])
        rows['upper'].append(upper[stops])
        rows['target'].append(target[starts])
        rows['penalty'].append(penalty[starts])
        rows['allowed'].append(torch.ones(len(starts), dtype=torch.bool))
    return dict(group, **{key: pad_sequence(values, batch_first=True,
        padding_value=1 if key == 'upper' else 0) for key, values in rows.items()})


def compact_labels(labels):
    return dict(labels, groups={key: compact_time_group(group) if 'lower' in group else group
                               for key, group in labels['groups'].items()})
