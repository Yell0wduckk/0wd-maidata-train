"""Composition for R/P supervision; no Encoder or Decoder instance is owned."""
from .windows import select_windows
from .losses import reconstruction_loss, planner_loss


def predict_auxiliary(z, groups, starts, valid, reconstruction, planner, config, layouts=None, queries=None):
    """Reuse a single full-song Z graph across selected windows and both heads."""
    r_windows = select_windows(z, groups, starts, valid, target_beats=config['target_beats'],
        before_beats=0, after_beats=0, z_stride_beats=config['draft_z_stride_beats'], layout=None if layouts is None else layouts['reconstruction'])
    p_windows = select_windows(z, groups, starts, valid, target_beats=config['target_beats'],
                               before_beats=0, after_beats=0, z_stride_beats=config['draft_z_stride_beats'], layout=None if layouts is None else layouts['planner'])
    prepared = None if layouts is None else layouts.get('auxiliary')
    return reconstruction(r_windows, queries, layout=None if prepared is None else prepared['readout']), planner(p_windows)


def auxiliary_losses(reconstruction, planner, reconstruction_targets, planner_targets, config, *, prepared=None):
    """Return separately normalized terms. The training recipe chooses weights."""
    return {**reconstruction_loss(reconstruction, reconstruction_targets,
        label_smoothing=config['reconstruction_label_smoothing'], prepared=prepared),
        **planner_loss(planner, planner_targets, prepared=prepared)}
