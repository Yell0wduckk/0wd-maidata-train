"""Composition for R/P supervision; no Encoder or Decoder instance is owned."""
from .windows import select_windows
from .losses import (reconstruction_loss, planner_loss, count_supervision_enabled,
                     onset_presence_supervision_enabled)


def prepare_auxiliary(items, config, ownership, draft):
    """Common CPU collation/layouts for the pretrainer and shared joint trainer."""
    import torch
    from .layout import prepare_selection, prepare_centers
    from .targets import collate_reconstruction, collate_planner
    from .losses import prepare_supervision
    windows = [window for item in items for window in item['windows']]
    groups = torch.tensor([ownership[window[1]] for window in windows], dtype=torch.long)
    starts = torch.tensor([window[2] for window in windows], dtype=torch.float32)
    valid = torch.ones(len(windows), dtype=torch.bool)
    mode = config.get('draft_z_anchor_mode', 'block-center')
    r_layout = prepare_selection(draft.query_beats, draft.draft_valid, groups, starts, valid,
        target_beats=config['target_beats'], before_beats=0, after_beats=0,
        z_stride_beats=config['draft_z_stride_beats'], anchor_mode=mode)
    if config.get('planner_layout', 'legacy') == 'centered-overlap':
        p_layout = prepare_centers(draft.query_beats, draft.draft_valid, groups,
                                  starts[:, None]+torch.arange(-8, 25).float(), valid)
    elif config.get('planner_layout', 'legacy') == 'window-single':
        from .spacing import integer_offsets
        offsets = torch.tensor(integer_offsets(0, 16, config['draft_z_stride_beats'])).float()
        p_layout = prepare_centers(draft.query_beats, draft.draft_valid, groups,
                                  starts[:, None]+offsets, valid)
    else:
        p_layout = prepare_selection(draft.query_beats, draft.draft_valid, groups, starts, valid,
            target_beats=config['target_beats'], before_beats=0, after_beats=0,
            z_stride_beats=config['draft_z_stride_beats'], anchor_mode=mode)
    reconstruction = collate_reconstruction([window[3] for window in windows],
        target_beats=config['target_beats'], device='cpu', validate_input=False)
    planner = collate_planner([window[4] for window in windows], device='cpu', validate_input=False)
    return dict(groups=groups, starts=starts, valid=valid, reconstruction=reconstruction, planner=planner,
        window_ids=[window[0] for window in windows],
        layouts=dict(reconstruction=r_layout, planner=p_layout,
                     auxiliary=prepare_supervision(reconstruction, planner)))


def predict_auxiliary(z, groups, starts, valid, reconstruction, planner, config, layouts=None, queries=None):
    """Reuse a single full-song Z graph across selected windows and both heads."""
    r_windows = select_windows(z, groups, starts, valid, target_beats=config['target_beats'],
        before_beats=0, after_beats=0, z_stride_beats=config['draft_z_stride_beats'],
        anchor_mode=config.get('draft_z_anchor_mode', 'block-center'),
        layout=None if layouts is None else layouts['reconstruction'])
    from .planner import select_planner_windows
    p_windows = select_planner_windows(z, groups, starts, valid, config,
                                     layout=None if layouts is None else layouts['planner'])
    prepared = None if layouts is None else layouts.get('auxiliary')
    return reconstruction(r_windows, queries, layout=None if prepared is None else prepared['readout']), planner(p_windows)


def auxiliary_losses(reconstruction, planner, reconstruction_targets, planner_targets, config, *, prepared=None):
    """Return separately normalized terms. The training recipe chooses weights."""
    return {**reconstruction_loss(reconstruction, reconstruction_targets,
        label_smoothing=config['reconstruction_label_smoothing'], prepared=prepared,
        route_loss=config.get('reconstruction_route_loss', 'ce'),
        count_supervision=count_supervision_enabled(config),
        onset_presence_supervision=onset_presence_supervision_enabled(config)),
        **planner_loss(planner, planner_targets, prepared=prepared)}
