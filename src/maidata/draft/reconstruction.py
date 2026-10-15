"""Shared residual MLP: each valid field query reads exactly its owning beat-block Z."""
import torch
from torch import nn

from maidata.nn.position import sinusoidal
from .attention import positive_ints, numerical_config
from .schema import FIELD_SIZES, MOMENT_CLASSES, POSITIVE_COUNT_CLASSES, QUERY_ROLES, KINDS
from .types import ReconstructionOutput
from .route_vocabulary import route_class_count


class ReconstructionHead(nn.Module):
    def __init__(self, *, latent_dim, ffn_dim, dropout, norm_eps,
                 target_beats, position_base, position_scale, z_stride_beats, count_loss,
                 route_vocabulary='route71', anchor_mode='block-center', span_beats=None,
                 onset_presence_supervision=False):
        super().__init__()
        positive_ints(latent_dim=latent_dim, ffn_dim=ffn_dim, target_beats=target_beats)
        numerical_config(dropout, norm_eps, position_base, position_scale)
        self.latent_dim = self.dim = latent_dim
        self.target_beats = target_beats
        from .spacing import z_stride
        self.z_stride_beats = z_stride(z_stride_beats, target_beats=target_beats)
        self.anchor_mode = anchor_mode
        self.span_beats = self.z_stride_beats if span_beats is None else span_beats
        if type(self.span_beats) is not int or self.span_beats < 1:
            raise ValueError('reconstruction span must be a positive integer')
        if anchor_mode == 'integer':
            if self.z_stride_beats not in (1, 2) or self.span_beats != 4:
                raise ValueError('integer reconstruction requires stride one or two and span four')
        elif anchor_mode != 'block-center' or self.span_beats != self.z_stride_beats:
            raise ValueError('legacy reconstruction span must equal its block stride')
        self.base, self.scale = position_base, position_scale
        self.role = nn.Embedding(len(QUERY_ROLES), latent_dim)
        self.note_kind = nn.Embedding(len(KINDS)+1, latent_dim)
        self.route = nn.Embedding(72, latent_dim)
        self.condition = nn.Linear(3, latent_dim, bias=False)
        self.ordinal_encoders = nn.ModuleList(nn.Linear(latent_dim, latent_dim, bias=False)
                                             for _ in range(4))
        self.input_norm = nn.LayerNorm(latent_dim, eps=norm_eps)
        self.ffn = nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, latent_dim))
        self.norm = nn.LayerNorm(latent_dim, eps=norm_eps)
        self.route_vocabulary = route_vocabulary
        sizes = dict(FIELD_SIZES, route_id=route_class_count(route_vocabulary))
        self.field_heads = nn.ModuleDict({'field_' + name: nn.Linear(latent_dim, n)
                                         for name, n in sizes.items()})
        self.moment_head = nn.Linear(latent_dim, MOMENT_CLASSES)
        if count_loss not in ('mse', 'ce'):
            raise ValueError('unsupported reconstruction count loss')
        self.count_loss = count_loss
        if count_loss == 'mse':
            self.count_head = nn.Linear(latent_dim, 1)
        else:  # Explicit legacy checkpoint/recipe support.
            self.empty_head = nn.Linear(latent_dim, 2)
            self.positive_count_head = nn.Linear(latent_dim, POSITIVE_COUNT_CLASSES)
        if type(onset_presence_supervision) is not bool:
            raise ValueError('onset presence supervision must be boolean')
        self.onset_presence_supervision = onset_presence_supervision
        if onset_presence_supervision:
            self.onset_presence_head = nn.Linear(latent_dim, 1)

    def _positions(self, values, mask, reference):
        encoded = sinusoidal(values.masked_fill(~mask, 0).float(), self.dim,
                             self.base, self.scale).to(reference)
        return encoded.masked_fill(~mask[..., None], 0)

    def _hidden(self, x, valid, indices=None):
        # Padding is never executed as an extra prediction query.
        flat = x.flatten(0, 1)
        selected = valid.flatten().nonzero().flatten() if indices is None else indices
        active = flat.index_select(0, selected)
        hidden = self.norm(active + self.ffn(self.input_norm(active)))
        return flat.new_zeros(flat.shape).index_copy(0, selected, hidden).reshape_as(x)

    @staticmethod
    def _owners(memory, desired, enabled):
        # Window memories have ordered centers; matching is exact on the configured Z grid.
        index = torch.searchsorted(memory.beats.contiguous(), desired.contiguous())
        safe = index.clamp_max(memory.beats.shape[1]-1)
        batch = torch.arange(len(memory.values), device=desired.device)[:, None]
        valid = enabled & (index < memory.beats.shape[1]) & memory.valid[batch, safe]
        valid &= memory.beats[batch, safe] == desired
        values = memory.values[batch, safe].masked_fill(~valid[..., None], 0)
        return values, valid

    def _block_coordinates(self, beats, starts):
        stride = self.z_stride_beats
        block_start = starts[:, None] + ((beats-starts[:, None])/stride).floor()*stride
        return block_start + stride/2, beats.floor()-block_start

    def _beat_position(self, offset, valid, reference):
        # Known beat within the Z block, never an onset target. Zero at offset
        # zero preserves the exact stride-one path and its parameter inventory.
        return (self._positions(offset, valid, reference)
                - self._positions(torch.zeros_like(offset), valid, reference))

    def forward(self, windows, queries, *, layout=None):
        m, q = windows.memory, queries
        active = windows.window_valid & m.valid.any(-1)
        # Use safe desired coordinates for inert padding, including NaN placeholders.
        desired = q.beats.masked_fill(~q.valid, 0)
        if self.anchor_mode == 'integer':
            if q.owner_z_beats is None:
                raise ValueError('integer reconstruction requires explicit owner_z_beats')
            centers = q.owner_z_beats.masked_fill(~q.valid, 0)
            offset = desired.floor()-(centers-self.span_beats/2)
        else:
            centers, offset = self._block_coordinates(desired, windows.target_start)
        owned, valid = self._owners(m, centers, q.valid & active[:, None])
        roles = q.roles.masked_fill(~valid, 0)
        moment = (roles == QUERY_ROLES.index('MOMENT')) & valid
        note = ((roles == QUERY_ROLES.index('NOTE_KIND')) |
                (roles == QUERY_ROLES.index('NOTE_FLAGS'))) & valid
        route = ((roles == QUERY_ROLES.index('ROUTE_ID')) |
                 (roles == QUERY_ROLES.index('TRACK_BREAK'))) & valid
        lower_kind = ((roles == QUERY_ROLES.index('NOTE_FLAGS')) | route) & valid
        track_break = (roles == QUERY_ROLES.index('TRACK_BREAK')) & valid
        # Enforce allowed conditions by role even if a caller supplies extra values.
        known_tick = (q.origins-q.beats.floor()).masked_fill(~(note | route), 0)
        previous = q.history_gaps.masked_fill(~(moment & q.last_real_valid), 0)
        scalars = torch.stack((known_tick, previous,
                               (moment & q.last_real_valid).float()), -1)
        scalars = scalars.sign()*scalars.abs().log1p()
        x = owned + self.role(roles) + self.condition(scalars.to(owned))
        if self.span_beats > 1:
            x = x + self._beat_position(offset, valid, owned)
        for encoder, values, mask in zip(self.ordinal_encoders,
                (q.ordinals, q.note_ordinals, q.tracks, q.segments),
                (moment | note | route, note | route, route, route)):
            x = x + encoder(self._positions(values, mask, owned))
        kind_ids = q.note_kinds.masked_fill(~lower_kind, -1)+1
        x = x + self.note_kind(kind_ids).masked_fill(~lower_kind[..., None], 0)
        current = q.current_route_ids.masked_fill(~track_break, -1)+1
        x = x + self.route(current).masked_fill(~track_break[..., None], 0)
        prefix_valid = q.route_valid & route[..., None]
        prefix_ids = q.route_ids.masked_fill(~prefix_valid, 0)+1
        prefix_positions = torch.arange(q.route_ids.shape[-1], device=owned.device).float()
        order = sinusoidal(prefix_positions, self.dim, self.base, self.scale).to(owned)
        prefix = self.route(prefix_ids)*(1+order)
        x = x + prefix.masked_fill(~prefix_valid[..., None], 0).sum(-2)
        hidden = self._hidden(x, valid, None if layout is None else layout['query_indices'])
        count_beats = windows.target_start[:, None] + torch.arange(self.target_beats, device=owned.device)+.5
        if self.anchor_mode == 'integer':
            # COUNT remains an independent per-real-beat head; the new recipe
            # keeps its supervision disabled, without changing head inventory.
            centers = (windows.target_start[:, None]
                       + ((count_beats.floor()-windows.target_start[:, None])/self.z_stride_beats).floor()
                       * self.z_stride_beats)
            offset = count_beats.floor()-(centers-self.span_beats/2)
        else:
            centers, offset = self._block_coordinates(count_beats, windows.target_start)
        count_z, count_valid = self._owners(m, centers, active[:, None].expand(-1, self.target_beats))
        count_x = count_z+self.role.weight[QUERY_ROLES.index('COUNT')]
        if self.span_beats > 1:
            count_x = count_x + self._beat_position(offset, count_valid, count_z)
        count_hidden = self._hidden(count_x, count_valid, None if layout is None else layout['count_indices'])
        def predict(head, h, mask):
            return head(h).masked_fill(~mask[..., None], 0)
        presence_logits = presence_valid = None
        if self.onset_presence_supervision:
            presence = q.onset_presence
            if presence is None:
                raise ValueError('onset presence readout requires independent geometry queries')
            presence_z, presence_valid = self._owners(m, presence.owner_z_beats,
                presence.valid & active[:, None])
            offset = presence.beats-(presence.owner_z_beats-self.span_beats/2)
            presence_x = (presence_z + self.role.weight[QUERY_ROLES.index('COUNT')]
                          + self._beat_position(offset, presence_valid, presence_z))
            presence_hidden = self._hidden(presence_x, presence_valid,
                None if layout is None else layout['onset_presence_indices'])
            presence_logits = predict(self.onset_presence_head, presence_hidden, presence_valid).squeeze(-1)
        return ReconstructionOutput(
            {name.removeprefix('field_'): predict(head, hidden, valid)
             for name, head in self.field_heads.items()},
            predict(self.moment_head, hidden, valid),
            predict(self.empty_head, count_hidden, count_valid) if self.count_loss == 'ce' else None,
            predict(self.positive_count_head, count_hidden, count_valid) if self.count_loss == 'ce' else None,
            active, valid, count_valid,
            predict(self.count_head, count_hidden, count_valid).squeeze(-1) if self.count_loss == 'mse' else None,
            presence_logits, presence_valid)
