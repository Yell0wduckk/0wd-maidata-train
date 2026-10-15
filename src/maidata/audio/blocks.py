"""Shared L/S Conformer blocks and the local stride transition."""

from torch import Tensor, nn
from torch.nn import functional as F

from maidata.audio.attention import (
    apply_rope,
    global_joint_attention,
    local_joint_attention,
)
from maidata.audio.geometry import mask_values, pool_stride
from maidata.audio.types import ScaleMemory
from maidata.nn.qk_norm import normalize_qk, validate_qk_normalization


class MacaronFeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, valid: Tensor) -> Tensor:
        values = self.norm(mask_values(values, valid))
        values = self.dropout(F.silu(self.linear1(values)))
        return mask_values(self.dropout(self.linear2(values)), valid)


class TemporalConvolution(nn.Module):
    """A local-only convolution; S is never adjacent to L in this sequence."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("temporal kernel size must be positive and odd")
        self.norm = nn.LayerNorm(d_model)
        self.expand = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model
        )
        self.project = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, valid: Tensor) -> Tensor:
        values = self.norm(mask_values(values, valid))
        values = mask_values(F.glu(self.expand(values), dim=-1), valid)
        values = self.depthwise(values.transpose(1, 2)).transpose(1, 2)
        values = mask_values(F.silu(values), valid)
        return mask_values(self.dropout(self.project(values)), valid)


class SharedConformerBlock(nn.Module):
    """One parameter set, with synchronous local and slot attention branches."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        ffn_dim: int,
        temporal_kernel_size: int,
        attention_radius: float,
        dropout: float,
        rope_base: float,
        rope_position_scale: float,
        qk_normalization: bool = False,
    ) -> None:
        super().__init__()
        if heads < 1 or d_model < 1 or d_model % heads or (d_model // heads) % 2:
            raise ValueError("d_model must split into positive, even-sized attention heads")
        self.heads = heads
        self.attention_radius = attention_radius
        self.dropout_p = dropout
        self.rope_base = rope_base
        self.rope_position_scale = rope_position_scale
        self.qk_normalization = validate_qk_normalization(qk_normalization)
        self.ffn1 = MacaronFeedForward(d_model, ffn_dim, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution = TemporalConvolution(d_model, temporal_kernel_size, dropout)
        self.ffn2 = MacaronFeedForward(d_model, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def _heads(self, values: Tensor) -> Tensor:
        return values.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def _rotated(self, values: Tensor, memory: ScaleMemory) -> Tensor:
        return apply_rope(
            normalize_qk(self._heads(mask_values(values, memory.valid)), self.qk_normalization),
            memory.beats.masked_fill(~memory.valid, 0.0),
            self.rope_base,
            self.rope_position_scale,
        )

    def _key_values(self, normalized: Tensor, memory: ScaleMemory) -> tuple[Tensor, Tensor]:
        return (
            self._rotated(self.k(normalized), memory),
            self._heads(mask_values(self.v(normalized), memory.valid)),
        )

    def project_song(self, g0: ScaleMemory) -> tuple[Tensor, Tensor]:
        """Prepare static G0 K/V once; G0 never executes the Macaron FFN."""
        normalized = mask_values(self.attention_norm(mask_values(g0.values, g0.valid)), g0.valid)
        return self._key_values(normalized, g0)

    def _attention_residual(self, attended: Tensor, usable: Tensor) -> Tensor:
        values = attended.transpose(1, 2).flatten(-2)
        # A zero attention result must remain zero even when out has a bias.
        return mask_values(self.attention_dropout(self.out(values)), usable)

    def forward(
        self,
        local: ScaleMemory,
        slots: ScaleMemory,
        song_kv: tuple[Tensor, Tensor] | None,
        song_valid: Tensor | None,
        *,
        update_slots: bool, attention_layout=None,
    ) -> tuple[ScaleMemory, ScaleMemory]:
        local_input = mask_values(local.values, local.valid)
        slot_input = mask_values(slots.values, slots.valid)
        local_first = mask_values(local_input + 0.5 * self.ffn1(local_input, local.valid), local.valid)
        slot_first = mask_values(slot_input + 0.5 * self.ffn1(slot_input, slots.valid), slots.valid)
        local_normalized = mask_values(self.attention_norm(local_first), local.valid)
        slot_normalized = mask_values(self.attention_norm(slot_first), slots.valid)

        q_local = self._rotated(self.q(local_normalized), local)
        k_local, v_local = self._key_values(local_normalized, local)
        k_slots, v_slots = self._key_values(slot_normalized, slots)
        dropout_p = self.dropout_p if self.training else 0.0
        local_attended = local_joint_attention(
            q_local, k_local, v_local, k_slots, v_slots,
            local.beats.masked_fill(~local.valid, 0.0), local.valid, slots.valid,
            self.attention_radius, dropout_p, layout=attention_layout,
        )
        # Every valid local query has its own local key within the nonnegative radius.
        local_usable = local.valid
        local_value = mask_values(
            local_first + self._attention_residual(local_attended, local_usable), local.valid
        )
        local_value = mask_values(local_value + self.convolution(local_value, local.valid), local.valid)
        local_value = mask_values(local_value + 0.5 * self.ffn2(local_value, local.valid), local.valid)
        local_value = mask_values(self.final_norm(local_value), local.valid)
        local_output = ScaleMemory(local_value, local.beats, local.valid)

        if not update_slots:
            # The final L still read slot_first through K/V, but no S update has a consumer.
            return local_output, slots
        q_slots = self._rotated(self.q(slot_normalized), slots)
        slot_attended = global_joint_attention(
            q_slots, *song_kv, k_local, v_local,
            slots.valid, song_valid, local.valid, dropout_p,
        )
        slot_usable = slots.valid & (
            song_valid.any(-1, keepdim=True) | local.valid.any(-1, keepdim=True)
        )
        slot_value = mask_values(
            slot_first + self._attention_residual(slot_attended, slot_usable), slots.valid
        )
        slot_value = mask_values(slot_value + 0.5 * self.ffn2(slot_value, slots.valid), slots.valid)
        slot_value = mask_values(self.final_norm(slot_value), slots.valid)
        return local_output, ScaleMemory(slot_value, slots.beats, slots.valid)


class StridedTransition(nn.Module):
    """Stride-2 learned local branch plus mask-aware pooled residual."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, 3, stride=2, padding=1, groups=d_model)
        self.project = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, local: ScaleMemory) -> ScaleMemory:
        pooled = pool_stride(local.values, local.beats, local.valid)
        values = mask_values(local.values, local.valid).transpose(1, 2)
        values = self.depthwise(values)
        values = values.masked_fill(~pooled.valid[:, None], 0.0)
        values = self.project(F.silu(values)).transpose(1, 2)
        values = mask_values(self.dropout(values), pooled.valid)
        values = mask_values(self.norm(pooled.values + values), pooled.valid)
        return ScaleMemory(values, pooled.beats, pooled.valid)
