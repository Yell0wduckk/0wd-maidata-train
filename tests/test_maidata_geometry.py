"""Small CPU references for coordinate, mask and local pyramid behavior."""

import pytest
import torch

from maidata.audio.geometry import (
    gather_slots,
    interpolate,
    local_neighbors,
    mask_values,
    masked_upsample,
    pool_stride,
)
from maidata.audio.pyramid import LocalPyramid
from maidata.audio.types import ScaleMemory


def memory(values, beats, valid):
    return ScaleMemory(
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(beats, dtype=torch.float32),
        torch.tensor(valid, dtype=torch.bool),
    )


def test_mask_clears_nonfinite_padding_but_preserves_real_silence():
    values = torch.tensor([[[0.0], [float("nan")], [float("inf")]]])
    valid = torch.tensor([[True, False, False]])
    assert torch.equal(mask_values(values, valid), torch.zeros_like(values))


@pytest.mark.parametrize("length", [5, 6])
def test_stride_pool_weights_valid_values_and_requires_valid_center(length):
    values = torch.arange(length, dtype=torch.float32).reshape(1, length, 1).requires_grad_()
    beats = torch.arange(length, dtype=torch.float32).unsqueeze(0) - 2.25
    valid = torch.tensor([[True, True, False, True, True, True][:length]])
    result = pool_stride(values, beats, valid)
    expected = [0.5, 0.0, sum(range(3, length)) / (length - 3)]
    torch.testing.assert_close(result.values[0, :, 0], torch.tensor(expected))
    assert result.valid.tolist() == [[True, False, True]]
    torch.testing.assert_close(result.beats[result.valid], result.values[..., 0][result.valid] - 2.25)
    assert result.beats[0, 1] == 0
    result.values.sum().backward()
    assert values.grad[0, 2, 0] == 0


def test_slot_gather_uses_absolute_negative_centers_and_accumulates_shared_gradients():
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1).requires_grad_()
    g0 = ScaleMemory(values, torch.tensor([[-2.5, -1.5, -0.5, 0.5], [9.5, 10.5, 11.5, 12.5]]),
                     torch.tensor([[True, True, False, True], [True, True, True, True]]))
    result = gather_slots(g0, torch.tensor([-1, -1, 11]), torch.tensor([0, 0, 1]), 1, 3)
    assert result.beats.tolist() == [[-1.5, -0.5, 0.5], [-1.5, -0.5, 0.5], [10.5, 11.5, 12.5]]
    assert result.valid.tolist() == [[True, False, True], [True, False, True], [True, True, True]]
    assert result.values[:, :, 0].tolist() == [[1, 0, 3], [1, 0, 3], [5, 6, 7]]
    result.values.sum().backward()
    assert values.grad[:, :, 0].tolist() == [[0, 2, 0, 2], [0, 1, 1, 1]]


def test_slot_gather_missing_centers_and_empty_song_are_masked():
    g0 = memory([[[float("nan")], [float("inf")]]], [[-0.5, 0.5]], [[False, False]])
    result = gather_slots(g0, torch.tensor([-3]), torch.tensor([0]), 1, 4)
    assert not result.valid.any()
    assert torch.equal(result.values, torch.zeros_like(result.values))


def test_local_neighbors_match_exact_dense_distance_reference():
    source = torch.tensor([-2.0, -1.0, -1.0, 0.25, 1.0, float("nan")])
    source_valid = torch.tensor([True, False, True, True, True, False])
    queries = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    indices, valid = local_neighbors(source, source_valid, queries, 1.0)
    for row, query in enumerate(queries):
        expected = torch.nonzero(source_valid & ((source - query).abs() <= 1), as_tuple=False).flatten()
        assert torch.equal(indices[row, valid[row]], expected)
    assert indices.shape[1] == 3
    assert indices.numel() < source.numel() * queries.numel()


def test_local_neighbors_empty_source_and_zero_radius():
    indices, valid = local_neighbors(torch.tensor([0.0]), torch.tensor([False]), torch.tensor([0.0]), 0)
    assert indices.shape == valid.shape == (1, 0)
    indices, valid = local_neighbors(torch.tensor([0.0, 0.0, 1.0]), torch.ones(3, dtype=torch.bool),
                                    torch.tensor([0.0, 0.5]), 0)
    assert indices[0, valid[0]].tolist() == [0, 1]
    assert not valid[1].any()


def test_local_neighbors_use_actual_fp32_distance_at_rounded_bounds():
    # q+r can include a point whose rounded difference is outside the radius,
    # or exclude one whose rounded difference is inside it after cancellation.
    for source, query, radius in [([0.3], 0.1, 0.2), ([0.10000005], -1.0, 1.1)]:
        source = torch.tensor(source)
        queries = torch.tensor([query])
        indices, valid = local_neighbors(source, torch.ones_like(source, dtype=torch.bool), queries, radius)
        expected = torch.nonzero((source - queries[0]).abs() <= radius).flatten()
        assert torch.equal(indices[0, valid[0]], expected)


def test_interpolate_has_valid_coordinate_coverage_and_ignores_padding():
    source = memory([[[float("nan")], [2], [6], [float("inf")]]],
                    [[-1, 0, 2, 3]], [[False, True, True, False]])
    result = interpolate(source, torch.tensor([[-0.1, 0, 0.5, 2, 2.1]]))
    assert result.valid.tolist() == [[False, True, True, True, False]]
    assert result.values[0, :, 0].tolist() == [0, 2, 3, 6, 0]


def test_interpolate_empty_single_and_duplicate_sources():
    empty = memory([[[float("nan")]]], [[0]], [[False]])
    result = interpolate(empty, torch.tensor([[-1.0, 0, 1]]))
    assert not result.valid.any()
    assert torch.equal(result.values, torch.zeros_like(result.values))
    single = memory([[[7]]], [[0.5]], [[True]])
    result = interpolate(single, torch.tensor([[0.0, 0.5, 1.0]]))
    assert result.valid.tolist() == [[False, True, False]]
    assert result.values[0, :, 0].tolist() == [0, 7, 0]
    duplicate = memory([[[0], [10], [20], [30]]], [[0, 1, 1, 2]], [[True] * 4])
    result = interpolate(duplicate, torch.tensor([[0, 1, 1.5, 2]]))
    assert result.valid.all()
    assert result.values[0, :, 0].tolist() == [0, 10, 25, 30]


def test_interpolation_retains_gradients_and_feature_dtype():
    values = torch.tensor([[[2.0], [6.0]]], dtype=torch.bfloat16, requires_grad=True)
    source = ScaleMemory(values, torch.tensor([[0.0, 2.0]]), torch.ones((1, 2), dtype=torch.bool))
    result = interpolate(source, torch.tensor([[1.0]]))
    assert result.values.dtype == torch.bfloat16
    result.values.float().sum().backward()
    torch.testing.assert_close(values.grad.float(), torch.full((1, 2, 1), 0.5))


def test_masked_upsample_renormalizes_valid_endpoints_and_clamps_grid_edges():
    source = memory([[[float("nan")], [6]]], [[0.25, 0.75]], [[False, True]])
    result = masked_upsample(source, torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]]))
    assert result.valid.tolist() == [[False, False, True, True, True]]
    assert result.values[0, :, 0].tolist() == [0, 0, 6, 6, 6]
    full = memory([[[2], [6]]], [[0.25, 0.75]], [[True, True]])
    result = masked_upsample(full, torch.tensor([[0.0, 1.0]]))
    assert result.values[0, :, 0].tolist() == [2, 6]


def stage_grid(length, start=-3.0, width=2, valid=True):
    beats = start + (torch.arange(length, dtype=torch.float32) + 0.5) * 3 / length
    values = beats.unsqueeze(-1).expand(length, width).unsqueeze(0).clone().requires_grad_()
    return ScaleMemory(values, beats.unsqueeze(0), torch.full((1, length), valid))


def test_pyramid_accepts_nondefault_rates_and_keeps_absolute_grid_and_gradients():
    stages = [stage_grid(length) for length in (48, 24, 12)]
    pyramid = LocalPyramid(2, (5, 3, 2), 3, 1)
    output = pyramid(stages, torch.tensor([-2]))
    assert set(vars(output)) == {"P0", "P1", "P2"}
    for result, rate in zip((output.P0, output.P1, output.P2), (5, 3, 2)):
        assert result.values.shape == (1, 3 * rate, 2)
        torch.testing.assert_close(result.beats, (-3 + (torch.arange(3 * rate) + 0.5) / rate).unsqueeze(0))
        assert result.valid.all()
    output.P0.values.sum().backward()
    assert all(stage.values.grad is not None and stage.values.grad.abs().sum() > 0 for stage in stages)
    assert all(projection.weight.grad is not None for projection in pyramid.projections)


def test_pyramid_projection_bias_cannot_revive_empty_branch_or_fine_padding():
    pyramid = LocalPyramid(2, (5, 3, 2), 3, 1)
    for projection in pyramid.projections:
        torch.nn.init.zeros_(projection.weight)
        torch.nn.init.constant_(projection.bias, 100)
    stages = [stage_grid(length, valid=False) for length in (48, 24, 12)]
    for stage in stages:
        stage.values = torch.full_like(stage.values, float("nan"))
    output = pyramid(stages, torch.tensor([-2]))
    for result in (output.P0, output.P1, output.P2):
        assert not result.valid.any()
        assert torch.equal(result.values, torch.zeros_like(result.values))
    stages[0] = stage_grid(48)
    stages[0].valid[:, :16] = False
    stages[0].valid[:, 32:] = False
    output = pyramid(stages, torch.tensor([-2]))
    base = interpolate(stages[0], output.P0.beats)
    assert torch.equal(output.P0.valid, base.valid)
    torch.testing.assert_close(output.P0.values, base.values)


def test_pyramid_rejects_unrepresentable_output_grid():
    with pytest.raises(ValueError, match="integer grid length"):
        LocalPyramid(2, (1.1, 2, 3), 3, 1)
