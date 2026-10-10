"""Encoder routing, shared song graphs, shapes and actual input boundaries."""

from copy import deepcopy
from dataclasses import fields, replace
from unittest.mock import patch

import pytest
import torch

from maidata.audio.encoder import AudioEncoder
from maidata.audio.types import EncoderOutput, SongInputs, WindowInputs
from maidata.entry import CONFIG
from maidata.model import build_model


def scales(output):
    return (output.P0, output.P1, output.P2)


def single_window(inputs, index):
    return WindowInputs(**{field.name: getattr(inputs, field.name)[index:index + 1] for field in fields(inputs) if field.name != "layout"})


def test_default_shapes_and_public_output():
    torch.manual_seed(4)
    model = build_model(CONFIG).eval()
    c, r, m = 24, CONFIG["samples_per_beat"], CONFIG["mel_bins"]
    centers = torch.arange(c).float()[None] - 3.5
    beats = (centers[..., None] - 0.5 + (torch.arange(r).float() + 0.5) / r).flatten(1)
    song = SongInputs(torch.zeros(1, c, r, m), torch.ones(1, c, r, dtype=torch.bool),
                      centers, torch.zeros(1, c, 32))
    window = WindowInputs(song.mel_patches.flatten(1, 2), torch.zeros(1, c * r, 32),
                          beats, song.sample_valid.flatten(1), torch.tensor([0]), torch.tensor([0]))
    with torch.no_grad():
        output = model(song, window)
    assert isinstance(model, AudioEncoder)
    assert [field.name for field in fields(EncoderOutput)] == ["P0", "P1", "P2"]
    for memory, length in zip(scales(output), (192, 96, 48)):
        assert memory.values.shape == (1, length, 256)
        assert memory.valid.any()  # Silent audio remains valid.
        assert torch.isfinite(memory.values).all()
        torch.testing.assert_close(memory.values[~memory.valid], torch.zeros_like(memory.values[~memory.valid]))


def test_nondefault_configuration_forward_backward(encoder_config, audio_inputs):
    model = build_model(encoder_config).eval()
    output = model(*audio_inputs)
    for memory, rate in zip(scales(output), encoder_config["fpn_rates"]):
        assert memory.values.shape == (3, 6 * rate, 24)
        assert memory.beats.dtype == torch.float32
        assert memory.valid.dtype == torch.bool
        assert torch.isfinite(memory.values).all()
    sum(memory.values.square().mean() for memory in scales(output)).backward()
    for module in (model.song_tokenizer.project, model.spectral.stem,
                   model.feature_fusion.project, model.transitions[0].depthwise):
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    assert model.stages[0][0].q.weight is not model.stages[0][1].q.weight


def test_cpu_autocast_with_dropout_preserves_coordinates_masks_and_gradients(encoder_config, audio_inputs):
    model = build_model(dict(encoder_config, dropout=.15, stage_attention_radii=(.5, 1.5, 3.),
                             rope_base=317., rope_position_scale=.7)).train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(*audio_inputs)
        loss = sum((scale.values.float() * torch.randn_like(scale.values).float()).sum()
                   for scale in scales(output))
    loss.backward()
    for scale in scales(output):
        assert scale.values.dtype == torch.bfloat16
        assert scale.beats.dtype == torch.float32 and scale.valid.dtype == torch.bool
        assert torch.isfinite(scale.values).all()
        assert scale.values[~scale.valid].count_nonzero() == 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.song_tokenizer.project.weight.grad.abs().sum() > 0
    assert model.spectral.stem.weight.grad.abs().sum() > 0


def test_song_preparation_reused_with_matching_outputs_and_gradients(encoder_config, audio_inputs):
    torch.manual_seed(5)
    model = build_model(encoder_config).eval()
    reference = deepcopy(model)
    song, windows = audio_inputs
    blocks = [block for stage in model.stages for block in stage]
    tokenizer_calls = []
    hook = model.song_tokenizer.register_forward_hook(lambda *args: tokenizer_calls.append(1))
    # project_song wraps the complete G0 LN/K/V calculation, not its cheap routing.
    from contextlib import ExitStack
    with ExitStack() as stack:
        projectors = [stack.enter_context(patch.object(block, "project_song", wraps=block.project_song))
                      for block in blocks]
        memory = model.prepare_song(song)
        original_g0 = memory.g0.values.detach().clone()
        output = model.encode_windows(memory, windows)
        again = model.encode_windows(memory, single_window(windows, 1))
        assert [method.call_count for method in projectors] == [1, 1, 1, 0]
        assert memory.block_kv[-1] is None
        assert tokenizer_calls == [1]
        torch.testing.assert_close(memory.g0.values, original_g0)
        for whole, one in zip(scales(output), scales(again)):
            torch.testing.assert_close(whole.values[1:2], one.values, atol=2e-6, rtol=2e-5)
    hook.remove()
    probes = [torch.randn_like(scale.values) for scale in scales(output)]
    sum((memory.values * probe).sum() for memory, probe in zip(scales(output), probes)).backward()
    for index in range(3):
        separate = reference(song, single_window(windows, index))
        for whole, one in zip(scales(output), scales(separate)):
            torch.testing.assert_close(whole.values[index:index + 1], one.values, atol=2e-6, rtol=2e-5)
        sum((memory.values * probe[index:index + 1]).sum()
            for memory, probe in zip(scales(separate), probes)).backward()
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert (actual.grad is None) == (expected.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-5, rtol=3e-4, msg=name)
    assert model.song_tokenizer.project.weight.grad.abs().sum() > 0


def test_padding_cannot_change_valid_outputs_or_revive_empty_audio(encoder_config, audio_inputs):
    model = build_model(encoder_config).eval()
    song, window = audio_inputs
    reference = model(song, window)
    changed_song = replace(song,
        mel_patches=song.mel_patches.masked_fill(~song.sample_valid[..., None], 1e6),
        aux32_centers=song.aux32_centers.masked_fill(~song.sample_valid.any(-1)[..., None], -1e6))
    changed_window = replace(window, mel=window.mel.masked_fill(~window.valid[..., None], -1e6),
                             aux32=window.aux32.masked_fill(~window.valid[..., None], 1e6))
    changed = model(changed_song, changed_window)
    for expected, actual in zip(scales(reference), scales(changed)):
        torch.testing.assert_close(expected.values, actual.values)
        assert torch.equal(expected.valid, actual.valid)
    # Nonzero learned biases must still not create output for padded songs/windows.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.fill_(0.7)
    empty_song = replace(song, sample_valid=torch.zeros_like(song.sample_valid))
    empty_window = replace(window, valid=torch.zeros_like(window.valid))
    empty = model(empty_song, empty_window)
    for memory in scales(empty):
        assert not memory.valid.any()
        assert torch.isfinite(memory.values).all()
        assert torch.count_nonzero(memory.values) == 0


def test_songs_route_independently_and_windows_use_absolute_coordinates(encoder_config, audio_inputs):
    model = build_model(encoder_config).eval()
    song, windows = audio_inputs
    output = model(song, windows)
    other_mel = song.mel_patches.clone()
    other_mel[1] += 10
    changed = model(replace(song, mel_patches=other_mel), windows)
    for actual, expected in zip(scales(changed), scales(output)):
        torch.testing.assert_close(actual.values[:2], expected.values[:2])
    for memory, rate in zip(scales(output), encoder_config["fpn_rates"]):
        expected = windows.target_start_beats[:, None] - 2 + (torch.arange(6 * rate).float() + 0.5) / rate
        torch.testing.assert_close(memory.beats, expected)


@pytest.mark.parametrize("depths", [(1, 1, 1), (2, 1, 2)])
def test_final_block_is_derived_from_actual_depth(encoder_config, audio_inputs, depths):
    model = build_model(dict(encoder_config, stage_depths=depths)).eval()
    blocks = [block for stage in model.stages for block in stage]
    calls = [0] * len(blocks)
    handles = []
    for index, block in enumerate(blocks):
        def count_q(_module, _args, _output, index=index):
            calls[index] += 1
        handles.append(block.q.register_forward_hook(count_q))
    memory = model.prepare_song(audio_inputs[0])
    assert len(memory.block_kv) == sum(depths)
    assert all(pair is not None for pair in memory.block_kv[:-1])
    assert memory.block_kv[-1] is None
    model.encode_windows(memory, audio_inputs[1])
    assert calls == [2] * (len(blocks) - 1) + [1]
    for handle in handles:
        handle.remove()


def test_prepared_tempo_change_features_are_consumed_without_pooling_or_reinjection(encoder_config, audio_inputs):
    model = build_model(encoder_config).eval()
    song, window = audio_inputs
    # aux field 24 is local_tempo_bpm in the documented 18+4+10 layout.
    # The producer supplies a tempo jump at absolute beat zero and phases at
    # actual sample positions. The network consumes these rows unchanged.
    aux = torch.zeros_like(window.aux32)
    aux[..., 24] = torch.where(window.beats < 0, 0.5, 1.0)
    aux[..., 25] = (2 * torch.pi * window.beats).sin()
    aux[..., 26] = (2 * torch.pi * window.beats).cos()
    aux = aux.masked_fill(~window.valid[..., None], 0)
    prepared_window = replace(window, aux32=aux)
    captured = []
    handle = model.feature_fusion.register_forward_pre_hook(lambda _module, args: captured.append(args[1]))
    output = model(song, prepared_window)
    handle.remove()
    assert len(captured) == 1 and captured[0] is aux
    assert not any("bpm" in name or "tempo" in name for name, _ in model.named_parameters())
    for scale in scales(output):
        assert torch.isfinite(scale.values).all()


@pytest.mark.parametrize("change", [
    {"d_model": 25}, {"d_model": 21, "heads": 3}, {"temporal_kernel_size": 4},
    {"stage_depths": (1, 1)}, {"target_beats": 0}, {"audio_after_beats": -1},
    {"fpn_rates": (0.3, 1, 1)}, {"dropout": -0.1}, {"rope_base": 0},
])
def test_only_real_mathematical_constraints_rejected(encoder_config, change):
    with pytest.raises(ValueError):
        build_model(dict(encoder_config, **change))
