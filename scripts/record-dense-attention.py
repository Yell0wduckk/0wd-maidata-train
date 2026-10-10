"""Observe real dense generation without replacing the native SDPA result.

The side computation uses the actual post-RoPE Q/K and the actual SDPA mask.
Raw probabilities retain every head and key (float16 storage, float32 softmax).
Native cache snapshots also snapshot an external node-id journal, so rejected
queries can be distinguished from the final causal history.
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F


def json_file(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def probabilities(q, k, mask=None, causal=False, scale=None):
    with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
        logits = (q.float() @ k.float().transpose(-1, -2)) * (scale or q.shape[-1] ** -.5)
        if causal:
            allowed = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
            logits.masked_fill_(~allowed, -torch.inf)
        if mask is not None:
            if mask.dtype == torch.bool:
                logits.masked_fill_(~mask, -torch.inf)
            else:
                logits += mask.float()
        return logits.softmax(-1).nan_to_num(0.)


class Recorder:
    def __init__(self, root):
        self.root = root
        self.archive = zipfile.ZipFile(root / 'raw-attention.zip', 'w', compression=zipfile.ZIP_STORED, allowZip64=True)
        self.index = []
        self.calls = []
        self.node_ids = []
        self.nodes = []
        self.snapshots = {}
        self.accepted = {}
        self.events = []
        self.active = None
        self.cache = None
        self.names = {}
        self.checks = {}
        self.sequence = 0
        self.native_sdpa = F.scaled_dot_product_attention
        self.patches = []

    def patch(self, obj, name, replacement):
        self.patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def write(self, kind, arrays, **metadata):
        name = f'{kind}/{self.sequence:06d}.npz'
        self.sequence += 1
        data = io.BytesIO()
        np.savez_compressed(data, **arrays)
        self.archive.writestr(name, data.getvalue())
        self.index.append(dict(path=name, kind=kind, **metadata))

    def install(self):
        from maimai_baseline.inference import api
        from maimai_baseline.inference.song_dense import SongDenseKVCache
        load = api.load_baseline_inference_checkpoint

        def loaded(*args, **kwargs):
            result = load(*args, **kwargs)
            model = result[0]
            self.names = {id(module): name for name, module in model.named_modules()}
            assert model.song_dense and len(model.decoder_layers) == 8
            return result
        self.patch(api, 'load_baseline_inference_checkpoint', loaded)
        decode = SongDenseKVCache.decode
        snapshot = SongDenseKVCache.snapshot
        restore = SongDenseKVCache.restore
        trim = SongDenseKVCache.trim_local

        def recorded_decode(cache, **kwargs):
            assert self.active is None
            self.cache = cache
            previous = list(self.node_ids)
            assert len(previous) == cache.song_length
            call = dict(id=len(self.calls), arrays={})
            self.active = call
            try:
                result = decode(cache, **kwargs)
                count = cache.song_length - len(previous)
                qb = cache.positions[:, -count:].detach().cpu().numpy()
                first = kwargs['input_ids']['event'].shape[1] - count
                events = kwargs['input_ids']['event'][:, first:].detach().cpu().numpy()
                nodes = []
                for row in range(count):
                    node_id = len(self.nodes)
                    self.nodes.append(dict(id=node_id, call_id=call['id'], query_row=row,
                        beat=float(qb[0, row]), event_id=int(events[0, row]),
                        event=cache.model.vocabulary.fields['event'].values[int(events[0, row])],
                        window_start_tick=int(cache.start[0])))
                    nodes.append(node_id)
                self.node_ids = previous + nodes
                arrays = call['arrays']
                assert len([k for k in arrays if k.startswith('self_')]) == 8
                assert len([k for k in arrays if k.startswith('cross_')]) == 8
                arrays.update(query_beats=qb, self_key_beats=cache.positions.detach().cpu().numpy(),
                    cross_key_beats=cache.memory_beats.detach().cpu().numpy(),
                    query_node_ids=np.array(nodes, np.int64), key_node_ids=np.array(self.node_ids, np.int64),
                    query_valid=(~cache.padding[:, -count:]).cpu().numpy(),
                    self_key_valid=(~cache.padding).cpu().numpy(),
                    cross_key_valid=(~cache.memory_padding).cpu().numpy(),
                    family=cache.families[:, -count:].cpu().numpy())
                self.write('decoder', arrays, call_id=call['id'], query_count=count,
                           history_keys=len(self.node_ids))
                self.calls.append(dict(id=call['id'], node_ids=nodes, window_start_tick=int(cache.start[0])))
                return result
            finally:
                self.active = None

        def recorded_snapshot(cache):
            state = snapshot(cache)
            self.snapshots[id(state)] = list(self.node_ids)
            return state

        def recorded_restore(cache, state, *, replay_last=False):
            restore(cache, state, replay_last=replay_last)
            self.node_ids = self.snapshots[id(state)][:cache.song_length]
            assert len(self.node_ids) == cache.song_length
            self.events.append(dict(event='cache_restore', replay_last=replay_last, song_length=cache.song_length))

        def recorded_trim(cache, length):
            trim(cache, length)
            self.node_ids = self.node_ids[:cache.song_length]
            self.events.append(dict(event='cache_trim', song_length=cache.song_length))

        self.patch(SongDenseKVCache, 'decode', recorded_decode)
        self.patch(SongDenseKVCache, 'snapshot', recorded_snapshot)
        self.patch(SongDenseKVCache, 'restore', recorded_restore)
        self.patch(SongDenseKVCache, 'trim_local', recorded_trim)
        self.patch(F, 'scaled_dot_product_attention', self.sdpa)

    def stream(self, event, **data):
        if event == 'row':
            row = dict(data, query_node_id=self.node_ids[-1])
            self.accepted.setdefault(data['index'], []).append(row)
        elif event == 'rollback':
            self.accepted[data['index']] = self.accepted.get(data['index'], [])[:data['row_count']]
        elif event == 'window_end':
            print(json.dumps(dict(event=event, **data, captures=self.sequence, nodes=len(self.node_ids))), flush=True)
        if event != 'row':
            self.events.append(dict(event=event, **data))

    def sdpa(self, q, k, v, attn_mask=None, dropout_p=0., is_causal=False, scale=None, enable_gqa=False):
        # Return precisely the original backend result; tracing consumes no RNG.
        output = self.native_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                  is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        frame = sys._getframe(1)
        frames = []
        try:
            for _ in range(15):
                if frame is None:
                    break
                frames.append((frame.f_code.co_name, frame.f_globals.get('__name__', ''), frame.f_locals))
                frame = frame.f_back
            find = lambda name, module: next((loc for fn, mod, loc in frames if fn == name and mod.endswith(module)), None)
            dual = find('attention', 'model.dual')
            dense = find('dense_layers', 'model.song_dense')
            local = find('local_attention', 'model.local_attention')
            global_audio = find('forward', 'model.song_audio')
            if dual is not None and dense is not None and self.active is not None:
                kind = 'self' if dual['causal'] else 'cross'
                label = f"{kind}_{dense['d']:02d}"
                assert dropout_p == 0 and not enable_gqa
                p = probabilities(q, k, attn_mask, is_causal, scale)
                self.check(label, p, v, output)
                self.active['arrays'][label] = p.cpu().numpy().astype(np.float16)
            elif local is not None:
                assert dropout_p == 0 and not enable_gqa
                beats, qi, ki = local['beats'], local['qi'], local['ki']
                b, blocks, width = local['batch'], local['blocks'], local['width']
                qb = beats[:, qi].reshape(b*blocks, width)
                kb = beats[:, ki].reshape(b*blocks, local['key_width'])
                valid = local['valid'].reshape(b*blocks, width)
                enc = find('encode_stages', 'model.audio')
                starts = enc['start_tick'].float().repeat_interleave(blocks) / 96.
                canonical = valid & (qb >= starts[:, None]) & (qb < starts[:, None]+16)
                label = self.names[id(local['attention'])]
                for start in range(0, len(q), 32):
                    stop = min(start+32, len(q))
                    p = probabilities(q[start:stop], k[start:stop], attn_mask[start:stop], is_causal, scale)
                    self.check(label, p, v[start:stop], output[start:stop])
                    self.write('audio_local', dict(probability=p.cpu().numpy().astype(np.float16),
                        query_beats=qb[start:stop].cpu().numpy(), key_beats=kb[start:stop].cpu().numpy(),
                        query_valid=valid[start:stop].cpu().numpy(), canonical_query=canonical[start:stop].cpu().numpy(),
                        allowed=attn_mask[start:stop].cpu().numpy()), layer=label)
            elif global_audio is not None and type(global_audio.get('self')).__name__ == 'SongAudioBlock':
                assert dropout_p == 0 and not enable_gqa
                att = find('attend', 'model.rope')
                p = probabilities(q, k, attn_mask, is_causal, scale)
                label = self.names[id(global_audio['self'])]
                qb, kb = att['positions'], global_audio['beats']
                # Whole-song chunks are exact consecutive slices, not resampled positions.
                full_valid = global_audio['valid']
                indices = (qb[:, :, None] == kb[:, None, :]).float().argmax(-1)
                valid = full_valid.gather(1, indices)
                self.check(label, p, v, output)
                self.write('audio_global', dict(probability=p.cpu().numpy().astype(np.float16),
                    query_beats=qb.cpu().numpy(), key_beats=kb.cpu().numpy(), query_valid=valid.cpu().numpy(),
                    key_valid=full_valid.cpu().numpy(), allowed=attn_mask.cpu().numpy()), layer=label)
        finally:
            del frames, frame
        return output

    def check(self, label, p, v, output):
        if label in self.checks:
            return
        reconstructed = p @ v.float()
        self.checks[label] = dict(shape=list(p.shape), finite=bool(torch.isfinite(p).all()),
            probability_sum_max_error=float((p.sum(-1)-1).abs().max()),
            output_reconstruction_max_abs=float((reconstructed-output.float()).abs().max()),
            fp16_probability_max_abs=float((p-p.half().float()).abs().max()))

    def close(self):
        for obj, name, original in reversed(self.patches):
            setattr(obj, name, original)
        self.archive.close()
        json_file(self.root/'index.json', self.index)
        json_file(self.root/'journal.json', dict(nodes=self.nodes, calls=self.calls,
            final_history_node_ids=self.node_ids,
            accepted_predictions=[row for window in sorted(self.accepted) for row in self.accepted[window]], events=self.events))
        json_file(self.root/'checks.json', self.checks)


def self_test():
    torch.manual_seed(31)
    q, k, v = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    mask = torch.ones(2, 1, 4, 7, dtype=torch.bool)
    mask[..., 5:] = False
    p = probabilities(q, k, mask)
    torch.testing.assert_close(p @ v, F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
    assert not p[..., 5:].any()
    p = probabilities(q, k, causal=True)
    torch.testing.assert_close(p @ v, F.scaled_dot_product_attention(q, k, v, is_causal=True))
    assert not p.triu(1).any()
    assert not probabilities(q, k, torch.zeros_like(mask)).any()
    print('probability reconstruction, causal mask, padding and all-masked tests passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.source or not args.output:
        parser.error('--source and --output are required')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source.resolve()/'src'))
    from maimai_baseline.inference.api import generate_baseline_inference
    base = Path('/root/autodl-tmp')
    data = base/'maimai-full-dataset-lr-fixed-20260911'
    input_root = base/'moment-simai-c50-five-song-20261003/inputs/11739'
    checkpoint = base/'maidata-decoder8-song-dense-20261003/training/branch-090/B/ar.ckpt'
    recorder = Recorder(root)
    recorder.install()
    json_file(root/'invocation.json', dict(source=str(args.source), argv=sys.argv,
        checkpoint=str(checkpoint), song_id='11739', started_unix=time.time(), torch=torch.__version__,
        cuda=torch.version.cuda, probability_definition='softmax(post-RoPE QK^T / sqrt(d) + actual mask)',
        raw_storage='float16 probability per head; coordinates float32; no top-k or key subsampling',
        native_output_unchanged=True))
    try:
        result = generate_baseline_inference(checkpoint=checkpoint,
            route_artifact=data/'slide-geometry/route-artifact', profile_root=data/'profile', song_id='11739',
            timing_path=input_root/'timing.json', output=root/'inference', level=14., cabinet='DX',
            difficulty_id=5, force=False, device='cuda', seed=20260811, temperature=0.,
            audio_path=input_root/'audio.mp3', physical_hard_mask=True,
            physical_path_catalog=data/'physical/path-catalog', miacode_slide_check=True, trace=True,
            local_seed=20260812, style_mode='ar', latent_temperature=1., mixture_temperature=1., stream=recorder.stream)
        previous = base/'decoder8-song-dense-b50-five-song-20261004/output/11739/model.rows.json'
        rows = json.loads((root/'inference/model.rows.json').read_text())
        same = rows == json.loads(previous.read_text())
        json_file(root/'completion.json', dict(exit_code=0, ended_unix=time.time(),
            generated_rows=result['generated_rows'], truncated=result['truncated'],
            same_rows_as_uninstrumented_b50=same, reference=str(previous),
            checkpoint_epoch=result['checkpoint_epoch'], checkpoint_step=result['checkpoint_step'],
            checkpoint_sha256=result['checkpoint_sha256']))
        print('COMPLETE',same,result['generated_rows'],flush=True)
    except Exception as exc:
        json_file(root/'failure.json', dict(error=repr(exc), ended_unix=time.time()))
        raise
    finally:
        recorder.close()


if __name__ == '__main__':
    main()
