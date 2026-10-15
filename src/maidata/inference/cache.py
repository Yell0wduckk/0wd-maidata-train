"""Two immutable inference caches; identities use metadata, never payload hashing."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import tempfile
import time

import torch

from maidata.draft.types import BeatMemory
from maidata.draft.windows import SongCoverage, song_coverage
from maidata.model import ENCODER_KEYS
from .conditioning import Conditions, encode_joint_audio, decode_joint_draft, empty_plan_geometry
from .progress import stage

AUDIO_SCHEMA = 'maidata.inference-audio.v1'
DRAFT_SCHEMA = 'maidata.inference-draft-plan.v1'


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def identity_key(identity):
    # Only the small identity document is hashed, never audio/weights/features.
    return hashlib.sha256(json.dumps(canonical(identity), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def execution(device, precision):
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return dict(device=str(device), precision=precision)


def encoder_recipe(config):
    return dict({key: config[key] for key in ENCODER_KEYS},
                fpn_p0_normalization=config.get('fpn_p0_normalization', False))


def feature_identity(*, grid=None, mel_root=None, mel_stats=None):
    if grid:
        result = dict(grid=file_identity(grid), mel_root=str(Path(mel_root).resolve()) if mel_root else None)
        metadata = Path(grid).with_suffix('.json')
        if metadata.is_file():
            result['metadata'] = file_identity(metadata)
            reference = json.loads(metadata.read_text('utf-8')).get('mel_reference')
            if reference:
                feature = (Path(mel_root)/PureWindowsPath(reference['feature_path']).name
                           if mel_root else Path(reference['feature_path']))
                result['referenced_mel'] = file_identity(feature)
        return result
    return dict(mel_stats=file_identity(mel_stats))


def audio_identity(checkpoint, audio, timing, *, grid=None, mel_root=None, mel_stats=None,
                   device, precision, encoder_window_batch):
    if encoder_window_batch < 1:
        raise ValueError('encoder-window-batch must be positive')
    return canonical(dict(schema=AUDIO_SCHEMA, checkpoint=checkpoint.identity,
        encoder=encoder_recipe(checkpoint.config), audio=file_identity(audio),
        timing=dict(first_seconds=timing.first_seconds,
                    bpm=[[b.numerator, b.denominator, value] for b, value in timing.points]),
        features=feature_identity(grid=grid, mel_root=mel_root, mel_stats=mel_stats),
        execution=execution(device, precision), encoder_window_batch=encoder_window_batch,
        audio_scope='window', first_beat=0, include_tail=True))


def draft_identity(checkpoint, audio_key, *, difficulty, level, device, precision):
    if difficulty not in range(1, 8) or not math.isfinite(level):
        raise ValueError('difficulty must be 1..7 and level finite')
    config = checkpoint.config
    keys = [key for key in config if key.startswith(('draft_', 'planner_'))
            and not key.startswith(('draft_train_', 'planner_train_'))]
    return canonical(dict(schema=DRAFT_SCHEMA, checkpoint=checkpoint.identity, audio_key=audio_key,
        config={key: config[key] for key in keys},
        rope={key: config[key] for key in ('rope_base', 'rope_position_scale')},
        target_beats=config['target_beats'], difficulty=difficulty, level=level,
        execution=execution(device, precision)))


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        for attempt in range(101):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 100:
                    raise
                time.sleep(.01)
    finally:
        Path(temporary).unlink(missing_ok=True)


def memory_payload(memory):
    return {key: getattr(memory, key).detach().cpu() for key in ('values', 'beats', 'valid')}


def read_memory(row, *, width, rows, length=None, empty=False):
    values, beats, valid = (row[key] for key in ('values', 'beats', 'valid'))
    if (values.ndim != 3 or values.shape[0] != rows or values.shape[-1] != width
            or valid.shape != values.shape[:2] or beats.shape != valid.shape
            or valid.dtype != torch.bool or not values.is_floating_point()
            or not beats.is_floating_point()
            or (length is not None and values.shape[1] != length)
            or (not empty and values.shape[1] < 1)
            or not torch.isfinite(values).all() or not torch.isfinite(beats).all()):
        raise ValueError('invalid cached memory shape, dtype or values')
    return BeatMemory(values, beats, valid)


def read_payload(path, schema, identity=None):
    path = Path(path)
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload['schema'] != schema or payload['complete'] is not True
                or payload['identity']['schema'] != schema
                or payload['key'] != identity_key(payload['identity'])
                or payload['key'] != path.stem
                or (identity is not None and payload['identity'] != canonical(identity))):
            raise ValueError('cache identity/schema differs')
        return payload
    except Exception as exc:
        raise ValueError(f'invalid inference cache {path}: {exc}') from exc


@dataclass
class EncodedAudio:
    local: Conditions
    centers: BeatMemory
    duration: float
    key: str
    identity: dict


def load_audio(path, config, *, device, identity=None):
    payload = read_payload(path, AUDIO_SCHEMA, identity)
    try:
        coverage = SongCoverage(**payload['coverage'])
        starts = tuple(payload['starts'])
        duration = payload['duration_seconds']
        expected_encoder = canonical(encoder_recipe(config))
        from maidata.time import TimingMap
        expected_coverage = song_coverage(TimingMap.from_audio_timing(payload['identity']['timing']),
            duration, first_beat=0, target_beats=config['target_beats'])
        if (coverage.first_beat != 0 or coverage.target_beats != config['target_beats']
                or coverage != expected_coverage
                or starts != coverage.starts(include_tail=True) or len(starts) < 2
                or not math.isfinite(duration) or duration <= 0
                or payload['identity']['encoder'] != expected_encoder):
            raise ValueError('cached audio coverage/config differs')
        context = config['audio_before_beats']+config['target_beats']+config['audio_after_beats']
        local = read_memory(payload['local'], width=config['d_model'], rows=len(starts),
                            length=context*config['fpn_rates'][0])
        centers = read_memory(payload['centers'], width=config['d_model'], rows=1,
                              length=coverage.audio_windows*config['target_beats']*config['fpn_rates'][0])
        local, centers = (BeatMemory(*(getattr(bank, k).to(device) for k in ('values', 'beats', 'valid')))
                          for bank in (local, centers))
        z, plans, plan_valid = empty_plan_geometry(starts, coverage, config, device)
        conditions = Conditions(local, z, plans, plan_valid, starts, coverage, 'window')
        return EncodedAudio(conditions, centers, duration, payload['key'], payload['identity'])
    except Exception as exc:
        raise ValueError(f'invalid inference cache {path}: {exc}') from exc


class JointCache:
    def __init__(self, root, checkpoint, *, device, precision, progress=None):
        self.root, self.checkpoint = Path(root), checkpoint
        self.device, self.precision, self.progress = device, precision, progress
        self.report = dict(audio=None, draft=None)

    def find_audio(self, identity):
        path = self.root/'audio'/(identity_key(identity)+'.pt')
        self.report['audio'] = dict(path=str(path.resolve()), hit=path.is_file())
        if not path.exists():
            return None
        with stage(self.progress, 'audio_cache_read'):
            return load_audio(path, self.checkpoint.config, device=self.device, identity=identity)

    def encode(self, identity, grid, timing, duration):
        models = self.checkpoint.models(device=self.device, components=('encoder',))
        local, centers = encode_joint_audio(models.encoder, models.config, grid, timing, duration,
            device=self.device, encoder_window_batch=identity['encoder_window_batch'], progress=self.progress)
        path = self.root/'audio'/(identity_key(identity)+'.pt')
        result = EncodedAudio(local, centers, duration, identity_key(identity), identity)
        payload = dict(schema=AUDIO_SCHEMA, complete=True, identity=identity, key=result.key,
            duration_seconds=duration, coverage=asdict(local.coverage), starts=list(local.starts),
            local=memory_payload(local.audio), centers=memory_payload(centers))
        with stage(self.progress, 'audio_cache_write'):
            atomic_save(path, payload)
        self.report['audio'] = dict(path=str(path.resolve()), hit=False)
        return result

    def draft(self, encoded, *, difficulty, level):
        identity = draft_identity(self.checkpoint, encoded.key, difficulty=difficulty, level=level,
                                  device=self.device, precision=self.precision)
        path = self.root/'draft'/(identity_key(identity)+'.pt')
        self.report['draft'] = dict(path=str(path.resolve()), hit=path.is_file())
        if path.exists():
            with stage(self.progress, 'draft_cache_read'):
                payload = read_payload(path, DRAFT_SCHEMA, identity)
                try:
                    c = self.checkpoint.config
                    z = read_memory(payload['z'], width=c['draft_latent_dim'], rows=1,
                                    length=encoded.local.coverage.draft_beats//c['draft_z_stride_beats'])
                    plans, valid = payload['plans'], payload['plan_valid']
                    plan_shape = ((len(encoded.local.starts),
                        (c['target_beats']+c['decoder_z_before_beats']+c['decoder_z_after_beats'])//4, 12)
                        if c.get('planner_span_beats', 16) == 4 else (len(encoded.local.starts), 12))
                    if (payload['starts'] != list(encoded.local.starts)
                            or plans.shape != plan_shape or not plans.is_floating_point()
                            or valid.shape != plans.shape[:-1] or valid.dtype != torch.bool
                            or not torch.isfinite(plans).all()):
                        raise ValueError('invalid cached plan shape, dtype or values')
                except Exception as exc:
                    raise ValueError(f'invalid inference cache {path}: {exc}') from exc
                z = BeatMemory(*(getattr(z, k).to(self.device) for k in ('values', 'beats', 'valid')))
                return replace(encoded.local, z=z, plans=plans.to(self.device), plan_valid=valid.to(self.device))
        models = self.checkpoint.models(device=self.device, components=('draft', 'planner'))
        conditions = decode_joint_draft(models.draft, models.planner, models.config,
            encoded.local, encoded.centers, difficulty=difficulty, level=level,
            device=self.device, progress=self.progress)
        payload = dict(schema=DRAFT_SCHEMA, complete=True, identity=identity, key=identity_key(identity),
            starts=list(conditions.starts), z=memory_payload(conditions.z),
            plans=conditions.plans.detach().cpu(), plan_valid=conditions.plan_valid.cpu())
        with stage(self.progress, 'draft_cache_write'):
            atomic_save(path, payload)
        return conditions
