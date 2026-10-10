"""Shared song-contiguous ordering of complete audio groups."""
from collections import Counter, defaultdict
from itertools import groupby
from maimai_baseline.data.repack import flatten, reference_hash
from maimai_dataset.manifest import canonical_json_sha256, DatasetError

POLICY = dict(id="song-contiguous-v1", seed=20260811, token_budget=4096)
SONG_FIRST_POLICY = dict(id="song-first-a-v1", token_budget=4096,
    ordering_salt=POLICY['id'], oversize="complete-audio-groups")


def song_first(batches, *, lengths, audio_keys, song_ids):
    """Repack ordered references, keeping fitting songs and audio groups intact."""
    refs = flatten(batches)
    result, batch, used, seen = [], [], 0, set()
    budget = SONG_FIRST_POLICY['token_budget']
    def close():
        nonlocal batch, used
        if batch:
            result.append(tuple(batch))
        batch, used = [], 0
    for sid, members in groupby(refs, key=lambda r: str(song_ids[r.canonical_index])):
        if sid in seen:
            raise DatasetError('source plan is not song contiguous')
        seen.add(sid)
        members = tuple(members)
        cost = sum(max(1, int(lengths[r.canonical_index])) for r in members)
        if cost <= budget:
            if batch and used + cost > budget:
                close()
            batch.extend(members); used += cost
            continue
        close()
        for _, group in groupby(members, key=lambda r: audio_keys[r.canonical_index]):
            group = tuple(group)
            cost = sum(max(1, int(lengths[r.canonical_index])) for r in group)
            if batch and used + cost > budget:
                close()
            batch.extend(group); used += cost
            if cost > budget:
                close()
    close()
    assert flatten(result) == refs
    return tuple(result)


def arrange(refs, *, epoch, lengths, audio_keys, song_ids):
    """Move original groups intact; never merge separated occurrences by key."""
    songs = defaultdict(list)
    for _, members in groupby(refs, key=lambda r: audio_keys[r.canonical_index]):
        group = tuple(members)
        names = {str(song_ids[r.canonical_index]) for r in group}
        if len(names) != 1:
            raise DatasetError("shared audio group spans different song identities")
        songs[names.pop()].append(group)
    order = sorted(songs, key=lambda sid: (
        canonical_json_sha256([POLICY["id"], POLICY["seed"], epoch, sid]), sid))
    batches, batch, used = [], [], 0
    for sid in order:
        for group in songs[sid]:
            cost = sum(max(1, lengths[r.canonical_index]) for r in group)
            if batch and used + cost > POLICY["token_budget"]:
                batches.append(tuple(batch)); batch, used = [], 0
            batch.extend(group); used += cost
    if batch:
        batches.append(tuple(batch))
    if Counter(flatten(batches)) != Counter(refs):
        raise AssertionError("song sampling changed reference multiplicity")
    return tuple(batches)


def statistics(batches, *, lengths, song_ids, record_ids, audio_keys):
    rows, songs, globals_, windows = [], [], [], []
    per_song = Counter()
    for batch in batches:
        names = {str(song_ids[r.canonical_index]) for r in batch}
        per_song.update(names)
        rows.append(sum(max(1, lengths[r.canonical_index]) for r in batch))
        songs.append(len(names))
        globals_.append(len({(record_ids[r.canonical_index], r.view_id) for r in batch}))
        windows.append(len({audio_keys[r.canonical_index] for r in batch}))
    return dict(batch_count=len(batches), batch_target_rows=rows, target_rows=sum(rows),
                batch_song_counts=songs, song_update_counts=dict(sorted(per_song.items())),
                batch_global_reference_groups=globals_, global_reference_group_evaluations=sum(globals_),
                batch_audio_windows=windows, audio_window_evaluations=sum(windows),
                unique_audio_windows=len({audio_keys[r.canonical_index] for r in flatten(batches)}),
                oversized_batches=[i for i, n in enumerate(rows) if n > POLICY["token_budget"]],
                reference_sha256=reference_hash(flatten(batches)))
