"""File hashing with an optional, single-verification-session cache."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import time


_session = ContextVar("artifact_hash_session", default=None)


def file_sha256(path):
    path = Path(path).resolve()
    session = _session.get()
    stat = path.stat()
    def identity(value):
        # Some network filesystems expose no inode; do not alias unrelated files.
        return (value.st_dev, value.st_ino or str(path), value.st_size, value.st_mtime_ns)
    key = identity(stat)
    if session is not None and key in session["cache"]:
        return session["cache"][key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
            if session is not None:
                session["bytes"] += len(block)
                now = time.perf_counter()
                if now - session["last_progress"] >= 1:
                    session["progress"](path, session["bytes"])
                    session["last_progress"] = now
    after = path.stat()
    if identity(after) != key:
        raise ValueError(f"file changed during verification: {path}")
    value = digest.hexdigest()
    if session is not None:
        session["cache"][key] = value
    return value


@contextmanager
def hashing_session(progress):
    """Deduplicate reads within this call only; never trust an old receipt."""
    state = {"cache": {}, "bytes": 0, "progress": progress, "last_progress": 0.0}
    token = _session.set(state)
    try:
        yield state
    finally:
        _session.reset(token)
