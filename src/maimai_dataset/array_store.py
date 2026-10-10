"""Keep readonly NPY mappings file-backed when sending stores to spawn workers."""
from pathlib import Path
import numpy as np


def _open_readonly(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class _ArrayReference:
    def __init__(self, array):
        self.path = str(Path(array.filename).resolve())

    def __reduce__(self):
        return _open_readonly, (self.path,)


def mmap_pickle_state(value):
    # Stores hold whole NPY arrays opened by np.load, never sliced memmaps.
    if isinstance(value, np.memmap):
        return _ArrayReference(value)
    if isinstance(value, dict):
        return {key: mmap_pickle_state(child) for key, child in value.items()}
    return value


def _restore_state(value):
    if isinstance(value, _ArrayReference):
        return _open_readonly(value.path)
    if isinstance(value, dict):
        return {key: _restore_state(child) for key, child in value.items()}
    return value


class ReadOnlyArrayStore:
    def __getstate__(self):
        return mmap_pickle_state(self.__dict__)

    def __setstate__(self, state):
        # copy.copy uses __getstate__ without unpickling the nested references.
        self.__dict__.update(_restore_state(state))
