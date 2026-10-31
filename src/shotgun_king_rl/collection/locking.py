"""Serialize inbox importers across processes and threads."""

from contextlib import contextmanager
import fcntl
from functools import wraps
from pathlib import Path
import threading

_held = threading.local()


@contextmanager
def collection_lock(collection):
    root = Path(collection).resolve()
    root.mkdir(parents=True, exist_ok=True)
    locks = getattr(_held, 'locks', {})
    if root in locks:
        yield
        return
    with (root / '.pipeline.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another collection importer or pipeline is running') from exc
        _held.locks = locks
        locks[root] = handle
        try:
            yield
        finally:
            del locks[root]
            fcntl.flock(handle, fcntl.LOCK_UN)


def serialized(function):
    @wraps(function)
    def wrapped(collection, *args, **kwargs):
        with collection_lock(collection):
            return function(collection, *args, **kwargs)
    return wrapped
