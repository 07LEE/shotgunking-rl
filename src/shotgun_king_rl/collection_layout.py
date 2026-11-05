"""Locate collection files kept in flat originals/ and annotations/ folders.

Each stored file name starts with its session ID and SEPARATOR, so the session
stays the split unit without one directory per play session.
"""

from pathlib import Path
import re

ORIGINALS = 'originals'
ANNOTATIONS = 'annotations'
SEPARATOR = '__'
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg')
# Single '-' or '_' between alphanumerics keeps SEPARATOR unambiguous.
SESSION_ID = re.compile(r'[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*')


def validate_session_id(session_id):
    if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
        raise ValueError("Session ID must be letters and digits joined by single '-' or '_'")
    return session_id


def originals_dir(collection):
    return Path(collection) / ORIGINALS


def annotations_dir(collection):
    return Path(collection) / ANNOTATIONS


def stored_name(session_id, name):
    return validate_session_id(session_id) + SEPARATOR + name


def session_of(path):
    """Return the session prefix of a stored file name, or None."""
    session, separator, _ = Path(path).name.partition(SEPARATOR)
    return session if separator and SESSION_ID.fullmatch(session) else None


def _in_sessions(path, sessions):
    return sessions is None or session_of(path) in sessions


def annotation_paths(collection, sessions=None):
    sessions = None if sessions is None else set(sessions)
    return sorted(p for p in annotations_dir(collection).glob('*.json') if _in_sessions(p, sessions))


def session_images(collection, session_id):
    return sorted(
        p for p in originals_dir(collection).glob(stored_name(session_id, '*'))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and session_of(p) == session_id
    )
