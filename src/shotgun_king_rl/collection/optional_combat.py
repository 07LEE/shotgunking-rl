"""Use conservative optional-stat readers for collection proposals."""

import json
from pathlib import Path

import cv2

from ..perception.knockback import KnockbackReader
from ..perception.penetration import PenetrationReader
from ..perception.melee import MeleeDamageReader, melee_damage_patch
from .images import normalize


def melee_title_reference(collection, exclude=None):
    """Load an explicitly selected, reviewed training title, never infer one."""
    root = Path(collection).resolve()
    manifest = root / 'title_references.json'
    if not manifest.exists():
        return None
    relative = json.loads(manifest.read_text()).get('melee_damage')
    if relative is None:
        return None
    path = (root / relative).resolve()
    if root not in path.parents or path.suffix != '.json':
        raise ValueError('Invalid melee title reference path')
    if exclude is not None and path == Path(exclude).resolve():
        return None
    data = json.loads(path.read_text())
    state = data.get('screen_state') or {}
    reviews = state.get('reviews') or {}
    value = (state.get('combat_stats') or {}).get('melee_damage')
    if (data.get('split') != 'train' or state.get('screen_kind') != 'gameplay'
            or not reviews.get('screen_kind') or not reviews.get('combat_stats')
            or type(value) is not int or value <= 0):
        raise ValueError('Melee title requires a reviewed positive train reference')
    image_path = (path.parent / data['image']).resolve()
    if image_path.parent != (path.parent.parent / 'originals').resolve():
        raise ValueError('Invalid melee title image path')
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Unreadable melee title image')
    row = melee_damage_patch(normalize(image, data['game_crop']))
    if row is None or row[:30].std() < 5:
        raise ValueError('Melee title is not visible')
    return row[:30]


def propose_optional_combat(screen, collection, exclude=None):
    """Keep absent rows unknown unless a reviewed zero-panel match proves zero."""
    readers = {
        'penetration_percent': PenetrationReader.from_collection(collection, exclude=exclude),
        'knockback_percent': KnockbackReader.from_collection(collection, exclude=exclude),
        'melee_damage': MeleeDamageReader.from_collection(collection, exclude=exclude, title_reference=melee_title_reference(collection, exclude)),
    }
    values, evidence = {}, {}
    for field, reader in readers.items():
        result = reader.read(screen)
        values[field] = result.value if result.is_accepted else None
        evidence[field] = {'source': result.source, 'reason': result.reason, 'confidence': result.confidence}
    return values, evidence
