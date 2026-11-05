"""Load explicitly selected melee titles with the same checks for every caller."""

import hashlib
import json
from pathlib import Path

import cv2

from .visibility import visual_combat_stats

from ..vision.layout import normalize_game_screen
from .melee import melee_damage_patch


def load_melee_title_reference(collection, title_annotation=None, *, exclude=None):
    """Use an explicit override or the collection's selected train annotation.

    No manifest means no positive reference. Invalid selections raise instead
    of silently disabling validation or selecting another training example.
    """
    root = Path(collection).resolve()
    selection = 'explicit'
    if title_annotation is None:
        manifest = root / 'title_references.json'
        if not manifest.exists():
            return None, None
        data = json.loads(manifest.read_text())
        if not isinstance(data, dict):
            raise ValueError('Invalid title reference manifest')
        relative = data.get('melee_damage')
        if relative is None:
            return None, None
        if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
            raise ValueError('Invalid melee title reference path')
        path = (root / relative).resolve()
        selection = 'collection_manifest'
    else:
        path = Path(title_annotation).resolve()
    if not path.is_relative_to(root) or path.suffix != '.json':
        raise ValueError('Invalid melee title reference path')
    if exclude is not None and path == Path(exclude).resolve():
        return None, None
    if path.parent != root / 'annotations':
        raise ValueError('Title reference must be a collection annotation')
    raw = path.read_bytes()
    data = json.loads(raw)
    state = data.get('screen_state') or {}
    reviews = state.get('reviews') or {}
    value = visual_combat_stats(state).get('melee_damage')
    if (data.get('split') != 'train' or state.get('screen_kind') != 'gameplay'
            or reviews.get('screen_kind') is not True or reviews.get('combat_stats') is not True
            or type(value) is not int or value <= 0):
        raise ValueError('Melee title requires a reviewed positive gameplay train reference')
    image_path = (path.parent / data.get('image', '')).resolve()
    if image_path.parent != (path.parent.parent / 'originals').resolve():
        raise ValueError('Invalid melee title image path')
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Unreadable melee title image')
    row = melee_damage_patch(normalize_game_screen(image, data.get('game_crop')))
    if row is None or row[:30].std() < 5:
        raise ValueError('Melee title is not visible')
    return row[:30], {
        'annotation': path.relative_to(root).as_posix(),
        'annotation_sha256': hashlib.sha256(raw).hexdigest(),
        'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
        'selection': selection,
    }
