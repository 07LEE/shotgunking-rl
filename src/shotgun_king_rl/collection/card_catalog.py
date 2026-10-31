"""Versioned card names and effects backed by local game files and review."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2

from .images import normalize


SCHEMA = 'card-catalog-v1'
CHOICE_BOXES = ((540, 216, 624, 332), (651, 216, 735, 332), (540, 416, 624, 532), (651, 416, 735, 532))


def slot_box(slot):
    group, raw_index = slot.split(':')
    index = int(raw_index)
    if group == 'choice' and 0 <= index < 4:
        return CHOICE_BOXES[index]
    if group not in ('left', 'right') or not 0 <= index < 10:
        raise ValueError('Invalid card slot')
    left = (30 if group == 'left' else 1062) + (index % 2) * 96
    top = 50 + (index // 2) * 128
    return left, top, left + 84, top + 116


def slot_side(slot):
    group, index = slot.split(':')
    return ('black' if int(index) % 2 == 0 else 'white') if group == 'choice' else 'black' if group == 'left' else 'white'


def translations(text):
    return dict(line.split('::', 1) for line in text.splitlines() if '::' in line and not line.lstrip().startswith('--'))


def import_catalog(game_dir, collection):
    """Import names and untranslated effect templates, never guess associations."""
    game_dir, collection = Path(game_dir), Path(collection)
    english_path, korean_path = game_dir / 'lang/english.txt', game_dir / 'lang/korean.txt'
    english = english_path.read_text(encoding='utf-8-sig')
    korean = translations(korean_path.read_text(encoding='utf-8-sig'))
    entries, side = [], None
    for line in english.splitlines():
        if '-- CARD NAMES' in line:
            side = 'black'
        elif '-- Next are the names for the white cards' in line:
            side = 'white'
        elif '-- CARDS DYNAMIC' in line:
            break
        elif side and '::' in line and not line.lstrip().startswith('--'):
            source_key, name = line.split('::', 1)
            key = re.sub(r'[^a-z0-9]+', '_', source_key.lower()).strip('_')
            entries.append({'key': key, 'source_key': source_key, 'side': side, 'names': {'en': name, 'ko': korean.get(source_key)}, 'effects': None, 'references': []})
    if not entries or not any(entry['side'] == 'white' for entry in entries):
        raise ValueError('Card-name sections not found in the game localization')
    localized = translations(english)
    catalog = {'schema': SCHEMA, 'revision': 1, 'source': {'kind': 'installed-game-localization', 'english_sha256': hashlib.sha256(english_path.read_bytes()).hexdigest(), 'korean_sha256': hashlib.sha256(korean_path.read_bytes()).hexdigest()}, 'entries': entries, 'empty_references': [], 'effect_templates': {key: {'en': value, 'ko': korean.get(key)} for key, value in localized.items() if key.startswith('effect_')}}
    validate_catalog(catalog)
    collection.mkdir(parents=True, exist_ok=True)
    with (collection / 'card_catalog.json').open('x') as stream:
        json.dump(catalog, stream, ensure_ascii=False, indent=2)
    return catalog


def validate_catalog(catalog):
    if catalog.get('schema') != SCHEMA or type(catalog.get('revision')) is not int or catalog['revision'] < 1:
        raise ValueError('Invalid card catalog version')
    keys = set()
    for entry in catalog.get('entries', []):
        key = entry.get('key')
        if not isinstance(key, str) or not re.fullmatch(r'[a-z0-9_]+', key) or key in keys or key in ('empty', 'unknown'):
            raise ValueError('Invalid or duplicate card key')
        keys.add(key)
        if entry.get('side') not in ('black', 'white') or not isinstance(entry.get('names'), dict) or not entry['names'].get('en'):
            raise ValueError('Invalid card name or side')
        effects = entry.get('effects')
        if effects is not None and (not isinstance(effects, dict) or not isinstance(effects.get('description'), str) or not effects['description'].strip() or effects.get('reviewed') is not True):
            raise ValueError('Card effects require an explicitly reviewed description')
        if not isinstance(entry.get('references'), list):
            raise ValueError('Invalid card references')
    return catalog


def load_catalog(collection):
    path = Path(collection) / 'card_catalog.json'
    if not path.exists():
        return {'schema': SCHEMA, 'revision': 1, 'entries': [], 'empty_references': []}
    return validate_catalog(json.loads(path.read_text()))


def reference_patch(collection, reference, expected_key, exclude=None):
    """Accept only unchanged, explicitly reviewed training slot associations."""
    root = Path(collection).resolve()
    path = (root / reference['annotation']).resolve()
    if root not in path.parents or path.suffix != '.json':
        raise ValueError('Invalid card reference path')
    if exclude is not None and path == Path(exclude).resolve():
        return None
    data = json.loads(path.read_text())
    state = data.get('screen_state') or {}
    reviews = state.get('reviews') or {}
    slot = reference['slot']
    slot_reviewed = reviews.get('cards') is True or slot in (state.get('card_reviewed_slots') or [])
    if data.get('split') != 'train' or not slot_reviewed or reviews.get('screen_kind') is not True:
        return None
    if 'game_crop' in reference and reference['game_crop'] != data.get('game_crop'):
        return None
    left, top, right, bottom = slot_box(slot)
    group, index = slot.split(':')
    if group == 'choice':
        values = state.get('card_choices') or []
        if state.get('screen_kind') != 'card_selection':
            return None
    else:
        values = (state.get('cards') or {}).get(group) or []
    if int(index) >= len(values) or values[int(index)] != expected_key:
        return None
    image_path = (path.parent / data['image']).resolve()
    if image_path.parent != (path.parent.parent / 'originals').resolve():
        raise ValueError('Invalid card reference image path')
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != reference['image_sha256']:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Unreadable card reference')
    image = normalize(image, data['game_crop'])
    return cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-dir', required=True, type=Path)
    parser.add_argument('--collection', type=Path, default=Path('data/collection'))
    args = parser.parse_args()
    catalog = import_catalog(args.game_dir, args.collection)
    print(json.dumps({'cards': len(catalog['entries']), 'effect_templates': len(catalog['effect_templates']), 'reviewed_image_associations': 0}))
