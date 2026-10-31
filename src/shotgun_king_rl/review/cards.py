"""Explicit human card review, independent of piece and page approval."""

import hashlib
import json
from pathlib import Path

from ..collection.card_catalog import load_catalog, slot_box, slot_side, validate_catalog


def save_card_review(collection, payload):
    # Imported lazily because the HTTP server also imports this handler.
    from .server import load_item, validate_screen_state

    path, current, image, revision = load_item(collection, payload['id'])
    if payload.get('revision') != revision:
        raise ValueError('Annotation changed; reload before saving cards')
    catalog = load_catalog(collection)
    if payload.get('catalog_revision') != catalog['revision']:
        raise ValueError('Card catalog changed; reload before saving cards')
    if type(payload.get('confirmed')) is not bool or type(payload.get('screen_kind_reviewed')) is not bool:
        raise ValueError('Explicit card and screen review flags are required')
    entries = {entry['key']: entry for entry in catalog['entries']}
    state = validate_screen_state(current.get('screen_state', {}))
    choices = payload.get('card_choices')
    if state['screen_kind'] == 'card_selection':
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError('Four card choices are required')
    elif choices is not None:
        raise ValueError('Choices apply only to card-selection screens')
    cards = payload.get('cards')
    if not isinstance(cards, dict) or set(cards) != {'left', 'right'} or any(not isinstance(values, list) or len(values) != 10 for values in cards.values()):
        raise ValueError('Ten persistent card slots per side are required')
    slots = {f'{side}:{index}': key for side, values in cards.items() for index, key in enumerate(values)}
    if choices is not None:
        slots.update({f'choice:{index}': key for index, key in enumerate(choices)})
    for slot, key in slots.items():
        slot_box(slot)
        if key is None:
            continue
        if key == 'empty' and not slot.startswith('choice:'):
            continue
        if not isinstance(key, str) or key not in entries or entries[key]['side'] != slot_side(slot):
            raise ValueError('Unknown card key or wrong card side')
    descriptions = payload.get('effect_descriptions', {})
    if not isinstance(descriptions, dict):
        raise ValueError('Invalid effect descriptions')
    for key, text in descriptions.items():
        if key not in slots.values() or key not in entries or not isinstance(text, str) or len(text) > 10000:
            raise ValueError('Effects must describe a selected catalog card')
    state['card_choices'] = choices
    state['cards'] = {side: ['unknown' if key is None else key for key in values] for side, values in cards.items()}
    reviewed_slots = [slot for slot, key in slots.items() if key is not None] if payload['confirmed'] else []
    state['card_reviewed_slots'] = reviewed_slots
    state['reviews']['cards'] = payload['confirmed'] and len(reviewed_slots) == len(slots)
    state['reviews']['screen_kind'] = payload['screen_kind_reviewed']
    if state['reviews']['screen_kind'] and state['screen_kind'] is None:
        raise ValueError('Select and save the screen kind first')
    current['screen_state'] = state
    image_path = (path.parent / current['image']).resolve()
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    # Remove superseded associations from this annotation, including drafts.
    for entry in entries.values():
        entry['references'] = [ref for ref in entry['references'] if ref['annotation'] != payload['id']]
    catalog['empty_references'] = [ref for ref in catalog.get('empty_references', []) if ref['annotation'] != payload['id']]
    eligible = payload['confirmed'] and payload['screen_kind_reviewed'] and current['split'] == 'train'
    for key, entry in entries.items():
        effects = entry.get('effects')
        if effects and effects.get('source_annotation') == payload['id'] and (not eligible or key not in slots.values()):
            entry['effects'] = None
    if eligible:
        for slot, key in slots.items():
            if slot not in reviewed_slots:
                continue
            reference = {'annotation': payload['id'], 'slot': slot, 'image_sha256': image_hash, 'game_crop': current['game_crop']}
            if key == 'empty':
                catalog['empty_references'].append(reference)
            else:
                entries[key]['references'].append(reference)
    # Descriptions from held-out screens must not become recognition metadata.
    if eligible:
        for key, text in descriptions.items():
            if text.strip():
                entries[key]['effects'] = {'description': text.strip(), 'reviewed': True, 'source_annotation': payload['id']}
    catalog['revision'] += 1
    validate_catalog(catalog)
    # Write the annotation first: stale references are rejected by label checks.
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(current, indent=2) + '\n')
    temporary.replace(path)
    catalog_path = Path(collection) / 'card_catalog.json'
    temporary = catalog_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(catalog_path)
    return {'revision': hashlib.sha256(path.read_bytes()).hexdigest(), 'catalog_revision': catalog['revision'], 'training_references_added': eligible}
