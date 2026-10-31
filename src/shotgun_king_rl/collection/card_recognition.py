"""Recognize card slots using reviewed train associations from the catalog."""

import cv2

from .card_catalog import load_catalog, reference_patch, slot_box, slot_side


def match_slot(screen, slot, examples):
    left, top, right, bottom = slot_box(slot)
    query = cv2.cvtColor(screen[top - 4:bottom + 4, left - 4:right + 4], cv2.COLOR_BGR2GRAY)
    if query.std() < 5:
        return None
    scores = {}
    for key, template in examples:
        if template.std() < 5:
            continue
        score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED).max())
        scores[key] = max(scores.get(key, -1), score)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    if not ranked or ranked[0][1] < .97 or (len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04):
        return None
    return ranked[0]


def propose_cards(screen, collection, screen_kind, exclude=None):
    catalog = load_catalog(collection)
    examples = {'black': [], 'white': []}
    entries = {entry['key']: entry for entry in catalog['entries']}
    for entry in entries.values():
        for reference in entry['references']:
            if slot_side(reference['slot']) != entry['side']:
                raise ValueError('Card reference side conflicts with its catalog entry')
            template = reference_patch(collection, reference, entry['key'], exclude)
            if template is not None:
                examples[entry['side']].append((entry['key'], template))
    for reference in catalog.get('empty_references', []):
        template = reference_patch(collection, reference, 'empty', exclude)
        if template is not None:
            examples[slot_side(reference['slot'])].append(('empty', template))
    cards = {'left': None, 'right': None}
    choices = [None] * 4 if screen_kind == 'card_selection' else None
    matches = []
    slots = [f'{side}:{index}' for side in cards for index in range(10)]
    if choices is not None:
        slots += [f'choice:{index}' for index in range(4)]
    for slot in slots:
        result = match_slot(screen, slot, examples[slot_side(slot)])
        if result is None:
            continue
        key, score = result
        group, index = slot.split(':')
        if group == 'choice':
            if key == 'empty':
                continue
            choices[int(index)] = key
        else:
            if cards[group] is None:
                cards[group] = ['unknown'] * 10
            cards[group][int(index)] = key
        matches.append({'slot': slot, 'key': key, 'confidence': score, 'effects': entries[key].get('effects') if key in entries else None})
    return {'cards': cards, 'card_choices': choices, 'matches': matches, 'catalog_revision': catalog['revision'], 'reference_count': sum(map(len, examples.values()))}
