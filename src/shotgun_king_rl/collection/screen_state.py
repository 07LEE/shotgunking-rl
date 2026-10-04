"""Conservative screen-state proposals from private, reviewed examples."""
from pathlib import Path
import json

import cv2

from .images import normalize

# Coordinates refer to a normalized 1280x720 gameplay screenshot.
REGIONS = {
    'ammo_loaded': (382, 34, 550, 75),
    'ammo_reserve': (382, 78, 706, 111),
    'attack': (270, 262, 354, 300),
    'range_min': (270, 326, 354, 365),
    'range_max': (270, 326, 354, 365),
    'spread_degrees': (270, 390, 354, 430),
    'knockback_percent': (270, 454, 354, 492),
}


def patch(image, box):
    left, top, right, bottom = box
    return cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)


def match(query, examples, threshold=.97, margin=.04):
    """Reject ambiguous or featureless matches, including conflicting labels."""
    if query.std() < 5:
        return None
    scores = {}
    for label, template in examples:
        if template.shape != query.shape or template.std() < 5:
            continue
        score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
        scores[label] = max(scores.get(label, -1), score)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < threshold:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < margin:
        return None
    return ranked[0]


def propose(screen, collection, exclude=None):
    """Use only reviewed training references; never infer effects from card art."""
    if screen.shape[:2] != (720, 1280):
        raise ValueError('Screen must be normalized to 1280x720')
    examples = {key: [] for key in REGIONS}
    cards = []
    references = []
    for path in sorted((Path(collection) / 'sessions').glob('*/annotations/*.json')):
        if exclude is not None and path.resolve() == Path(exclude).resolve():
            continue
        data = json.loads(path.read_text())
        state = data.get('screen_state') or {}
        if data.get('split') != 'train' or not state.get('reviewed'):
            continue
        image_path = (path.parent / data['image']).resolve()
        if image_path.parent != (path.parent.parent / 'originals').resolve():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = normalize(image, data['game_crop'])
        references.append(str(path.relative_to(collection)))
        for key, box in REGIONS.items():
            value = state.get('stats', {}).get(key)
            if type(value) is int and value >= 0:
                examples[key].append((value, patch(image, box)))
        for side in ('left', 'right'):
            for index, label in enumerate(state.get('cards', {}).get(side) or []):
                if label != 'unknown' and index < 10:
                    cards.append((label, patch(image, card_box(side, index))))
    state = {'stats': {}, 'cards': {}, 'locked_cells': [], 'reviewed': False}
    confidence = {}
    for key, box in REGIONS.items():
        result = match(patch(screen, box), examples[key])
        state['stats'][key] = result[0] if result else None
        confidence[key] = result[1] if result else None
    for side in ('left', 'right'):
        # Preserve slot positions; trailing unidentified slots remain unknown too.
        values = []
        for index in range(10):
            result = match(patch(screen, card_box(side, index)), cards)
            values.append(result[0] if result else 'unknown')
            confidence[f'{side}_{index}'] = result[1] if result else None
        state['cards'][side] = values if any(v != 'unknown' for v in values) else None
    return {'screen_state': state, 'prediction': {'method': 'reviewed-region-template-v1', 'confidence': confidence, 'references': references, 'human_review_required': True}}


def card_box(side, index):
    left = (30 if side == 'left' else 1062) + (index % 2) * 96
    top = 50 + (index // 2) * 128
    return left, top, left + 84, min(top + 116, 720)
