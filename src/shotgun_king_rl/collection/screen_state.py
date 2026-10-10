"""Conservative screen-state proposals from private, reviewed examples."""
from pathlib import Path
import json

import cv2
import numpy as np

from .images import normalize
from ..vision import analyzer, screens

# Coordinates refer to a normalized 1280x720 gameplay screenshot.
REGIONS = {
    'loaded': (382, 34, 550, 75),
    'reserve': (382, 78, 706, 111),
    'attack': (270, 262, 354, 300),
    'range_min': (270, 326, 354, 365),
    'range_max': (270, 326, 354, 365),
    'spread_degrees': (270, 390, 354, 430),
    'knockback_percent': (270, 454, 354, 492),
}
FLOOR_REGION = (670, 15, 725, 75)
COMBAT_FIELDS = ('attack', 'range_min', 'range_max', 'spread_degrees', 'penetration_percent', 'knockback_percent', 'grenades_remaining', 'grenade_damage', 'melee_damage')


def patch(image, box):
    left, top, right, bottom = box
    return cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)


def floor_patch(image):
    """Return a normalized mask of the red floor digits, excluding blue label text."""
    left, top, right, bottom = FLOOR_REGION
    region = image[top:bottom, left:right]
    blue, green, red = cv2.split(region)
    red_i = red.astype(np.int16)
    mask = ((red_i > 110) & (red_i > green.astype(np.int16) + 25) & (red_i > blue.astype(np.int16) + 25)).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return np.zeros((36, 24), dtype=np.uint8)
    glyph = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(glyph, (24, 36), interpolation=cv2.INTER_NEAREST)


def propose_screen_kind(image):
    """Recognize explicit overlays, then accept gameplay only with one player king."""
    if screens.check_retry_popup(image):
        return 'retry'
    if screens.check_card_selection_screen(image):
        return 'card_selection'
    board = analyzer.get_state_matrix(image)
    return 'gameplay' if int(np.count_nonzero(board == 1)) == 1 else None


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
    floor_examples = []
    cards = []
    references = []
    for path in sorted((Path(collection) / 'sessions').glob('*/annotations/*.json')):
        if exclude is not None and path.resolve() == Path(exclude).resolve():
            continue
        data = json.loads(path.read_text())
        state = data.get('screen_state') or {}
        reviews = _reviews(state)
        floor = data.get('floor')
        has_floor_example = data.get('confirmed') is True and type(floor) is int and floor >= 1
        if data.get('split') != 'train' or not (any(reviews.values()) or has_floor_example):
            continue
        image_path = (path.parent / data['image']).resolve()
        if image_path.parent != (path.parent.parent / 'originals').resolve():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = normalize(image, data['game_crop'])
        used = False
        if has_floor_example:
            floor_examples.append((floor, floor_patch(image)))
            used = True
        for key, box in REGIONS.items():
            section = 'ammo' if key in ('loaded', 'reserve') else 'combat_stats'
            if not reviews[section]:
                continue
            value = _section(state, section).get(key)
            if type(value) is int and value >= 0:
                examples[key].append((value, patch(image, box)))
                used = True
        if reviews['cards']:
            for side in ('left', 'right'):
                for index, label in enumerate(state.get('cards', {}).get(side) or []):
                    if label != 'unknown' and index < 10:
                        cards.append((label, patch(image, card_box(side, index))))
                        used = True
        if used:
            references.append(str(path.relative_to(collection)))
    screen_kind = propose_screen_kind(screen)
    state = {
        'screen_kind': screen_kind,
        'ammo': {},
        'combat_stats': {key: None for key in COMBAT_FIELDS},
        'cards': {},
        'soul_piece': None,
        'locked_cells': [],
        'reviews': {
            'screen_kind': False,
            'ammo': False,
            'combat_stats': False,
            'cards': False,
            'soul': False,
        },
    }
    confidence = {}
    confidence['screen_kind'] = 1.0 if screen_kind is not None else None
    floor_result = match(floor_patch(screen), floor_examples, threshold=.85, margin=.1)
    confidence['floor'] = floor_result[1] if floor_result else None
    for key, box in REGIONS.items():
        result = match(patch(screen, box), examples[key])
        section = 'ammo' if key in ('loaded', 'reserve') else 'combat_stats'
        state[section][key] = result[0] if result else None
        confidence[key] = result[1] if result else None
    for side in ('left', 'right'):
        # Preserve slot positions; trailing unidentified slots remain unknown too.
        values = []
        for index in range(10):
            result = match(patch(screen, card_box(side, index)), cards)
            values.append(result[0] if result else 'unknown')
            confidence[f'{side}_{index}'] = result[1] if result else None
        state['cards'][side] = values if any(v != 'unknown' for v in values) else None
    return {'floor': floor_result[0] if floor_result else None, 'screen_state': state, 'prediction': {'method': 'reviewed-region-template-v1', 'confidence': confidence, 'references': references, 'human_review_required': True}}


def _reviews(state):
    reviews = state.get('reviews') or {}
    legacy = state.get('reviewed') is True
    return {
        'screen_kind': reviews.get('screen_kind') is True,
        'ammo': reviews.get('ammo', legacy) is True,
        'combat_stats': reviews.get('combat_stats', legacy) is True,
        'cards': reviews.get('cards', legacy) is True,
        'soul': reviews.get('soul', False) is True,
    }


def _section(state, name):
    if name in state:
        return state.get(name) or {}
    stats = state.get('stats') or {}
    if name == 'ammo':
        return {
            'loaded': stats.get('ammo_loaded'),
            'reserve': stats.get('ammo_reserve'),
        }
    return {key: stats.get(key) for key in COMBAT_FIELDS}


def card_box(side, index):
    left = (30 if side == 'left' else 1062) + (index % 2) * 96
    top = 50 + (index // 2) * 128
    return left, top, left + 84, min(top + 116, 720)
