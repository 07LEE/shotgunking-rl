"""Conservative screen-state proposals from private, reviewed examples."""
from pathlib import Path
import json

import cv2
import numpy as np

from .images import normalize
from ..vision import analyzer, screens
from ..perception.hud import stat_patch
from .numeric import numeric_token, match_numeric, match_floor_shapes
from .named_stats import match_named_integer
from .optional_combat import propose_optional_combat

# Coordinates refer to a normalized 1280x720 gameplay screenshot.
REGIONS = {
    'loaded': (382, 34, 550, 75),
    'reserve': (382, 78, 706, 111),
    'grenades_remaining': (592, 78, 642, 112),
}
STAT_OFFSETS = {'attack': 0, 'range': 64, 'spread_degrees': 128}
FLOOR_REGION = (670, 15, 725, 75)
SOUL_REGION = (928, 311, 1013, 428)
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
    if query is None or query.std() < 5:
        return None
    scores = {}
    for label, template in examples:
        if any(t > q for t, q in zip(template.shape, query.shape)) or template.std() < 5:
            continue
        score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED).max())
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
    stat_examples = {key: [] for key in STAT_OFFSETS}
    numeric_examples = {key: [] for key in STAT_OFFSETS}
    grenade_damage_examples = []
    optional_titles = []
    integer_examples = []
    floor_examples = []
    souls = []
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
        if reviews['combat_stats']:
            stats = _section(state, 'combat_stats')
            damage = stats.get('grenade_damage')
            if type(damage) is int and damage >= 0:
                template = stat_patch(image, row_offset=256)
                if template is not None:
                    grenade_damage_examples.append((damage, template))
                    optional_titles.append(('grenade_damage', template[:30]))
                    token = numeric_token(image, 256)
                    if token is not None:
                        integer_examples.append((damage, token))
            for key, offset in STAT_OFFSETS.items():
                value = stats.get(key)
                if key == 'range':
                    value = (stats.get('range_min'), stats.get('range_max'))
                    valid = all(type(v) is int for v in value) and 0 <= value[0] <= value[1]
                else:
                    valid = type(value) is int and value >= 0
                template = stat_patch(image, row_offset=offset)
                if valid and template is not None:
                    stat_examples[key].append((value, template))
                    token = numeric_token(image, offset)
                    if token is not None:
                        numeric_examples[key].append((value, token, template))
                        if key == 'attack':
                            integer_examples.append((value, token))
                    used = True
        if reviews['cards']:
            for side in ('left', 'right'):
                for index, label in enumerate(state.get('cards', {}).get(side) or []):
                    if label != 'unknown' and index < 10:
                        cards.append((label, patch(image, card_box(side, index))))
                        used = True
        reviewed_souls = state.get('souls')
        if reviews['soul'] and reviewed_souls is not None and len(reviewed_souls) <= 1:
            souls.append(((reviewed_souls or ['none'])[0], patch(image, SOUL_REGION)))
            used = True
        if used:
            references.append(str(path.relative_to(collection)))
    screen_kind = propose_screen_kind(screen)
    state = {
        'screen_kind': screen_kind,
        'ammo': {},
        'combat_stats': {key: None for key in COMBAT_FIELDS},
        'cards': {'left': None, 'right': None},
        'souls': None,
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
    numeric_fields = []
    confidence['screen_kind'] = 1.0 if screen_kind is not None else None
    floor_result = match(floor_patch(screen), floor_examples, threshold=.85, margin=.1)
    floor_method = 'red-mask-correlation-v1' if floor_result else None
    if floor_result is None:
        floor_result = match_floor_shapes(floor_patch(screen), floor_examples)
        if floor_result is not None:
            floor_method = 'red-stroke-distance-v1'
    confidence['floor'] = floor_result[1] if floor_result else None
    for key, box in REGIONS.items():
        if key == 'grenades_remaining':
            left, top, right, bottom = box
            query = patch(screen, (left - 4, top - 4, right + 4, bottom + 4))
        else:
            query = patch(screen, box)
        result = match(query, examples[key])
        section = 'ammo' if key in ('loaded', 'reserve') else 'combat_stats'
        state[section][key] = result[0] if result else None
        confidence[key] = result[1] if result else None
    damage_candidates = []
    for offset in (192, 256, 320):
        result = match(stat_patch(screen, row_offset=offset, padding=8), grenade_damage_examples)
        if result is not None:
            damage_candidates.append(result)
    if damage_candidates and len({value for value, _ in damage_candidates}) == 1:
        state['combat_stats']['grenade_damage'], confidence['grenade_damage'] = max(damage_candidates, key=lambda result: result[1])
    if state['combat_stats']['grenade_damage'] is None:
        result = match_named_integer(screen, 'grenade_damage', optional_titles, integer_examples)
        if result is not None:
            state['combat_stats']['grenade_damage'], confidence['grenade_damage'] = result
            numeric_fields.append('grenade_damage')
    optional_values, optional_evidence = propose_optional_combat(screen, collection, exclude=exclude)
    state['combat_stats'].update(optional_values)
    for field, evidence in optional_evidence.items():
        confidence[field] = evidence['confidence']
    for key, offset in STAT_OFFSETS.items():
        context = stat_patch(screen, row_offset=offset, padding=8)
        result = match(context, stat_examples[key])
        if result is None:
            result = match_numeric(numeric_token(screen, offset), context, numeric_examples[key])
            if result is not None:
                numeric_fields.append(key)
        fields = ('range_min', 'range_max') if key == 'range' else (key,)
        values = result[0] if result and key == 'range' else (result[0],) if result else (None,) * len(fields)
        for field, value in zip(fields, values):
            state['combat_stats'][field] = value
            confidence[field] = result[1] if result else None
    for side in ('left', 'right'):
        # Preserve slot positions; trailing unidentified slots remain unknown too.
        values = []
        for index in range(10):
            result = match(patch(screen, card_box(side, index)), cards)
            values.append(result[0] if result else 'unknown')
            confidence[f'{side}_{index}'] = result[1] if result else None
        state['cards'][side] = values if any(v != 'unknown' for v in values) else None
    left, top, right, bottom = SOUL_REGION
    soul_result = match(patch(screen, (left - 4, top - 4, right + 4, bottom + 4)), souls, threshold=.9, margin=.04)
    state['souls'] = ([] if soul_result[0] == 'none' else [soul_result[0]]) if soul_result else None
    confidence['soul'] = soul_result[1] if soul_result else None
    return {'floor': floor_result[0] if floor_result else None, 'screen_state': state, 'prediction': {'method': 'reviewed-region-template-v8', 'optional_combat': optional_evidence, 'floor_method': floor_method, 'numeric_fields': numeric_fields, 'confidence': confidence, 'references': references, 'human_review_required': True}}


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
