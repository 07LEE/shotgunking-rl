"""Refresh blank, unreviewed collection fields without changing manual labels."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2

from .images import normalize
from .screen_state import propose


def fill_unknowns(data, proposal):
    """Fill supported blanks only; approval and board labels remain untouched."""
    changed = {}
    state = data.setdefault('screen_state', {})
    reviews = state.get('reviews') or {}
    predicted = proposal['screen_state']
    if not data.get('confirmed') and data.get('floor') is None and proposal.get('floor') is not None:
        data['floor'] = proposal['floor']
        changed['floor'] = proposal['floor']
    if not reviews.get('screen_kind') and state.get('screen_kind') is None and predicted.get('screen_kind') is not None:
        state['screen_kind'] = predicted['screen_kind']
        changed['screen_kind'] = predicted['screen_kind']
    for section in ('ammo', 'combat_stats'):
        if reviews.get(section):
            continue
        values = state.setdefault(section, {})
        for field, value in predicted.get(section, {}).items():
            if values.get(field) is None and value is not None:
                values[field] = value
                changed[f'{section}.{field}'] = value
    if not reviews.get('soul') and state.get('souls') is None and predicted.get('souls') is not None:
        state['souls'] = predicted['souls']
        changed['souls'] = predicted['souls']
    if not reviews.get('cards'):
        groups = [('cards', side, values) for side, values in predicted.get('cards', {}).items()]
        if predicted.get('card_choices') is not None:
            groups.append(('card_choices', None, predicted['card_choices']))
        for section, side, candidates in groups:
            if candidates is None:
                continue
            current = (state.get('cards') or {}).get(side) if side else state.get(section)
            values = list(current) if current is not None else [None] * len(candidates)
            # A shorter manual slot list is preserved rather than extended.
            for index in range(min(len(values), len(candidates))):
                if values[index] in (None, 'unknown') and candidates[index] not in (None, 'unknown'):
                    values[index] = candidates[index]
                    changed[f'{section}.{side or "choice"}.{index}'] = candidates[index]
            if values != current and any(key.startswith(f'{section}.{side or "choice"}.') for key in changed):
                if side:
                    state.setdefault('cards', {})[side] = values
                else:
                    state[section] = values
    return changed


def refresh_session(collection, session, output, apply=False):
    root, output = Path(collection), Path(output)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', session):
        raise ValueError('Invalid session ID')
    paths = sorted((root / 'sessions' / session / 'annotations').glob('*.json'))
    if not paths:
        raise ValueError('No prepared annotations for this session')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'session': session, 'applied': apply, 'annotations_changed': 0, 'fields_filled': 0, 'results': [], 'errors': []}
    with output.open('x') as stream:
        for path in paths:
            try:
                raw = path.read_bytes()
                data = json.loads(raw)
                image_path = (path.parent / data['image']).resolve()
                if image_path.parent != (path.parent.parent / 'originals').resolve():
                    raise ValueError('Invalid source image path')
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError('Unreadable original image')
                proposal = propose(normalize(image, data['game_crop']), root, exclude=path)
                changed = fill_unknowns(data, proposal)
                if path.read_bytes() != raw:
                    raise ValueError('Annotation changed during prediction; reload before retrying')
                if changed and apply:
                    evidence = data.setdefault('prediction', {})
                    evidence.setdefault('refreshes', []).append({'report': str(output), 'fields': changed, 'method': proposal['prediction']['method']})
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(data, indent=2) + '\n')
                    temporary.replace(path)
                report['annotations_changed'] += bool(changed)
                report['fields_filled'] += len(changed)
                report['results'].append({'annotation': path.relative_to(root).as_posix(), 'before_sha256': hashlib.sha256(raw).hexdigest(), 'fields': changed, 'prediction': proposal['prediction']})
            except (ValueError, KeyError, OSError, cv2.error) as error:
                report['errors'].append({'annotation': str(path), 'error': str(error)})
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=Path('data/collection'))
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Fill supported blank fields; default is a report only')
    args = parser.parse_args()
    result = refresh_session(args.collection, args.session, args.output, args.apply)
    print(json.dumps({key: result[key] for key in ('session', 'applied', 'annotations_changed', 'fields_filled', 'errors')}))
    if result['errors']:
        parser.exit(1, 'Some annotations need attention; see the report.\n')
