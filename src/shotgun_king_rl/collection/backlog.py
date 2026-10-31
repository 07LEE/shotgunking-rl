"""Export unresolved recognition evidence without inventing labels."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from .card_catalog import load_catalog, slot_box, slot_side
from .images import normalize
from ..perception.royals import RoyalTemplates
from ..vision import analyzer


def build_backlog(collection, output, sessions=()):
    root, output = Path(collection), Path(output)
    paths = sorted((root / 'sessions').glob('*/annotations/*.json'))
    if sessions:
        paths = [path for path in paths if path.parent.parent.name in sessions]
    if not paths:
        raise ValueError('No matching collection annotations')
    output.mkdir(parents=True, exist_ok=False)
    catalog = load_catalog(root)
    royal_reader = RoyalTemplates.from_collection(root)
    royal_cells, card_slots, missing_fields, visual_groups = [], [], [], []
    counts = Counter()
    for screen_index, path in enumerate(paths):
        raw = path.read_bytes()
        data = json.loads(raw)
        image_path = (path.parent / data['image']).resolve()
        if image_path.parent != (path.parent.parent / 'originals').resolve():
            raise ValueError('Invalid original image path')
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError('Unreadable original image')
        screen = normalize(image, data['game_crop'])
        state = data.get('screen_state') or {}
        annotation = path.relative_to(root).as_posix()
        counts['screens'] += 1
        counts['piece_reviews_pending'] += not data.get('confirmed') and state.get('screen_kind') in (None, 'gameplay')
        if state.get('screen_kind') == 'gameplay':
            board = analyzer.crop_chessboard(screen)
            for row, labels in enumerate(data['board']):
                for col, label in enumerate(labels):
                    if label != 'ambiguous_royal' or [row, col] in data.get('exclude_cells', []):
                        continue
                    cell = board[row * 65:(row + 1) * 65, col * 65:(col + 1) * 65]
                    name = f'royal_{screen_index:04d}_{row}_{col}.png'
                    if not cv2.imwrite(str(output / name), cell):
                        raise OSError('Unable to export royal cell')
                    signal = analyzer._extract_signal(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)).astype(np.float32)
                    scores = {}
                    for kind, template in royal_reader.examples:
                        if signal.std() < 5 or template.std() < 5:
                            continue
                        score = float(cv2.matchTemplate(np.pad(signal, 1, mode='edge'), template, cv2.TM_CCOEFF_NORMED).max())
                        key = 'queen' if kind == 8 else 'white_king'
                        scores[key] = max(scores.get(key, -1), score)
                    royal_cells.append({'annotation': annotation, 'row': row, 'col': col, 'square': chr(97 + col) + str(8 - row), 'crop': name, 'candidate_scores': scores, 'confirmed': False, 'reason': 'Review or exclude; candidate scores are not labels'})
            for section, fields in (('ammo', ('loaded', 'reserve')), ('combat_stats', ('attack', 'range_min', 'range_max', 'spread_degrees', 'penetration_percent', 'knockback_percent', 'grenades_remaining', 'grenade_damage', 'melee_damage'))):
                for field in fields:
                    if (state.get(section) or {}).get(field) is None:
                        evidence = (data.get('prediction', {}).get('optional_combat') or {}).get(field)
                        missing_fields.append({'annotation': annotation, 'field': field, 'evidence': evidence, 'reason': 'Unknown; absence is not proof of zero'})
        if state.get('screen_kind') == 'card_selection':
            choices = state.get('card_choices') or [None] * 4
            for index, value in enumerate(choices):
                slot = f'choice:{index}'
                reviewed = state.get('reviews', {}).get('cards') or slot in state.get('card_reviewed_slots', [])
                if value is not None and reviewed:
                    continue
                left, top, right, bottom = slot_box(slot)
                card = screen[top:bottom, left:right]
                gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
                group = None
                if gray.std() >= 5:
                    scores = []
                    for candidate in visual_groups:
                        if candidate['split'] != data['split'] or candidate['side'] != slot_side(slot):
                            continue
                        template = candidate['_template']
                        score = float(cv2.matchTemplate(np.pad(gray, 4, mode='edge'), template, cv2.TM_CCOEFF_NORMED).max())
                        scores.append((score, candidate))
                    scores.sort(key=lambda pair: pair[0], reverse=True)
                    if scores and scores[0][0] >= .97 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= .04):
                        group = scores[0][1]
                    if group is None:
                        name = f'card_{len(visual_groups):04d}.png'
                        if not cv2.imwrite(str(output / name), card):
                            raise OSError('Unable to export card image')
                        group = {'id': len(visual_groups), 'split': data['split'], 'side': slot_side(slot), 'crop': name, 'occurrences': [], '_template': gray}
                        visual_groups.append(group)
                    group['occurrences'].append({'annotation': annotation, 'slot': slot})
                card_slots.append({'annotation': annotation, 'slot': slot, 'proposed_key': value, 'visual_group': group['id'] if group is not None else None})
        if path.read_bytes() != raw:
            raise ValueError('Annotation changed during backlog export; retry with a new output directory')
    for group in visual_groups:
        del group['_template']
    summary = {**counts, 'ambiguous_royal_cells': len(royal_cells), 'card_choices_needing_review': len(card_slots), 'card_visual_groups': len(visual_groups), 'unknown_gameplay_fields': len(missing_fields), 'catalog_cards': len(catalog['entries']), 'catalog_cards_with_references': sum(bool(entry['references']) for entry in catalog['entries']), 'catalog_effects_reviewed': sum(entry.get('effects') is not None for entry in catalog['entries'])}
    report = {'summary': summary, 'royal_cells': royal_cells, 'card_slots': card_slots, 'visual_groups': visual_groups, 'unknown_fields': missing_fields, 'notice': 'Read-only review queue. Similar images are not identified cards. Visual groups never cross train/val/test splits. No approval, game input, or training performed.'}
    (output / 'backlog.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=Path('data/collection'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--session', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(build_backlog(args.collection, args.output, args.session)['summary']))
