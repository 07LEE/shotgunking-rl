"""Extract unlabeled card-choice images while preserving session provenance."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2

from .images import normalize
from ..vision.screens import check_card_selection_screen

ROOT = Path(__file__).resolve().parents[3]
# Include the full calibrated card border plus a small margin.
CHOICE_BOXES = (
    (1, 'black', (538, 214, 626, 334)),
    (1, 'white', (650, 214, 738, 334)),
    (2, 'black', (538, 414, 626, 534)),
    (2, 'white', (650, 414, 738, 534)),
)


def extract_card_choices(sessions, output, include_unreviewed=False):
    sessions, output = Path(sessions), Path(output)
    if output.exists():
        raise ValueError(f'Output exists: {output}; choose a new directory')
    sources = []
    splits = {}
    for path in sorted(sessions.glob('*/annotations/*.json')):
        raw = path.read_bytes()
        data = json.loads(raw)
        state = data.get('screen_state') or {}
        if state.get('screen_kind') != 'card_selection':
            continue
        reviewed = (state.get('reviews') or {}).get('screen_kind') is True
        if not reviewed and not include_unreviewed:
            continue
        session = path.parent.parent.name
        split = data.get('split')
        if data.get('session_id') != session or split not in ('train', 'val', 'test'):
            raise ValueError(f'{path}: invalid session or split')
        if session in splits and splits[session] != split:
            raise ValueError(f'{path}: session spans multiple splits')
        splits[session] = split
        image_path = (path.parent / data.get('image', '')).resolve()
        if image_path.parent != (path.parent.parent / 'originals').resolve():
            raise ValueError(f'{path}: invalid image path')
        image_raw = image_path.read_bytes()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'{path}: unreadable image')
        screen = normalize(image, data.get('game_crop'))
        if not check_card_selection_screen(screen):
            raise ValueError(f'{path}: card-choice layout not verified')
        sources.append((path, data, screen, reviewed, hashlib.sha256(raw).hexdigest(), hashlib.sha256(image_raw).hexdigest()))
    if not sources:
        raise ValueError('No eligible card-selection screens')
    output.mkdir(parents=True)
    records = []
    for index, (path, data, screen, reviewed, annotation_hash, image_hash) in enumerate(sources):
        folder = output / data['session_id']
        folder.mkdir(exist_ok=True)
        for choice, side, box in CHOICE_BOXES:
            left, top, right, bottom = box
            relative = Path(data['session_id']) / f'{index:04d}_choice_{choice}_{side}.png'
            if not cv2.imwrite(str(output / relative), screen[top:bottom, left:right]):
                raise OSError(f'Failed to write {relative}')
            records.append({
                'image': relative.as_posix(), 'session_id': data['session_id'], 'split': data['split'],
                'source_annotation': path.relative_to(sessions).as_posix(),
                'source_image': data['image'], 'source_image_sha256': image_hash,
                'source_annotation_sha256': annotation_hash, 'game_crop': data.get('game_crop'),
                'normalized_box': list(box), 'choice': choice, 'side': side,
                'screen_kind_reviewed': reviewed, 'card_key': None, 'human_review_required': True,
            })
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    summary = {
        'screens': len(sources), 'card_images': len(records),
        'screens_by_split': dict(Counter(data['split'] for _, data, *_ in sources)),
        'unreviewed_screens': sum(not reviewed for _, _, _, reviewed, *_ in sources),
        'card_labels': 'Unlabeled crops; no card identity, effects, or selected choice inferred',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', type=Path, default=ROOT / 'data/collection/sessions')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-unreviewed', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(extract_card_choices(args.sessions, args.output, args.include_unreviewed), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Card extraction failed: {exc}\n')
