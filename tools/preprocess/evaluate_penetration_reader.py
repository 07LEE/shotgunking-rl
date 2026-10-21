"""Evaluate the partial penetration reader without game input or model training."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from shotgun_king_rl.perception import PenetrationReader
from shotgun_king_rl.vision.layout import normalize_game_screen


def evaluate(collection):
    collection = Path(collection)
    reader = PenetrationReader.from_collection(collection)
    samples = []
    digest = hashlib.sha256()
    for path in sorted((collection / 'sessions').glob('*/annotations/*.json')):
        raw = path.read_bytes()
        data = json.loads(raw)
        state = data.get('screen_state') or {}
        value = (state.get('combat_stats') or {}).get('penetration_percent')
        reviews = state.get('reviews') or {}
        if (state.get('screen_kind') != 'gameplay'
            or reviews.get('screen_kind') is not True
            or reviews.get('combat_stats') is not True
            or type(value) is not int or not 0 <= value <= 100):
            continue
        if data.get('split') not in ('train', 'val', 'test'):
            raise ValueError(f'{path}: invalid split')
        image_path = (path.parent / data.get('image', '')).resolve()
        if image_path.parent != (path.parent.parent / 'originals').resolve():
            raise ValueError(f'{path}: invalid image path')
        image_raw = image_path.read_bytes()
        digest.update(str(path.relative_to(collection)).encode())
        digest.update(raw)
        digest.update(hashlib.sha256(image_raw).digest())
        image = cv2.imread(str(image_path))
        image = normalize_game_screen(image, data.get('game_crop'))
        result = reader.read(image)
        samples.append({
            'annotation': str(path.relative_to(collection)), 'split': data['split'],
            'expected': value, 'predicted': result.value,
            'accepted': result.is_accepted, 'reason': result.reason,
        })
    metrics = {}
    for split in ('train', 'val', 'test'):
        rows = [sample for sample in samples if sample['split'] == split]
        accepted = sum(row['accepted'] for row in rows)
        correct = sum(row['accepted'] and row['expected'] == row['predicted'] for row in rows)
        metrics[split] = {
            'samples': len(rows), 'accepted': accepted, 'correct': correct,
            'accuracy': correct / len(rows) if rows else None,
            'accepted_accuracy': correct / accepted if accepted else None,
            'unknown': len(rows) - accepted,
            'values': dict(sorted(Counter(row['expected'] for row in rows).items())),
        }
    return {
        'source': reader.source, 'dataset_sha256': digest.hexdigest(),
        'training_references': len(reader.examples) + len(reader.zero_examples),
        'positive_training_references': len(reader.examples), 'zero_training_references': len(reader.zero_examples), 'splits': metrics, 'samples': samples,
        'limitations': 'Train scores reuse reference screens. Reviewed zero requires exact row and matching panel; unfamiliar absent rows remain unknown; no complete combat stats or policy readiness. Unseen HUD values are unsupported.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=ROOT / 'data/collection')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Output exists; choose a new report path')
    rendered = json.dumps(evaluate(args.collection), indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as report:
            report.write(rendered)
    print(rendered, end='')
