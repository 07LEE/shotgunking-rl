"""Evaluate the partial melee damage reader without game input or model training."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from shotgun_king_rl.perception import MeleeDamageReader
from shotgun_king_rl.vision.layout import normalize_game_screen
from shotgun_king_rl.perception.melee import melee_damage_patch


def evaluate(collection, title_annotation):
    collection = Path(collection)
    title_annotation = Path(title_annotation).resolve()
    if not title_annotation.is_relative_to((collection / 'sessions').resolve()):
        raise ValueError('Title reference must belong to the collection')
    reference = json.loads(title_annotation.read_text())
    state = reference.get('screen_state') or {}
    reviews = state.get('reviews') or {}
    value = (state.get('combat_stats') or {}).get('melee_damage')
    if (reference.get('split') != 'train' or state.get('screen_kind') != 'gameplay'
            or reviews.get('screen_kind') is not True or reviews.get('combat_stats') is not True
            or type(value) is not int or value <= 0):
        raise ValueError('Title reference must be reviewed positive gameplay train data')
    image_path = (title_annotation.parent / reference.get('image', '')).resolve()
    if image_path.parent != (title_annotation.parent.parent / 'originals').resolve():
        raise ValueError('Invalid title reference image path')
    image = normalize_game_screen(cv2.imread(str(image_path)), reference.get('game_crop'))
    patch = melee_damage_patch(image)
    reader = MeleeDamageReader.from_collection(collection, title_reference=None if patch is None else patch[:30])
    title_image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    samples = []
    digest = hashlib.sha256()
    for path in sorted((collection / 'sessions').glob('*/annotations/*.json')):
        raw = path.read_bytes()
        data = json.loads(raw)
        if data.get('split') == 'test':
            continue
        state = data.get('screen_state') or {}
        value = (state.get('combat_stats') or {}).get('melee_damage')
        reviews = state.get('reviews') or {}
        if (state.get('screen_kind') != 'gameplay'
            or reviews.get('screen_kind') is not True
            or reviews.get('combat_stats') is not True
            or (value is not None and (type(value) is not int or value < 0))):
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
        labeled = sum(row['expected'] is not None for row in rows)
        correct = sum(row['accepted'] and row['expected'] is not None and row['expected'] == row['predicted'] for row in rows)
        metrics[split] = {
            'samples': len(rows), 'labeled_samples': labeled, 'accepted': accepted, 'correct': correct,
            'accuracy': correct / labeled if labeled else None,
            'accepted_accuracy': correct / accepted if accepted else None,
            'unknown': len(rows) - accepted,
            'accepted_without_label': sum(row['accepted'] and row['expected'] is None for row in rows),
            'values': dict(sorted(Counter(row['expected'] for row in rows if row['expected'] is not None).items())),
        }
    return {
        'title_annotation': str(title_annotation.relative_to(collection.resolve())),
        'title_annotation_sha256': hashlib.sha256(title_annotation.read_bytes()).hexdigest(),
        'title_image_sha256': title_image_sha256,
        'source': reader.source, 'dataset_sha256': digest.hexdigest(),
        'training_references': len(reader.examples), 'splits': metrics, 'samples': samples,
        'limitations': 'Train scores reuse reference screens. Positive visible melee damage with caller-verified train title only. Zero and absent rows remain unknown. Test images excluded. No complete combat stats or policy readiness. Unseen values unsupported.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=ROOT / 'data/collection')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--title-annotation', type=Path, required=True, help='Reviewed train annotation whose visible melee title was verified')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Output exists; choose a new report path')
    rendered = json.dumps(evaluate(args.collection, args.title_annotation), indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as report:
            report.write(rendered)
    print(rendered, end='')
