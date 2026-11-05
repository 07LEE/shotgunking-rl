"""Classify existing images into unapproved visibility drafts only."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2

from ..collection_layout import annotation_paths
from ..perception.visibility_proposals import CombatVisibilityClassifier
from ..perception.visibility import combat_visibility
from .images import normalize


def classify_visibility(collection, *, apply=False, classifier=None):
    root = Path(collection).resolve()
    classifier = classifier or CombatVisibilityClassifier.from_collection(root)
    report = {'applied': apply, 'results': [], 'counts': Counter(), 'errors': [], 'test_skipped': 0}
    for path in annotation_paths(root):
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
            if data.get('split') == 'test':
                report['test_skipped'] += 1
                continue
            state = data.get('screen_state') or {}
            if state.get('screen_kind') != 'gameplay':
                continue
            image_path = (path.parent / data['image']).resolve()
            if image_path.parent != (path.parent.parent / 'originals').resolve():
                raise ValueError('Invalid original image path')
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError('Unreadable original image')
            labels = classifier.propose(normalize(image, data.get('game_crop')))
            manual = combat_visibility(state)
            proposals = {field: status for field, status in labels.items()
                         if status != 'unknown' and field not in manual}
            report['counts'].update(proposals.values())
            provenance = {'method': classifier.source, 'human_review_required': True,
                          'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest()}
            prediction = data.setdefault('prediction', {})
            changed = (prediction.get('combat_visibility') != proposals
                       or prediction.get('combat_visibility_provenance') != provenance)
            if apply and changed:
                if path.read_bytes() != raw:
                    raise ValueError('Annotation changed while classifying')
                prediction.update(combat_visibility=proposals, combat_visibility_provenance=provenance)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
                temporary.replace(path)
            report['results'].append({'annotation': str(path.relative_to(root)), 'labels': labels,
                                      'proposals': proposals, 'changed': changed})
        except (ValueError, OSError, KeyError, cv2.error) as error:
            report['errors'].append({'annotation': str(path), 'error': str(error)})
    report['counts'] = dict(report['counts'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=Path('data/collection'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    result = classify_visibility(args.collection, apply=args.apply)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'counts': result['counts'], 'errors': result['errors'], 'test_skipped': result['test_skipped']}))


if __name__ == '__main__':
    main()
