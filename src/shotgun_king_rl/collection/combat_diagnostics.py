"""Read-only train/val diagnosis of numeric evidence and combat-state blockers."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..collection_layout import annotation_paths
from ..perception.combat import combat_state_contract
from ..perception.optional_zero import ZERO_WHEN_NOT_DISPLAYED, ZERO_RULE_SOURCE, connect_optional_zero_readers
from ..perception.frame import PreparedFrame, read_frame
from ..perception.types import RangeReading
from ..perception.visibility import COMBAT_FIELDS, visual_combat_stats
from ..perception.visibility_proposals import combat_field_readers, combat_visibility_titles, propose_combat_visibility
from ..vision.layout import normalize_game_screen
from .combat_evaluation import build_combat_reader



def diagnose_combat(collection, *, splits=('train', 'val'), state_contract=False):
    if not splits or len(set(splits)) != len(splits) or any(s not in ('train', 'val') for s in splits):
        raise ValueError('Diagnosis is restricted to unique train/val splits')
    collection = Path(collection)
    combat, metadata = build_combat_reader(collection)
    titles = combat_visibility_titles(combat_field_readers(combat))
    if state_contract:
        connect_optional_zero_readers(combat, titles, metadata)
    readers = {
        'attack': combat.attack_reader, 'range': combat.range_reader,
        'spread_degrees': combat.spread_reader, 'penetration_percent': combat.penetration_reader,
        'knockback_percent': combat.knockback_reader, 'grenades_remaining': combat.grenade_count_reader,
        'grenade_damage': combat.grenade_damage_reader, 'melee_damage': combat.melee_reader,
    }
    counts, samples = {}, []
    digest = hashlib.sha256()
    for path in annotation_paths(collection):
        raw = path.read_bytes()
        data = json.loads(raw)
        state = data.get('screen_state') or {}
        reviews = state.get('reviews') or {}
        split = data.get('split')
        if (split not in splits or state.get('screen_kind') != 'gameplay'
                or reviews.get('screen_kind') is not True or reviews.get('combat_stats') is not True):
            continue
        image_path = (path.parent / data.get('image', '')).resolve()
        if image_path.parent != (collection / 'originals').resolve():
            raise ValueError(f'{path}: invalid image path')
        image_raw = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(image_raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = normalize_game_screen(image, data.get('game_crop'))
        frame = PreparedFrame(image, normalized=True)
        expected = visual_combat_stats(state, legacy=False)
        canonical = state.get('combat_visibility') or {}
        proposals = propose_combat_visibility(image, {}, titles)
        digest.update(path.name.encode() + b'\0' + raw + hashlib.sha256(image_raw).digest())
        fields = {}
        for name, reader in readers.items():
            result = read_frame(reader, frame)
            names = ('range_min', 'range_max') if name == 'range' else (name,)
            for field in names:
                value = (getattr(result.value, field) if isinstance(result.value, RangeReading)
                         else result.value) if result.is_accepted else None
                visibility = canonical.get(field, proposals[field])
                numeric_reader = getattr(reader, 'reader', reader)
                reference_counts = {kind: len(getattr(numeric_reader, kind, ())) for kind in (
                    'examples', 'zero_examples', 'numeric_examples', 'dim_numeric_examples', 'titles', 'integers')}
                references = sum(reference_counts.values())
                target = expected[field]
                rule_zero = (state_contract and field in ZERO_WHEN_NOT_DISPLAYED
                             and canonical.get(field) == 'not_displayed'
                             and all(canonical.get(primary) == 'visible'
                                     for primary in ('attack', 'range_min', 'range_max', 'spread_degrees')))
                state_target = 0 if rule_zero else target
                correct = value == target if result.is_accepted and target is not None else None
                category = ('accepted' if result.is_accepted else
                            'occluded' if visibility == 'occluded' else
                            'not_displayed_requires_state_evidence' if visibility == 'not_displayed' else
                            'reference_unavailable' if references == 0 else
                            'visible_numeric_unmatched' if visibility in ('visible', 'unreadable') else
                            'visibility_or_layout_unknown')
                detail = {'status': result.status.value, 'predicted': value,
                          'expected_visual': target, 'correct': correct, 'source': result.source,
                          'expected_state': state_target,
                          'state_target_source': ZERO_RULE_SOURCE if rule_zero else 'visible_reviewed' if target is not None else None,
                          'state_correct': value == state_target if result.is_accepted and state_target is not None else None,
                          'reason': result.reason, 'confidence': result.confidence,
                          'visibility': visibility, 'visibility_source': 'annotation' if field in canonical else 'proposal',
                          'category': category, 'reference_counts': reference_counts}
                fields[field] = detail
                count = counts.setdefault(f'{split}:{field}', Counter())
                count['samples'] += 1
                count[result.status.value] += 1
                count[category] += int(category != 'accepted')
                if correct is not None:
                    count['compared'] += 1
                    count['correct' if correct else 'incorrect'] += 1
                elif result.is_accepted:
                    count['accepted_without_visual_target'] += 1
                if result.is_accepted and result.source.endswith(':clean-panel-zero-v1'):
                    count['rule_derived_zero'] += 1
                if detail['state_correct'] is not None:
                    count['state_compared'] += 1
                    count['state_correct' if detail['state_correct'] else 'state_incorrect'] += 1
        # This factory is deliberately stateless; repeated reads cannot seed
        # live grenade memory or change annotations.
        composed = read_frame(combat, frame)
        complete_state = all(detail['expected_state'] is not None for detail in fields.values())
        state_correct = (all(detail['state_correct'] is True for detail in fields.values())
                         if composed.is_accepted and complete_state else None)
        samples.append({'complete_state_target': complete_state, 'state_correct': state_correct, 'annotation': path.relative_to(collection).as_posix(), 'split': split,
                        'fields': fields, 'composed_status': composed.status.value,
                        'composed_reason': composed.reason,
                        'composed_values': asdict(composed.value) if composed.is_accepted else None})
    return {'schema': 'combat-diagnostics-v2', 'mode': 'state_contract' if state_contract else 'visual_numeric', 'splits': list(splits),
            'dataset_sha256': digest.hexdigest(), 'reference_metadata': metadata,
            'support_contract': combat_state_contract(), 'target_policy': 'explicit_visible_reviewed_only', 'counts': {k: dict(v) for k, v in sorted(counts.items())},
            'composed': {s: {'samples': sum(row['split'] == s for row in samples),
                              'accepted': sum(row['split'] == s and row['composed_status'] == 'accepted' for row in samples),
                              'complete_state_targets': sum(row['split'] == s and row['complete_state_target'] for row in samples),
                              'state_correct': sum(row['split'] == s and row['state_correct'] is True for row in samples),
                              'state_incorrect': sum(row['split'] == s and row['state_correct'] is False for row in samples)}
                         for s in splits}, 'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=Path('data/collection'))
    parser.add_argument('--state-contract', action='store_true', help='Apply confirmed optional-zero rules; grenade damage stays stateless')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=('train', 'val'), default=['train', 'val'])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new report path')
    report = diagnose_combat(args.collection, splits=tuple(args.splits), state_contract=args.state_contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'composed': report['composed'], 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
