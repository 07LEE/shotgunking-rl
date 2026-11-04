"""Automatic capture keeps recognition gaps and skips well-covered screens."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from shotgun_king_rl.collection.monitor import DraftRecorder, monitor
from shotgun_king_rl.collection.selection import GapSelector, pending_reasons, reviewed_coverage
from shotgun_king_rl.perception import Prediction, ScreenKind
from shotgun_king_rl.perception.types import RangeReading
from tests.support.collection import game_screen
from tests.support.perception import FakeReader

VISIBILITY = 'shotgun_king_rl.collection.selection.propose_combat_visibility'


def accepted(value):
    return FakeReader(Prediction.accepted(value, source='fake:test', confidence=1.0))


def unknown():
    return FakeReader(Prediction.unknown(source='fake:test', reason='no_match'))


def labels(**overrides):
    result = dict(attack='visible', range_min='visible', range_max='visible', spread_degrees='visible',
                  penetration_percent='not_displayed', knockback_percent='not_displayed',
                  grenades_remaining='unknown', grenade_damage='not_displayed', melee_damage='not_displayed')
    result.update(overrides)
    return result


COVERED = {'attack=3': 9, 'range_min=3': 9, 'range_max=5': 9, 'spread_degrees=50': 9, 'grenades_remaining=0': 9}


class GapSelectorTests(unittest.TestCase):
    def selector(self, *, screen=ScreenKind.GAMEPLAY, attack=3, spread=None, count=0, **options):
        readers = {'attack': accepted(attack) if attack is not None else unknown(),
                   'range': accepted(RangeReading(3, 5)),
                   'spread_degrees': accepted(50) if spread is None else spread,
                   'grenade_damage': unknown()}
        screen_reader = accepted(screen) if screen is not None else unknown()
        grenades = accepted(count) if count is not None else unknown()
        return GapSelector(screen_reader, readers, grenades, [], COVERED, **options)

    def test_covered_gameplay_and_non_gameplay_screens_are_skipped(self):
        with patch(VISIBILITY, return_value=labels()):
            self.assertEqual(self.selector()(game_screen()), [])
            self.assertEqual(self.selector(screen=ScreenKind.CARD_SELECTION)(game_screen()), [])

    def test_unknown_screen_unreadable_rows_and_unread_grenades_are_reasons(self):
        self.assertEqual(self.selector(screen=None)(game_screen()), ['screen:unknown'])
        with patch(VISIBILITY, return_value=labels(spread_degrees='unreadable', grenade_damage='unreadable')):
            reasons = self.selector(spread=unknown(), count=None)(game_screen())
        self.assertEqual(reasons, ['spread_degrees:unreadable', 'grenade_damage:unreadable',
                                   'grenades_remaining:unknown'])

    def test_unrecognized_primary_row_is_a_reason_but_optional_unknown_is_not(self):
        with patch(VISIBILITY, return_value=labels(attack='unknown', melee_damage='unknown')):
            self.assertEqual(self.selector()(game_screen()), ['attack:unknown'])

    def test_rarely_reviewed_values_are_reasons_and_floats_match_integer_labels(self):
        with patch(VISIBILITY, return_value=labels()):
            self.assertEqual(self.selector(attack=8)(game_screen()), ['attack=8:rare'])
            self.assertEqual(self.selector(attack=3.0)(game_screen()), [])

    def test_tooltip_occlusion_waits_for_a_clean_frame(self):
        with patch(VISIBILITY, return_value=labels(attack='occluded', spread_degrees='unreadable')):
            self.assertEqual(self.selector(attack=8)(game_screen()), [])

    def test_reason_cap_counts_pending_drafts_and_new_selections(self):
        with patch(VISIBILITY, return_value=labels()):
            selector = self.selector(attack=8, max_per_reason=2, pending={'attack=8:rare': 1})
            self.assertEqual(selector(game_screen()), ['attack=8:rare'])
            self.assertEqual(selector(game_screen()), [])
            self.assertEqual(selector.gaps(game_screen()), ['attack=8:rare'])

    def test_invalid_limits(self):
        for options in ({'min_examples': 0}, {'max_per_reason': 0}, {'max_per_reason': 1.5}):
            with self.assertRaises(ValueError):
                self.selector(**options)


class CoverageTests(unittest.TestCase):
    def write(self, root, name, data):
        path = Path(root) / 'sessions' / 's' / 'annotations' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_only_reviewed_train_values_count_and_only_unreviewed_drafts_are_pending(self):
        state = {'screen_kind': 'gameplay', 'reviews': {'combat_stats': True},
                 'combat_stats': {'attack': 3, 'melee_damage': 2},
                 'combat_visibility': {'attack': 'visible', 'melee_damage': 'not_displayed'}}
        with tempfile.TemporaryDirectory() as root:
            self.write(root, 'train.json', {'split': 'train', 'screen_state': state})
            self.write(root, 'test.json', {'split': 'test', 'screen_state': state})
            self.write(root, 'draft.json', {'split': 'train', 'auto_capture': {'reasons': ['attack=8:rare']},
                                             'screen_state': {'reviews': {'combat_stats': False}}})
            self.write(root, 'done.json', {'split': 'train', 'auto_capture': {'reasons': ['attack=9:rare']},
                                            'screen_state': {'reviews': {'combat_stats': True}}})
            coverage = reviewed_coverage(root)
            self.assertEqual(coverage['attack=3'], 1)
            self.assertNotIn('melee_damage=2', coverage)
            self.assertEqual(pending_reasons(root), {'attack=8:rare': 1})


class SelectiveMonitorTests(unittest.TestCase):
    def test_skipped_screens_become_the_baseline_and_saved_drafts_keep_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DraftRecorder(directory, 'session')
            a = game_screen()
            b = a.copy()
            b[120:250, 390:520] = 240
            selector = Mock(side_effect=[[], ['attack=8:rare']])
            ticks = [0.0]

            def sleep(seconds):
                ticks[0] += seconds

            frames = [a, a, a, b, b]
            result = monitor(Mock(side_effect=frames), recorder, duration=len(frames), interval=1,
                             selector=selector, clock=lambda: ticks[0], sleep=sleep, report=lambda _: None)
            self.assertEqual((result['saved'], result['skipped']), (1, 1))
            self.assertEqual(selector.call_count, 2)
            data = json.loads(next((recorder.session / 'annotations').glob('*.json')).read_text())
            self.assertEqual(data['auto_capture']['reasons'], ['attack=8:rare'])
            self.assertFalse(data['confirmed'])
            self.assertFalse(any(data['screen_state']['reviews'].values()))
            self.assertEqual(len(list((recorder.session / 'originals').glob('*.png'))), 1)
            self.assertEqual(Path(data['image']).name, next((recorder.session / 'originals').glob('*.png')).name)
            self.assertTrue(np.array_equal(selector.call_args_list[1].args[0].shape, (720, 1280, 3)))
