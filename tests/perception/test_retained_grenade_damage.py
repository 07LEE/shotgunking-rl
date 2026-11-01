"""Remember only observed grenade damage within an explicitly bounded run."""

import tempfile
import unittest

import numpy as np

from shotgun_king_rl.collection.combat_evaluation import build_combat_reader
from shotgun_king_rl.perception import AmmoReading, GrenadeDamageReader, PerceptionPipeline, Prediction, ScreenKind
from shotgun_king_rl.perception.grenade_damage import RetainedGrenadeDamageReader
from shotgun_king_rl.runtime.perception import build_live_pipeline
from tests.support.hud import CASES, screen
from tests.support.perception import FakeReader, NormalContextReader, make_reader


class DamageReader(FakeReader):
    title_visible = False

    def has_visible_title(self, image):
        return self.title_visible


class RetainedGrenadeDamageTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((720, 1280, 3), np.uint8)
        self.raw = DamageReader(Prediction.unknown(source='raw:test', reason='grenade_damage_not_found'))
        self.reader = RetainedGrenadeDamageReader(self.raw)

    def observe(self, value=2):
        self.raw.prediction = Prediction.accepted(value, source='raw:test', confidence=.99)
        return self.reader.read(self.image)

    def hide(self):
        self.raw.prediction = Prediction.unknown(source='raw:test', reason='grenade_damage_no_match')
        return self.reader.read(self.image)

    def test_cold_start_is_unknown_and_does_not_invent_zero(self):
        self.assertIsNone(self.reader.read(self.image).value)
        self.assertIsNone(self.reader.memory_info()['value'])

    def test_visible_value_is_retained_and_updated_with_provenance(self):
        self.assertEqual(self.observe().value, 2)
        self.assertFalse(self.reader.memory_info()['used_memory'])
        remembered = self.hide()
        self.assertEqual(remembered.value, 2)
        self.assertEqual(remembered.source, self.reader.source)
        self.assertTrue(self.reader.memory_info()['used_memory'])
        self.assertEqual(self.reader.memory_info()['observed_source'], 'raw:test')
        self.assertEqual(self.reader.memory_info()['observed_confidence'], .99)
        self.assertEqual(self.observe(3).value, 3)
        self.assertEqual(self.hide().value, 3)
        self.reader.reset_run()
        self.assertIsNone(self.hide().value)

    def test_unreadable_visible_value_invalidates_old_memory(self):
        self.observe()
        self.raw.title_visible = True
        self.assertIsNone(self.hide().value)
        self.raw.title_visible = False
        self.assertIsNone(self.hide().value)

    def test_ambiguity_and_rejection_cannot_be_hidden_by_memory(self):
        for result in (
            Prediction.unknown(source='raw:test', reason='grenade_damage_ambiguous'),
            Prediction.rejected(None, source='raw:test', reason='conflict'),
        ):
            self.observe()
            self.raw.prediction = result
            self.assertIsNone(self.reader.read(self.image).value)
            self.assertIsNone(self.hide().value)

    def test_invalid_image_cannot_return_memory(self):
        self.observe()
        self.raw.prediction = Prediction.unknown(source='raw:test', reason='invalid_image')
        self.assertIsNone(self.reader.read(None).value)
        self.assertFalse(self.reader.memory_info()['used_memory'])
        self.assertEqual(self.hide().value, 2)

    def test_invalid_accepted_values_do_not_poison_memory(self):
        for value in (-1, True, 2.5):
            self.observe()
            self.assertFalse(self.observe(value).is_accepted)
            self.assertIsNone(self.hide().value)

    def test_retaining_damage_does_not_reuse_other_combat_fields(self):
        combat = make_reader()
        combat.grenade_damage_reader = self.reader
        self.observe()
        self.hide()
        combat.attack_reader.prediction = Prediction.unknown(source='attack:test', reason='missing')
        result = combat.read(self.image)
        self.assertFalse(result.is_accepted)
        self.assertIn('attack:missing', result.reason)
        self.assertNotIn('grenade_damage:', result.reason)

    def test_card_selection_keeps_memory_but_retry_clears_it(self):
        combat = make_reader()
        combat.grenade_damage_reader = self.reader
        board = np.zeros((8, 8), dtype=np.int8)
        board[7, 3] = 1
        screen_reader = FakeReader(Prediction.accepted(ScreenKind.CARD_SELECTION, source='screen:test'))
        pipeline = PerceptionPipeline(screen_reader, FakeReader(Prediction.accepted(board, source='board:test')),
                                      FakeReader(Prediction.accepted(AmmoReading(1, 5), source='ammo:test')),
                                      combat, NormalContextReader(board))
        self.observe()
        pipeline.observe(self.image)
        self.assertEqual(pipeline.memory_info()['value'], 2)
        screen_reader.prediction = Prediction.accepted(ScreenKind.RETRY, source='screen:test')
        pipeline.observe(self.image)
        self.assertIsNone(pipeline.memory_info()['value'])
        self.assertIsNone(self.hide().value)

    def test_real_title_detector_prevents_reusing_an_unseen_visible_number(self):
        case = next(case for case in CASES if case.name == 'grenade_damage')
        image = screen(case)
        raw = GrenadeDamageReader([(2, case.patch(image))],
                                 titles=[('grenade_damage', case.patch(image)[:30])])
        subject = RetainedGrenadeDamageReader(raw)
        self.assertEqual(subject.read(image).value, 2)
        hidden = image.copy()
        hidden[450:650, 270:354] = 0
        self.assertEqual(subject.read(hidden).value, 2)
        self.assertIsNone(subject.read(screen(case, digits='8')).value)
        self.assertIsNone(subject.read(hidden).value)

    def test_live_factory_enables_memory_and_offline_factory_stays_stateless(self):
        with tempfile.TemporaryDirectory() as directory:
            offline, _ = build_combat_reader(directory)
            pipeline, metadata = build_live_pipeline(directory)
            self.assertIsInstance(offline.grenade_damage_reader, GrenadeDamageReader)
            self.assertIsInstance(pipeline.combat_stats_reader.grenade_damage_reader, RetainedGrenadeDamageReader)
            self.assertEqual(metadata['readers']['grenade_damage_reader']['memory_policy'], 'last_observed_in_current_run')
