"""Reference statistics are calculated once without changing match outcomes."""

import importlib
import unittest
from unittest.mock import patch

import numpy as np

from shotgun_king_rl.perception.royals import RoyalTemplates
from shotgun_king_rl.perception.types import RangeReading


class CountingTemplate(np.ndarray):
    std_calls = 0

    def std(self, *args, **kwargs):
        type(self).std_calls += 1
        return super().std(*args, **kwargs)


class ReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.query = np.random.default_rng(22).integers(0, 255, (45, 45), dtype=np.uint8)
        CountingTemplate.std_calls = 0

    def test_hud_readers_cache_stats_and_own_reference_copies(self):
        for name, class_name, patch_name in (
            ("attack", "AttackReader", "attack_patch"),
            ("range", "RangeReader", "range_patch"),
            ("spread", "SpreadReader", "spread_patch"),
            ("knockback", "KnockbackReader", "knockback_patch"),
            ("penetration", "PenetrationReader", "penetration_patch"),
            ("grenade_damage", "GrenadeDamageReader", "grenade_damage_patch"),
            ("grenade_count", "GrenadeCountReader", "grenade_count_patch"),
        ):
            with self.subTest(reader=name):
                module = importlib.import_module(f"shotgun_king_rl.perception.{name}")
                template = self.query.copy().view(CountingTemplate)
                blank = np.zeros_like(self.query).view(CountingTemplate)
                value = RangeReading(3, 5) if name == "range" else 4
                before = CountingTemplate.std_calls
                reader = getattr(module, class_name)(((value, template), (value, blank)))
                self.assertEqual(CountingTemplate.std_calls - before, 2)
                template[:] = 0  # Caller mutations must not change copied references.
                with patch.object(module, patch_name, return_value=self.query):
                    results = [reader.read(np.zeros((720, 1280, 3), dtype=np.uint8)) for _ in range(2)]
                self.assertEqual(CountingTemplate.std_calls - before, 2)
                self.assertTrue(all(result.is_accepted and result.value == value for result in results))
                self.assertEqual(results[0].confidence, results[1].confidence)

    def test_royals_cache_stats_but_keep_conflicting_references_ambiguous(self):
        first = self.query.astype(np.float32).view(CountingTemplate)
        second = first.copy()
        reader = RoyalTemplates(((8, first), (9, second)))
        self.assertEqual(CountingTemplate.std_calls, 2)
        first[:] = 0
        for _ in range(2):
            self.assertEqual(reader.classify(self.query), 6)
        self.assertEqual(CountingTemplate.std_calls, 2)
