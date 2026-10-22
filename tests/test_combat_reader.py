"""Offline aggregation checks; no collection fixtures or game input."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shotgun_king_rl.perception import (
    AmmoReading, CombatStatsReader, PerceptionPipeline, Prediction,
    PredictionStatus, RangeReading, ScreenKind,
)
from shotgun_king_rl.perception.frame import PreparedFrame


class StubReader:
    def __init__(self, value):
        self.prediction = Prediction.accepted(value, source="stub:test")
        self.frames = []

    def read_prepared(self, frame):
        self.frames.append(frame)
        return self.prediction


def make_reader():
    return CombatStatsReader(
        StubReader(4), StubReader(RangeReading(3, 5)), StubReader(65),
        StubReader(0), StubReader(0), StubReader(0), StubReader(2), StubReader(0),
    )


class CombatReaderTests(unittest.TestCase):
    def test_observed_zeros_and_damage_without_grenades_are_preserved(self):
        reader = make_reader()
        result = reader.read(object())
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.value.penetration_percent, 0)
        self.assertEqual(result.value.knockback_percent, 0)
        self.assertEqual(result.value.melee_damage, 0)
        self.assertEqual(result.value.grenades_remaining, 0)
        self.assertEqual(result.value.grenade_damage, 2)
        self.assertEqual((result.value.range_min, result.value.range_max), (3, 5))
        self.assertIsNone(result.confidence)

    def test_missing_melee_reader_blocks_the_pipeline(self):
        reader = replace(make_reader(), melee_reader=None)
        pipeline = PerceptionPipeline(
            StubReader(ScreenKind.GAMEPLAY), StubReader(np.zeros((8, 8))),
            StubReader(AmmoReading(1, 5)), reader,
        )
        result = pipeline.observe(object())
        self.assertFalse(result.ready_for_policy)
        self.assertIsNone(result.combat_stats)
        self.assertEqual(result.issues, (
            "combat_stats:melee_damage:reader_unavailable@unavailable",
        ))

    def test_unknown_each_field_is_never_filled_even_with_no_grenades(self):
        for field in make_reader().__dataclass_fields__:
            with self.subTest(field=field):
                reader = make_reader()
                getattr(reader, field).prediction = Prediction.unknown(
                    source="hud:test", reason="unreadable"
                )
                result = reader.read(object())
                self.assertEqual(result.status, PredictionStatus.UNKNOWN)
                self.assertIsNone(result.value)
                self.assertIn("unreadable@hud:test", result.reason)

    def test_rejection_survives_other_unknown_readings(self):
        reader = make_reader()
        reader.attack_reader.prediction = Prediction.rejected(
            4, source="attack:test", reason="ambiguous"
        )
        reader.melee_reader = None
        result = reader.read(object())
        self.assertEqual(result.status, PredictionStatus.REJECTED)
        self.assertIn("attack:ambiguous@attack:test", result.reason)
        self.assertIn("melee_damage:reader_unavailable", result.reason)
        self.assertIsNone(result.value)

    def test_invalid_accepted_values_are_rejected(self):
        cases = (
            ("attack_reader", None), ("range_reader", (3, 5)),
            ("range_reader", RangeReading(6, 5)),
            ("penetration_reader", 101), ("melee_reader", -1),
            ("melee_reader", True), ("melee_reader", float("nan")),
            ("grenade_count_reader", 0.5),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                reader = make_reader()
                getattr(reader, field).prediction = Prediction.accepted(value, source="bad:test")
                result = reader.read(object())
                self.assertFalse(result.is_accepted)
                self.assertIsNone(result.value)

    def test_same_prepared_frame_is_shared_and_values_do_not_leak(self):
        reader = make_reader()
        frame = PreparedFrame(object())
        self.assertTrue(reader.read_prepared(frame).is_accepted)
        for field in reader.__dataclass_fields__:
            self.assertIs(getattr(reader, field).frames[0], frame)
        reader.melee_reader.prediction = Prediction.unknown(source="melee:test", reason="missing")
        self.assertFalse(reader.read(object()).is_accepted)

    def test_ordinary_image_reader_is_supported(self):
        image = object()

        class OrdinaryReader:
            def read(self, received):
                self.received = received
                return Prediction.accepted(0, source="ordinary:test")

        reader = make_reader()
        reader.melee_reader = OrdinaryReader()
        self.assertTrue(reader.read(image).is_accepted)
        self.assertIs(reader.melee_reader.received, image)


if __name__ == "__main__":
    unittest.main()
