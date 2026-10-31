"""Optional combat proposals preserve unknowns and exclude the current source."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from shotgun_king_rl.collection.optional_combat import propose_optional_combat, melee_title_reference
from shotgun_king_rl.perception.types import Prediction
from shotgun_king_rl.perception.knockback import KnockbackReader
from shotgun_king_rl.perception.penetration import PenetrationReader
from shotgun_king_rl.perception.melee import MeleeDamageReader


class OptionalCombatProposalTests(unittest.TestCase):
    def test_accepted_zero_is_preserved_and_unknown_is_not_defaulted(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('shotgun_king_rl.collection.optional_combat.PenetrationReader.from_collection') as penetration, patch('shotgun_king_rl.collection.optional_combat.KnockbackReader.from_collection') as knockback, patch('shotgun_king_rl.collection.optional_combat.MeleeDamageReader.from_collection') as melee:
                penetration.return_value.read.return_value = Prediction.accepted(0, source='zero')
                knockback.return_value.read.return_value = Prediction.accepted(50, source='positive')
                melee.return_value.read.return_value = Prediction.unknown(source='missing', reason='unseen')
                exclude = Path(directory) / 'current.json'
                values, evidence = propose_optional_combat(np.zeros((720, 1280, 3), np.uint8), directory, exclude)
                self.assertEqual(values, {'penetration_percent': 0, 'knockback_percent': 50, 'melee_damage': None})
                self.assertEqual(evidence['melee_damage']['reason'], 'unseen')
                penetration.assert_called_once_with(directory, exclude=exclude)

    def test_readers_skip_excluded_annotation_before_loading_it(self):
        for cls in (KnockbackReader, PenetrationReader, MeleeDamageReader):
            with self.subTest(reader=cls.__name__), tempfile.TemporaryDirectory() as directory:
                annotation = Path(directory) / 'sessions/ref/annotations/screen.json'
                annotation.parent.mkdir(parents=True)
                annotation.write_text('invalid json is intentionally excluded')
                cls.from_collection(directory, exclude=annotation)

    def test_title_manifest_rejects_escape_and_unreviewed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'title_references.json'
            self.assertIsNone(melee_title_reference(root))
            manifest.write_text(json.dumps({'melee_damage': '../outside.json'}))
            with self.assertRaises(ValueError):
                melee_title_reference(root)
            source = root / 'screen.json'
            source.write_text(json.dumps({'split': 'test'}))
            manifest.write_text(json.dumps({'melee_damage': 'screen.json'}))
            with self.assertRaises(ValueError):
                melee_title_reference(root)
            self.assertIsNone(melee_title_reference(root, exclude=source))
