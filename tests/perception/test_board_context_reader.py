"""Context snapshots must never turn absent or contradictory evidence into state."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from shotgun_king_rl.perception import (AmmoReading, BoardContextReader, BoardContextReading,
                                        PerceptionPipeline, Prediction, ScreenKind)
from tests.support.perception import FakeReader, VALID_COMBAT_STATS
from shotgun_king_rl.perception.frame import PreparedFrame
from shotgun_king_rl.runtime.perception import build_live_pipeline


class BoardContextReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'annotations').mkdir()
        (self.root / 'originals').mkdir()
        self.image = np.random.default_rng(3).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        board = [['empty'] * 8 for _ in range(8)]
        board[7][3] = 'player_king'
        sides = [[None] * 8 for _ in range(8)]
        sides[7][3] = 'ally'
        self.data = {'split': 'train', 'confirmed': True, 'piece_review': {'reviewed': True},
                     'screen_state': {'screen_kind': 'gameplay', 'reviews': {'screen_kind': True}},
                     'piece_sides': sides, 'board': board, 'moat_active': False,
                     'image': '../originals/run-01__screen.png', 'game_crop': [0, 0, 1280, 720]}
        cv2.imwrite(str(self.root / 'originals/run-01__screen.png'), self.image)
        self.path = self.root / 'annotations/run-01__screen.json'
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data))

    def test_accepts_both_states_and_returns_owned_arrays(self):
        for active in (False, True):
            self.data['moat_active'] = active
            self.save()
            reader = BoardContextReader.from_collection(self.root)
            result = reader.read(self.image)
            self.assertTrue(result.is_accepted)
            self.assertIs(result.value.moat_active, active)
            result.value.piece_sides[:] = 2
            self.assertEqual(reader.read(self.image).value.piece_sides[7, 3], 1)
            meta = reader.metadata()
            self.assertEqual(meta['references'], 1)
            self.assertEqual(len(meta['provenance'][0]['image_sha256']), 64)

    def test_never_loads_held_out_unreviewed_or_inferred_context(self):
        mutations = [lambda a: a.update(split='val'), lambda a: a.update(split='test'),
                     lambda a: a.update(confirmed=False), lambda a: a.pop('piece_sides'),
                     lambda a: a.pop('moat_active'), lambda a: a.update(moat_active=0),
                     lambda a: a.update(moat_active=None), lambda a: a.update(exclude_cells=[[0, 0]]),
                     lambda a: a.update(piece_review={'reviewed': False}),
                     lambda a: a['screen_state'].update(reviews={'screen_kind': False}),
                     lambda a: a['screen_state'].update(screen_kind='card_selection'),
                     lambda a: a['piece_sides'][7].__setitem__(3, 'unknown'),
                     lambda a: a['board'][0].__setitem__(0, 'ambiguous_royal'),
                     lambda a: a.update(image='../outside.png')]
        baseline = json.loads(self.path.read_text())
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.data = json.loads(json.dumps(baseline))
                mutate(self.data)
                self.save()
                self.assertEqual(BoardContextReader.from_collection(self.root).metadata()['references'], 0)

    def test_explicit_sides_load_without_inventing_missing_moat_state(self):
        self.data.pop('moat_active')
        self.save()
        reader = BoardContextReader.from_collection(self.root)
        self.assertEqual(reader.metadata()['side_reference_frames'], 1)
        self.assertEqual(reader.metadata()['moat_reference_frames'], 0)
        self.assertTrue(reader.side_reader.read_board(PreparedFrame(self.image).board_image).is_accepted)
        self.assertEqual(reader.read(self.image).reason, 'moat:references_unavailable')

    def test_changed_board_overlay_and_invalid_frame_do_not_reuse_state(self):
        reader = BoardContextReader.from_collection(self.root)
        self.assertTrue(reader.read(self.image).is_accepted)
        changed = self.image.copy()
        changed[250:350, 500:600] = 0
        self.assertFalse(reader.read(changed).is_accepted)
        self.assertFalse(reader.read(None).is_accepted)
        self.assertFalse(reader.read(np.zeros_like(self.image)).is_accepted)
        self.assertTrue(reader.read(self.image).is_accepted)

    def test_conflicting_same_image_is_unknown(self):
        self.data['moat_active'] = True
        (self.root / 'annotations/run-02__screen.json').write_text(json.dumps(self.data))
        result = BoardContextReader.from_collection(self.root).read(self.image)
        self.assertFalse(result.is_accepted)
        self.assertEqual(result.reason, 'moat:conflicting_references')

    def test_input_arrays_are_copied_and_prepared_path_matches_read(self):
        board = PreparedFrame(self.image).board_image
        sides = np.zeros((8, 8), dtype=np.int8)
        reader = BoardContextReader([(board, BoardContextReading(sides, True))])
        board[:] = 0
        sides[:] = 2
        result = reader.read_prepared(PreparedFrame(self.image))
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.value.piece_sides[0, 0], 0)

    def test_default_factory_constructs_reader_and_preserves_explicit_injection(self):
        pipeline, metadata = build_live_pipeline(self.root)
        self.assertIsInstance(pipeline.context_reader, BoardContextReader)
        self.assertEqual(metadata['readers']['board_context_reader']['references'], 1)
        injected = Mock()
        pipeline, metadata = build_live_pipeline(self.root, context_reader=injected)
        self.assertIs(pipeline.context_reader, injected)
        self.assertEqual(metadata['readers']['board_context_reader']['source'], 'injected')

    def test_empty_collection_has_reader_with_explicit_missing_reference_reason(self):
        self.path.unlink()
        pipeline, _ = build_live_pipeline(self.root)
        self.assertEqual(pipeline.context_reader.read(self.image).reason, 'references_unavailable')

    def test_pipeline_uses_context_and_blocks_a_changed_frame(self):
        board = np.zeros((8, 8), dtype=np.float32)
        board[7, 3] = 1
        pipeline = PerceptionPipeline(
            FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source='screen')),
            FakeReader(Prediction.accepted(board, source='board')),
            FakeReader(Prediction.accepted(AmmoReading(1, 4), source='ammo')),
            FakeReader(Prediction.accepted(VALID_COMBAT_STATS, source='combat')),
            BoardContextReader.from_collection(self.root))
        accepted = pipeline.observe(self.image)
        self.assertTrue(accepted.ready_for_policy)
        self.assertIs(accepted.board_context.moat_active, False)
        changed = self.image.copy()
        changed[250:350, 500:600] = 0
        rejected = pipeline.observe(changed)
        self.assertFalse(rejected.ready_for_policy)
        self.assertIsNone(rejected.board_context)
        self.assertTrue(any(issue.startswith('board_context:piece_sides:unmatched_cell:') for issue in rejected.issues))

    def test_accepted_controlled_king_identity_supplies_only_its_own_side(self):
        reader = BoardContextReader.from_collection(self.root)
        # Change the controlled king's appearance, leaving other cells and moat intact.
        board_image = PreparedFrame(self.image).board_image.copy()
        board_image[7*65:8*65, 3*65:4*65] = 0
        frame = Mock(board_image=board_image)
        board = np.zeros((8, 8), np.int32)
        board[7, 3] = 1
        self.assertFalse(reader.read_prepared(frame).is_accepted)
        result = reader.read_with_board(frame, board)
        self.assertTrue(result.is_accepted)
        self.assertEqual(result.value.piece_sides[7, 3], 1)
        self.assertEqual(board[7, 3], 1)
        board_image[:65, :65] = 0
        self.assertEqual(reader.read_with_board(frame, board).reason, 'piece_sides:unmatched_cell:0,0')

    def test_accepted_identity_cannot_overrule_conflicting_side_evidence(self):
        reader = BoardContextReader.from_collection(self.root)
        board = np.zeros((8, 8), np.int32)
        board[0, 0] = 1  # Reference explicitly labels this cell empty.
        result = reader.read_with_board(PreparedFrame(self.image), board)
        self.assertEqual(result.reason, 'piece_sides:identity_conflict:0,0')
        for invalid in (board.astype(bool), board + .5, np.full((8, 8), 7), np.ones((8, 8), int)):
            self.assertEqual(reader.read_with_board(PreparedFrame(self.image), invalid).reason,
                             'invalid_accepted_board')

    def test_rejected_piece_board_cannot_supply_controlled_king_identity(self):
        context = Mock()
        pipeline = PerceptionPipeline(
            FakeReader(Prediction.accepted(ScreenKind.GAMEPLAY, source='screen')),
            FakeReader(Prediction.rejected(None, source='board', reason='ambiguous_royal_kind')),
            FakeReader(Prediction.accepted(AmmoReading(1, 4), source='ammo')),
            FakeReader(Prediction.accepted(VALID_COMBAT_STATS, source='combat')), context)
        self.assertFalse(pipeline.observe(self.image).ready_for_policy)
        context.read_with_board.assert_not_called()
