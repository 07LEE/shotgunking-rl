"""Synthetic checks for passive, change-triggered draft collection."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import capture as desktop_capture
from shotgun_king_rl.collection.monitor import ChangeDetector, DraftRecorder, monitor
from shotgun_king_rl.collection.prepare import draft_annotation
from shotgun_king_rl.collection.dataset import build_dataset
from shotgun_king_rl.perception.royals import RoyalTemplates

from tests.support.collection import game_screen


class ChangeDetectorTests(unittest.TestCase):
    def setUp(self):
        self.a = np.full((720, 1280, 3), 50, dtype=np.uint8)
        self.b = np.full_like(self.a, 100)

    def test_first_and_changed_screen_require_stability(self):
        detector = ChangeDetector()
        self.assertFalse(detector.consider(self.a))
        self.assertTrue(detector.consider(self.a))
        detector.mark_saved(self.a)
        self.assertFalse(detector.consider(self.a))
        self.assertFalse(detector.consider(self.b))
        self.assertTrue(detector.consider(self.b))

    def test_animation_and_cumulative_drift_do_not_count_as_stable(self):
        detector = ChangeDetector(stability_threshold=1.1, change_threshold=3, stable_samples=3)
        for value in (50, 51, 52, 53, 54, 55):
            self.assertFalse(detector.consider(np.full_like(self.a, value)))
        detector.reset_pending()
        for image in (self.a, self.b, self.a, self.b):
            self.assertFalse(detector.consider(image))

    def test_small_noise_is_ignored(self):
        detector = ChangeDetector()
        detector.mark_saved(self.a)
        noise = self.a.copy()
        noise[10:12, 10:12] = 200
        self.assertFalse(detector.consider(noise))

    def test_invalid_thresholds_and_sample_count(self):
        for kwargs in ({'change_threshold': float('nan')}, {'stability_threshold': -1}, {'stable_samples': 1}, {'change_threshold': 0.1}):
            with self.assertRaises(ValueError):
                ChangeDetector(**kwargs)


class CollectionMonitorTests(unittest.TestCase):
    def run_monitor(self, frames, recorder, **kwargs):
        ticks = [0.0]
        capture = Mock(side_effect=frames)
        def sleep(seconds):
            ticks[0] += seconds
        return monitor(capture, recorder, duration=len(frames), interval=1,
                       clock=lambda: ticks[0], sleep=sleep, report=lambda _: None, **kwargs)

    def test_changed_settled_frames_only_and_duplicate_return(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DraftRecorder(directory, 'session')
            a = game_screen()
            b = a.copy()
            b[120:250, 390:520] = 240
            frames = [a, a, a, b, b, b, a, a]
            result = self.run_monitor(frames, recorder)
            self.assertEqual(result['saved'], 2)
            self.assertEqual(result['duplicates'], 1)
            annotations = list((recorder.session / 'annotations').glob('*.json'))
            self.assertEqual(len(annotations), 2)
            for path in annotations:
                data = json.loads(path.read_text())
                self.assertFalse(data['confirmed'])
                self.assertTrue(data['prediction']['human_review_required'])
                self.assertFalse(any(data['screen_state']['reviews'].values()))
                self.assertEqual(data['split'], recorder.split)
                self.assertTrue((path.parent / data['image']).exists())
            with self.assertRaisesRegex(ValueError, 'No usable annotations'):
                build_dataset(recorder.collection / 'sessions', Path(directory) / 'export')

    def test_capture_failure_resets_stability_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DraftRecorder(directory, 'session')
            a = game_screen()
            result = self.run_monitor([a, None, a], recorder)
            self.assertEqual(result['saved'], 0)
            self.assertEqual(result['capture_failures'], 1)
            self.assertFalse(recorder.session.exists())

    def test_resume_preserves_annotations_split_and_inbox(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DraftRecorder(directory, 'session')
            path = recorder.save(game_screen())
            data = json.loads(path.read_text())
            data['confirmed'] = True
            path.write_text(json.dumps(data))
            original_bytes = path.read_bytes()
            inbox = Path(directory) / 'inbox'
            inbox.mkdir()
            (inbox / 'untouched.png').write_bytes(b'inbox')
            resumed = DraftRecorder(directory, 'session')
            self.assertIsNone(resumed.save(game_screen()))
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(resumed.split, recorder.split)
            self.assertEqual((inbox / 'untouched.png').read_bytes(), b'inbox')

    def test_image_limit_stops_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DraftRecorder(directory, 'session')
            result = self.run_monitor([game_screen()] * 4, recorder, max_images=1)
            self.assertEqual(result['saved'], 1)
            self.assertEqual(result['polls'], 2)

    def test_unreadable_game_area_is_reported_without_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DraftRecorder(directory, 'session')
            result = self.run_monitor([np.zeros((100, 100, 3), dtype=np.uint8)] * 2, recorder)
            self.assertEqual(result['unreadable_screens'], 2)
            self.assertEqual(result['saved'], 0)

    def test_explicit_crop_is_used_for_detection_and_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.zeros((740, 1320, 3), dtype=np.uint8)
            image[10:730, 20:1300] = game_screen()
            recorder = DraftRecorder(directory, 'session', crop=[20, 10, 1300, 730])
            result = self.run_monitor([image, image], recorder)
            self.assertEqual(result['saved'], 1)
            data = json.loads(next((recorder.session / 'annotations').glob('*.json')).read_text())
            self.assertEqual(data['game_crop'], [20, 10, 1300, 730])

    def test_annotation_failure_does_not_leave_an_original(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = DraftRecorder(directory, 'session')
            with patch('shotgun_king_rl.collection.monitor.draft_annotation', side_effect=ValueError('failed')):
                with self.assertRaises(ValueError):
                    recorder.save(game_screen())
            self.assertFalse(recorder.session.exists())

    def test_invalid_session_or_loop_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            for session in ('../escape', '', '/absolute'):
                with self.assertRaises(ValueError):
                    DraftRecorder(directory, session)
            recorder = DraftRecorder(directory, 'session')
            for kwargs in ({'interval': 0}, {'duration': float('inf')}, {'max_images': 0}):
                with self.assertRaises(ValueError):
                    monitor(Mock(), recorder, **kwargs)

    def test_draft_preserves_distinct_and_ambiguous_royals(self):
        with tempfile.TemporaryDirectory() as directory:
            board = np.zeros((8, 8), dtype=int)
            board[0, :4] = (6, 6, 6, 7)
            rng = np.random.default_rng(42)
            queen, king, unseen = [rng.uniform(0, 255, (45, 45)).astype(np.float32) for _ in range(3)]
            reader = RoyalTemplates(((8, queen), (9, king)))
            with patch('shotgun_king_rl.collection.prepare.analyzer.get_state_matrix', return_value=board), patch(
                'shotgun_king_rl.collection.prepare.RoyalTemplates.from_collection'
            ) as templates, patch('shotgun_king_rl.perception.royals.analyzer._extract_signal', side_effect=(queen, king, unseen)):
                templates.return_value = reader
                result = draft_annotation(game_screen(), 'test.png', Path(directory), 'session', 'train')
            self.assertEqual(result['board'][0][:4], ['queen', 'white_king', 'ambiguous_royal', 'special_knight'])
            self.assertFalse(result['confirmed'])
            self.assertNotIn('provisionally mapped', result['annotation_source'])

    def test_missing_royal_references_leave_draft_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            board = np.zeros((8, 8), dtype=int)
            board[0, 0] = 6
            with patch('shotgun_king_rl.collection.prepare.analyzer.get_state_matrix', return_value=board):
                result = draft_annotation(game_screen(), 'test.png', Path(directory), 'session', 'train')
            self.assertEqual(result['board'][0][0], 'ambiguous_royal')
            self.assertFalse(result['confirmed'])

    def test_shared_draft_function_does_not_approve_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = draft_annotation(game_screen(), 'test.png', Path(directory), 'session', 'train')
            self.assertFalse(result['confirmed'])
            self.assertFalse(any(result['screen_state']['reviews'].values()))
            self.assertEqual(len(result['board']), 8)
            self.assertEqual(result['image'], '../originals/test.png')


class PassiveCaptureTests(unittest.TestCase):
    def test_window_lookup_can_leave_focus_unchanged(self):
        window = Mock()
        window.title = 'Shotgun King'
        window.box = Mock(left=10, top=20, width=1280, height=720)
        backend = Mock()
        backend.getAllWindows.return_value = [window]
        with patch.object(desktop_capture, 'pwc', backend):
            self.assertEqual(desktop_capture.find_window_geometry(activate=False)['left'], 10)
            window.activate.assert_not_called()
            desktop_capture.find_window_geometry()
            window.activate.assert_called_once()
