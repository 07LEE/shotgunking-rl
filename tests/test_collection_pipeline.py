"""Offline ingestion, recovery and review-boundary checks."""

import json
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from collection_fixtures import game_screen, piece_annotation
from shotgun_king_rl.collection.locking import collection_lock
from shotgun_king_rl.collection.pipeline import CollectionPipeline, main
from shotgun_king_rl.collection.importer import import_images


def draft(image, name, collection, session, split, crop):
    data = piece_annotation()
    data.update(image='../originals/' + name, session_id=session, split=split, confirmed=False, game_crop=[0, 0, 1280, 720])
    data['screen_state'] = {'screen_kind': 'gameplay', 'reviews': {}}
    return data


class CollectionPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'inbox').mkdir()
        patcher = patch('shotgun_king_rl.collection.prepare.draft_annotation', side_effect=draft)
        self.predict = patcher.start()
        self.addCleanup(patcher.stop)

    def image(self, name='screen.png', changed=False):
        image = game_screen()
        if changed:
            image[0, 0] = [255, 255, 255]
        path = self.root / 'inbox' / name
        cv2.imwrite(str(path), image)
        return path

    def pipeline(self, **kwargs):
        return CollectionPipeline(self.root, 'play', **kwargs)

    def test_settled_files_only_and_new_arrivals_keep_split(self):
        first = self.image()
        pipeline = self.pipeline(settle=2)
        self.assertEqual(pipeline.tick(0)['pending'], [first.name])
        second = self.image('second.png', changed=True)
        result = pipeline.tick(2)
        self.assertEqual((result['created'], result['pending']), (1, [second.name]))
        self.assertTrue(second.exists())
        result = pipeline.tick(4)
        self.assertEqual(result['created'], 1)
        annotations = list((self.root / 'sessions/play/annotations').glob('*.json'))
        values = [json.loads(p.read_text()) for p in annotations]
        self.assertEqual(len({v['split'] for v in values}), 1)
        self.assertTrue(all(not v['confirmed'] for v in values))
        self.assertFalse((self.root / 'datasets').exists())
        report = Path(result['report'])
        self.assertTrue((report / 'audit.json').exists())
        self.assertTrue((report / 'backlog/backlog.json').exists())
        self.assertIsNone(pipeline.tick(5)['report'])

    def test_partial_image_and_valid_image_are_independent_and_retry_on_change(self):
        bad = self.root / 'inbox/partial.png'
        bad.write_bytes(b'not yet a PNG')
        self.image()
        pipeline = self.pipeline(settle=0)
        result = pipeline.tick(0)
        self.assertEqual(result['created'], 1)
        self.assertIn('inbox/partial.png', result['errors'])
        self.assertTrue(bad.exists())
        self.assertIsNone(self.pipeline(settle=0).tick(1)['report'])
        self.image('partial.png', changed=True)
        result = pipeline.tick(2)
        self.assertEqual(result['created'], 1)
        self.assertFalse(result['errors'])

    def test_changed_file_restarts_settling(self):
        self.image()
        pipeline = self.pipeline(settle=2)
        pipeline.tick(0)
        self.image(changed=True)
        self.assertEqual(pipeline.tick(2)['created'], 0)
        self.assertEqual(pipeline.tick(4)['created'], 1)

    def test_duplicate_is_archived_and_collision_is_not_overwritten(self):
        original = self.image()
        raw = original.read_bytes()
        pipeline = self.pipeline(settle=0)
        pipeline.tick(0)
        duplicate = self.root / 'inbox/copy.png'
        duplicate.write_bytes(raw)
        result = pipeline.tick(1)
        self.assertEqual((result['duplicates'], result['created']), (1, 0))
        archived = list((self.root / 'sessions/play/duplicates').glob('*'))
        self.assertEqual(archived[0].read_bytes(), raw)
        self.image(changed=True)
        result = pipeline.tick(2)
        self.assertIn('inbox/screen.png', result['errors'])
        self.assertEqual((self.root / 'sessions/play/originals/screen.png').read_bytes(), raw)
        self.assertTrue((self.root / 'inbox/screen.png').exists())

    def test_interrupted_import_recovery_preserves_reviewed_annotation(self):
        self.image()
        import_images(self.root, 'play')
        self.assertEqual(self.pipeline(settle=0).tick(0)['created'], 1)
        annotation = self.root / 'sessions/play/annotations/screen.png.json'
        data = json.loads(annotation.read_text())
        data['confirmed'] = True
        annotation.write_text(json.dumps(data))
        raw = annotation.read_bytes()
        self.predict.reset_mock()
        self.assertEqual(self.pipeline(settle=0).tick(1)['created'], 0)
        self.predict.assert_not_called()
        self.assertEqual(annotation.read_bytes(), raw)

    def test_failed_prediction_is_not_repeated_until_requested(self):
        self.image()
        self.predict.side_effect = ValueError('crop required')
        result = self.pipeline(settle=0).tick(0)
        self.assertIn('originals/screen.png', result['errors'])
        self.predict.reset_mock()
        self.pipeline(settle=0).tick(1)
        self.predict.assert_not_called()
        self.predict.side_effect = draft
        result = self.pipeline(settle=0, retry_errors=True).tick(2)
        self.assertEqual(result['created'], 1)
        self.assertFalse(result['errors'])

    def test_report_failure_retries_without_regenerating_annotations(self):
        self.image()
        pipeline = self.pipeline(settle=0)
        with patch.object(pipeline, 'export', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                pipeline.tick(0)
        self.predict.reset_mock()
        result = self.pipeline(settle=0).tick(1)
        self.assertIsNotNone(result['report'])
        self.assertEqual(result['created'], 0)
        self.predict.assert_not_called()

    def test_duplicate_move_interruption_recovers_one_archive_and_one_event(self):
        base = self.root
        for phase in ('before', 'after'):
            with self.subTest(phase=phase):
                self.root = base / phase
                (self.root / 'inbox').mkdir(parents=True)
                raw = self.image().read_bytes()
                self.pipeline(settle=0).tick(0)
                source = self.root / 'inbox/copy.png'
                source.write_bytes(raw)
                rename = Path.rename
                def interrupt(path, target):
                    if path == source:
                        if phase == 'after':
                            rename(path, target)
                        raise KeyboardInterrupt
                    return rename(path, target)
                with patch.object(Path, 'rename', interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        self.pipeline(settle=0).tick(1)
                state = json.loads((self.root / '.pipeline/play.json').read_text())
                self.assertFalse(state['events'])
                self.assertIn('pending_duplicate', state)
                result = self.pipeline(settle=0).tick(2)
                self.assertEqual(result['duplicates'], 1)
                self.assertFalse(source.exists())
                archived = list((self.root / 'sessions/play/duplicates').glob('*'))
                self.assertEqual(len(archived), 1)
                self.assertEqual(archived[0].read_bytes(), raw)
                report = json.loads((Path(result['report']) / 'pipeline.json').read_text())
                self.assertEqual(len(report['events']), 1)
                self.assertEqual(self.root / report['events'][0]['archive'], archived[0])
                self.assertIsNone(self.pipeline(settle=0).tick(3)['report'])

    def test_interrupted_duplicate_does_not_archive_replaced_inbox_contents(self):
        self.image()
        self.pipeline(settle=0).tick(0)
        source = self.image('copy.png')
        rename = Path.rename
        def interrupt(path, target):
            if path == source:
                raise KeyboardInterrupt
            return rename(path, target)
        with patch.object(Path, 'rename', interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.pipeline(settle=0).tick(1)
        replacement = self.image('copy.png', changed=True).read_bytes()
        result = self.pipeline(settle=0).tick(2)
        self.assertEqual((result['duplicates'], result['created']), (0, 1))
        self.assertEqual((self.root / 'sessions/play/originals/copy.png').read_bytes(), replacement)
        report = json.loads((Path(result['report']) / 'pipeline.json').read_text())
        self.assertFalse(report['events'])

    def test_report_publication_interruption_recovers_same_completed_report(self):
        base = self.root
        for phase in ('before', 'after'):
            with self.subTest(phase=phase):
                self.root = base / phase
                (self.root / 'inbox').mkdir(parents=True)
                self.image()
                rename = Path.rename
                def interrupt(path, target):
                    if path.name.endswith('.partial'):
                        if phase == 'after':
                            rename(path, target)
                        raise KeyboardInterrupt
                    return rename(path, target)
                with patch.object(Path, 'rename', interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        self.pipeline(settle=0).tick(0)
                state = json.loads((self.root / '.pipeline/play.json').read_text())
                report_id = state['report_run']
                parent = self.root / 'reports/pipeline/play'
                self.assertEqual(len(list(parent.glob('[!.]*'))), int(phase == 'after'))
                self.predict.reset_mock()
                result = self.pipeline(settle=0).tick(1)
                self.predict.assert_not_called()
                report = Path(result['report'])
                self.assertEqual(report.name, report_id)
                self.assertEqual(json.loads((report / 'pipeline.json').read_text())['status'], 'complete')
                self.assertTrue((report / 'audit.json').exists())
                self.assertTrue((report / 'backlog/backlog.json').exists())
                self.assertFalse(list(parent.glob('.*.partial')))
                self.assertEqual(len(list(parent.iterdir())), 1)
                self.assertIsNone(self.pipeline(settle=0).tick(2)['report'])

    def test_backlog_export_failure_keeps_only_partial_output_until_retry(self):
        self.image()
        with patch('shotgun_king_rl.collection.pipeline.build_backlog', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.pipeline(settle=0).tick(0)
        parent = self.root / 'reports/pipeline/play'
        partial = list(parent.iterdir())
        self.assertEqual(len(partial), 1)
        self.assertTrue(partial[0].name.endswith('.partial'))
        self.assertFalse((partial[0] / 'pipeline.json').exists())
        result = self.pipeline(settle=0).tick(1)
        self.assertFalse(partial[0].exists())
        self.assertEqual(len(list(parent.iterdir())), 1)
        self.assertEqual(json.loads((Path(result['report']) / 'pipeline.json').read_text())['status'], 'complete')

    def test_process_exit_after_move_or_publication_recovers_without_duplicate_events(self):
        base = self.root
        script = """
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path.cwd() / 'src'))
from shotgun_king_rl.collection.pipeline import CollectionPipeline
rename = Path.rename
def interrupt(path, target):
    result = rename(path, target)
    if (sys.argv[2] == 'archive' and Path(target).parent.name == 'duplicates') or (sys.argv[2] == 'report' and path.name.endswith('.partial')):
        os._exit(71)
    return result
with patch.object(Path, 'rename', interrupt):
    CollectionPipeline(Path(sys.argv[1]), 'play', settle=0).tick(1)
"""
        for phase in ('archive', 'report'):
            with self.subTest(phase=phase):
                self.root = base / phase
                (self.root / 'inbox').mkdir(parents=True)
                raw = self.image().read_bytes()
                self.pipeline(settle=0).tick(0)
                (self.root / 'inbox/copy.png').write_bytes(raw)
                child = subprocess.run([sys.executable, '-c', script, str(self.root), phase], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
                self.assertEqual(child.returncode, 71, child.stderr)
                self.predict.reset_mock()
                result = self.pipeline(settle=0).tick(2)
                self.predict.assert_not_called()
                report = json.loads((Path(result['report']) / 'pipeline.json').read_text())
                self.assertEqual(len(report['events']), 1)
                self.assertEqual(len(list((self.root / 'sessions/play/duplicates').glob('*'))), 1)
                self.assertFalse((self.root / 'inbox/copy.png').exists())
                self.assertIsNone(self.pipeline(settle=0).tick(3)['report'])

    def test_collection_lock_blocks_other_session_and_allows_nested_import(self):
        errors = []
        def concurrent():
            try:
                import_images(self.root, 'other-session')
            except ValueError as exc:
                errors.append(str(exc))
        with collection_lock(self.root):
            self.assertEqual(import_images(self.root, 'play', []), 0)
            thread = threading.Thread(target=concurrent)
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn('Another collection importer', errors[0])
        self.assertFalse((self.root / 'sessions/other-session').exists())

    def test_batch_finishes_and_leaves_later_arrivals_for_next_run(self):
        self.image()
        def settle(interval):
            self.image('later.png', changed=True)
        argv = ['pipeline', '--collection', str(self.root), '--session', 'play', '--settle', '0.001']
        with patch('sys.argv', argv), patch('builtins.print') as output, patch('shotgun_king_rl.collection.pipeline.time.sleep', side_effect=settle):
            main()
        result = json.loads(output.call_args.args[0])
        self.assertEqual((result['moved'], result['created']), (1, 1))
        self.assertEqual(result['session_screen_kinds'], {'gameplay': 1})
        self.assertTrue((self.root / 'inbox/later.png').exists())
        self.assertTrue(Path(result['report']).exists())

    def test_one_shot_reports_failure_exit_code(self):
        (self.root / 'inbox/broken.png').write_bytes(b'broken')
        argv = ['pipeline', '--collection', str(self.root), '--session', 'play', '--settle', '0']
        with patch('sys.argv', argv), patch('builtins.print'):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)

    def test_invalid_session_does_not_create_paths(self):
        with self.assertRaises(ValueError):
            CollectionPipeline(self.root, '../escape')


if __name__ == '__main__':
    unittest.main()
