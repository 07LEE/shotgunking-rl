"""Exercise the review workflow against synthetic data in an isolated browser."""

import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import unittest

import cv2

from shotgun_king_rl.review.server import make_server
from tests.support.collection import game_screen, piece_annotation

CHROME = shutil.which('google-chrome') or shutil.which('chromium') or shutil.which('chromium-browser')


@unittest.skipUnless(CHROME, 'Headless Chrome is required for the browser workflow check')
class ReviewBrowserTests(unittest.TestCase):
    def test_completed_sections_and_screens_have_explicit_edit_and_browse_actions(self):
        with tempfile.TemporaryDirectory(prefix='review-completion-') as temporary:
            root = Path(temporary)
            cases = [('play', 'a-complete', True), ('play', 'b-partial', False), ('play', 'c-pending', False), ('finished', 'done', True), ('missing-floor', 'no-floor', False), ('cards', 'card-screen', True)]
            for session, name, complete in cases:
                directory = root / 'sessions' / session
                (directory / 'originals').mkdir(parents=True, exist_ok=True)
                (directory / 'annotations').mkdir(exist_ok=True)
                cv2.imwrite(str(directory / 'originals' / (name + '.png')), game_screen())
                data = piece_annotation()
                data.update(image='../originals/' + name + '.png', session_id=session, floor=1, confirmed=complete, moat_active=False)
                reviews = {key: complete for key in ('screen_kind', 'ammo', 'combat_stats', 'soul')}
                if name == 'b-partial':
                    reviews.update(screen_kind=True, ammo=True)
                if name == 'no-floor':
                    reviews = {key: True for key in reviews}
                    data['floor'] = None
                data['screen_state'] = {'screen_kind': 'gameplay', 'ammo': {'loaded': 2, 'reserve': 6}, 'combat_stats': {'attack': 4}, 'souls': [], 'reviews': reviews}
                if name == 'card-screen':
                    data['screen_state']['screen_kind'] = 'card_selection'
                    data['confirmed'] = False
                (directory / 'annotations' / (name + '.png.json')).write_text(json.dumps(data))
            try:
                server = make_server(root, 0)
            except PermissionError:
                self.skipTest('Local HTTP sockets are unavailable in this sandbox')
            script = Path(__file__).with_name('fixtures').joinpath('review_completion.js').read_bytes()
            handler = server.RequestHandlerClass
            original = handler.reply

            def reply(request, body, kind='application/json', status=200):
                if request.path == '/review_collection.js':
                    body += script
                try:
                    return original(request, body, kind, status)
                except (BrokenPipeError, ConnectionResetError):
                    # Rapid navigation can cancel an outstanding image request.
                    pass

            handler.reply = reply
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                result = subprocess.run([CHROME, '--headless', '--no-sandbox', '--disable-gpu', '--no-proxy-server', '--disable-dev-shm-usage', '--disable-background-networking', '--no-first-run', f'--user-data-dir={root}/chrome', '--virtual-time-budget=16000', '--dump-dom', f'http://127.0.0.1:{server.server_port}/?session=play'], capture_output=True, text=True, timeout=40)
                match = re.search(r'<pre id="browser-test-result">(.*?)</pre>', result.stdout, re.S)
                self.assertIsNotNone(match, result.stderr[-1500:] + result.stdout[-2000:])
                report = json.loads(html.unescape(match.group(1)))
                self.assertTrue(report['ok'], json.dumps(report, indent=2))
                self.assertEqual(len(report['checks']), 25)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)
