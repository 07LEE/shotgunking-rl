"""Review startup and existing-server reuse without opening desktop apps."""

from contextlib import ExitStack
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from shotgun_king_rl.review import server


class ReviewLauncherTests(unittest.TestCase):
    def launch(self, args, running=False):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('sys.argv', ['review'] + args))
        stack.enter_context(patch('builtins.print'))
        existing = stack.enter_context(patch.object(server, 'running_review', return_value=running))
        browser = stack.enter_context(patch.object(server.webbrowser, 'open', return_value=True))
        factory = stack.enter_context(patch.object(server, 'make_server'))
        factory.return_value.server_port = 8765
        return existing, browser, factory

    def test_script_start_opens_browser_without_importing_inbox(self):
        existing, browser, factory = self.launch(['--open'])
        with patch.object(server, 'prepare_session') as prepare:
            server.main()
        prepare.assert_not_called()
        factory.assert_called_once_with(server.COLLECTION, 8765)
        browser.assert_called_once_with('http://127.0.0.1:8765')
        factory.return_value.serve_forever.assert_called_once()
        factory.return_value.server_close.assert_called_once()

    def test_existing_review_is_opened_without_another_server(self):
        existing, browser, factory = self.launch(['--open'], running=True)
        server.main()
        factory.assert_not_called()
        browser.assert_called_once_with('http://127.0.0.1:8765')

    def test_concurrent_start_reuses_the_server_that_won_the_port(self):
        existing, browser, factory = self.launch(['--open'])
        existing.side_effect = [False, True]
        factory.side_effect = OSError('address in use')
        server.main()
        browser.assert_called_once_with('http://127.0.0.1:8765')

    def test_unrelated_port_occupant_is_not_opened(self):
        existing, browser, factory = self.launch(['--open'])
        factory.side_effect = OSError('address in use')
        with self.assertRaises(SystemExit) as error:
            server.main()
        self.assertEqual(error.exception.code, 1)
        browser.assert_not_called()

    def test_prepare_only_keeps_existing_import_behavior_without_opening_page(self):
        existing, browser, factory = self.launch(['--session', 'play', '--prepare-only', '--open'])
        with patch.object(server, 'prepare_session', return_value={}) as prepare:
            server.main()
        prepare.assert_called_once_with(server.COLLECTION, 'play', None)
        existing.assert_not_called()
        factory.assert_not_called()
        browser.assert_not_called()

    def test_browser_failure_keeps_review_available_and_prints_url(self):
        existing, browser, factory = self.launch(['--open'])
        browser.return_value = False
        with patch('builtins.print') as output:
            server.main()
        self.assertTrue(any('Open http://127.0.0.1:8765' in str(call) for call in output.call_args_list))
        factory.return_value.serve_forever.assert_called_once()
        factory.return_value.server_close.assert_called_once()

    def test_identity_check_requires_the_same_app_and_collection(self):
        root = Path('/tmp/review-test-collection')
        identities = [
            (200, server.server_identity(root), True),
            (200, server.server_identity(Path('/tmp/other-collection')), False),
            (200, {'application': 'other'}, False),
            (404, server.server_identity(root), False),
            (200, None, False),
        ]
        for status, identity, expected in identities:
            with self.subTest(status=status, identity=identity), patch.object(server, 'HTTPConnection') as factory:
                response = factory.return_value.getresponse.return_value
                response.status = status
                response.read.return_value = json.dumps(identity).encode() if identity is not None else b'not json'
                self.assertEqual(server.running_review(root, 8765), expected)
                factory.return_value.close.assert_called_once()
        with patch.object(server, 'HTTPConnection') as factory:
            factory.return_value.request.side_effect = ConnectionRefusedError
            self.assertFalse(server.running_review(root, 8765))
            factory.return_value.close.assert_called_once()
        with patch.object(server, 'HTTPConnection') as factory:
            self.assertFalse(server.running_review(root, 0))
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
