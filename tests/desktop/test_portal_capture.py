"""Portal and subprocess contracts without opening real screen sharing."""

import io
import os
import struct
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from shotgun_king_rl.collection import portal, portal_helper
from shotgun_king_rl.collection.monitor import capture_source, select_backend


def packet(kind, data):
    return kind + struct.pack('!I', len(data)) + data


def pipe_with(data):
    reader, writer = os.pipe()
    os.write(writer, data)
    os.close(writer)
    return os.fdopen(reader, 'rb', buffering=0)


class PortalPacketTests(unittest.TestCase):
    def test_partial_reads_are_reassembled(self):
        real_read = os.read
        with pipe_with(packet(b'R', b'ok')) as pipe, patch.object(portal.os, 'read', side_effect=lambda fd, size: real_read(fd, min(1, size))) as read:
            self.assertEqual(portal.read_packet(pipe, 1), (b'R', b'ok'))
            self.assertGreaterEqual(read.call_count, 2)

    def test_cancel_eof_oversize_and_invalid_type_fail(self):
        for data, message in [
            (packet(b'E', b'Screen sharing cancelled'), 'cancelled'),
            (b'F\x00', 'stopped'),
            (b'F' + struct.pack('!I', portal.MAX_PACKET + 1), 'Invalid'),
            (packet(b'X', b''), 'Invalid'),
        ]:
            with self.subTest(message=message), pipe_with(data) as pipe:
                with self.assertRaisesRegex(OSError, message):
                    portal.read_packet(pipe, 1)

    def test_timeout_is_bounded(self):
        reader, writer = os.pipe()
        try:
            with os.fdopen(reader, 'rb', buffering=0) as pipe:
                with self.assertRaisesRegex(OSError, 'Timed out'):
                    portal.read_packet(pipe, 0)
        finally:
            os.close(writer)

    def process(self, data):
        process = Mock()
        process.stdout = pipe_with(data)
        process.stdin = io.BytesIO()
        process.poll.return_value = None
        self.addCleanup(process.stdout.close)
        return process

    def test_session_requests_and_decodes_new_lossless_frame_then_closes(self):
        image = np.full((4, 6, 3), (25, 40, 90), dtype=np.uint8)
        ok, encoded = cv2.imencode('.png', image)
        self.assertTrue(ok)
        process = self.process(packet(b'R', b'{}') + packet(b'F', encoded.tobytes()))
        with patch.object(portal.subprocess, 'Popen', return_value=process) as spawn:
            with portal.PortalCapture('/system/python') as capture:
                np.testing.assert_array_equal(capture(), image)
                self.assertEqual(process.stdin.getvalue(), b'G')
                with self.assertRaisesRegex(OSError, 'stopped'):
                    capture()  # Never return the previous image after stream failure.
            self.assertTrue(process.stdin.closed)
            process.wait.assert_called_once_with(timeout=3)
            self.assertEqual(spawn.call_args.args[0][0], '/system/python')
            self.assertFalse(spawn.call_args.kwargs.get('shell', False))

    def test_cancelled_setup_and_invalid_frame_cleanup(self):
        process = self.process(packet(b'E', b'Screen sharing cancelled'))
        with patch.object(portal.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(OSError, 'cancelled'):
                with portal.PortalCapture():
                    self.fail('Cancelled capture must not start')
        self.assertTrue(process.stdin.closed)
        process = self.process(packet(b'R', b'{}') + packet(b'F', b'bad png'))
        with patch.object(portal.subprocess, 'Popen', return_value=process):
            with portal.PortalCapture() as capture:
                with self.assertRaisesRegex(OSError, 'Invalid shared-window'):
                    capture()

    def test_helper_termination_escalates_only_if_unresponsive(self):
        process = self.process(b'')
        process.wait.side_effect = [portal.subprocess.TimeoutExpired('helper', 3), None]
        capture = portal.PortalCapture()
        capture.process = process
        capture.close()
        process.terminate.assert_called_once()
        process.kill.assert_not_called()
        self.assertIsNone(capture.process)


class FakeVariant:
    def __init__(self, signature, value):
        self.signature, self.value = signature, value

    def unpack(self):
        return self.value


class PortalRequestTests(unittest.TestCase):
    def make_session(self, code=0):
        Gio, GLib, bus = Mock(), Mock(), Mock()
        Gio.bus_get_sync.return_value = bus
        GLib.Variant = FakeVariant
        bus.get_unique_name.return_value = ':1.55'
        events = []
        callback = []
        def subscribe(*args):
            events.append('subscribe')
            callback.append(args[6])
            return 1
        bus.signal_subscribe.side_effect = subscribe
        def call(*args):
            events.append('call')
            token = args[4].value[-1]['handle_token'].value
            path = f'{portal_helper.DESKTOP_PATH}/request/1_55/{token}'
            callback[-1](None, None, None, None, None, FakeVariant('', (code, {'session_handle': '/session'})), None)
            return FakeVariant('', (path,))
        bus.call_sync.side_effect = call
        return portal_helper.PortalSession(Gio, GLib), events

    def test_subscribe_before_method_and_handle_early_response(self):
        session, events = self.make_session()
        self.assertEqual(session.request('CreateSession', '', (), {})['session_handle'], '/session')
        self.assertEqual(events, ['subscribe', 'call'])
        session.GLib.MainLoop.return_value.run.assert_not_called()
        session.bus.signal_unsubscribe.assert_called_once()

    def test_cancelled_request_has_no_retry(self):
        session, events = self.make_session(code=1)
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            session.request('Start', 'os', ('/session', ''), {})
        self.assertEqual(events, ['subscribe', 'call'])
        session.bus.signal_unsubscribe.assert_called_once()

    def test_select_sources_is_window_only_single_source(self):
        session, _ = self.make_session()
        session.call = Mock(return_value=FakeVariant('', (2,)))
        session.request = Mock(side_effect=[{'session_handle': '/session'}, {}, {'streams': [(12, {'source_type': 2})]}])
        descriptors = Mock()
        descriptors.get.return_value = 99
        session.bus.call_with_unix_fd_list_sync.return_value = (FakeVariant('', (0,)), descriptors)
        fd, stream = session.open()
        self.assertEqual(fd, 99)
        self.assertEqual(stream[0], 12)
        options = session.request.call_args_list[1].args[3]
        self.assertEqual(options['types'].value, 2)
        self.assertFalse(options['multiple'].value)
        self.assertNotIn('persist_mode', options)

    def test_nonwindow_stream_and_missing_window_support_are_rejected(self):
        for streams in ([], [(1, {'source_type': 1})], [(1, {}), (2, {})]):
            session, _ = self.make_session()
            session.call = Mock(return_value=FakeVariant('', (2,)))
            session.request = Mock(side_effect=[{'session_handle': '/session'}, {}, {'streams': streams}])
            with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                session.open()
            session.bus.call_with_unix_fd_list_sync.assert_not_called()
        session, _ = self.make_session()
        session.call = Mock(return_value=FakeVariant('', (1,)))
        with self.assertRaisesRegex(RuntimeError, 'individual windows'):
            session.open()


class BackendSelectionTests(unittest.TestCase):
    def test_wayland_uses_portal_and_x11_remains_explicit(self):
        self.assertEqual(select_backend('auto', {'XDG_SESSION_TYPE': 'wayland'}), 'portal')
        self.assertEqual(select_backend('auto', {'WAYLAND_DISPLAY': 'wayland-0'}), 'portal')
        self.assertEqual(select_backend('auto', {'XDG_SESSION_TYPE': 'x11'}), 'x11')
        with self.assertRaisesRegex(ValueError, 'Wayland'):
            select_backend('x11', {'XDG_SESSION_TYPE': 'wayland'})

    def test_portal_context_uses_chosen_window_without_x11_lookup(self):
        capture = Mock()
        with patch.object(portal, 'PortalCapture') as constructor, patch('capture.find_window_geometry') as lookup:
            constructor.return_value.__enter__.return_value = capture
            with capture_source('portal', 'Shotgun King', '/system/python') as source:
                self.assertIs(source, capture)
            constructor.assert_called_once_with('/system/python')
            lookup.assert_not_called()
            constructor.return_value.__exit__.assert_called_once()
