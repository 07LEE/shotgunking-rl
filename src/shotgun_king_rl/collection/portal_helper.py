"""System-Python helper for a consented window-only PipeWire stream.

Protocol: one-byte kind, four-byte big-endian size, payload. The parent sends
G for one fresh PNG or Q to close. Importing this module never opens a portal.
"""
import argparse
import json
import os
import struct
import sys
import threading
from uuid import uuid4

BUS_NAME = 'org.freedesktop.portal.Desktop'
DESKTOP_PATH = '/org/freedesktop/portal/desktop'
SCREENCAST = 'org.freedesktop.portal.ScreenCast'
REQUEST = 'org.freedesktop.portal.Request'
SESSION = 'org.freedesktop.portal.Session'
ELEMENTS = ('pipewiresrc', 'videorate', 'videoconvert', 'pngenc', 'appsink')


def send(kind, payload=b''):
    sys.stdout.buffer.write(kind + struct.pack('!I', len(payload)) + payload)
    sys.stdout.buffer.flush()


def libraries():
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gio, GLib, Gst
    except (ImportError, ValueError) as exc:
        raise RuntimeError('System Python needs python3-gi and gir1.2-gstreamer-1.0') from exc
    Gst.init(None)
    missing = [name for name in ELEMENTS if not Gst.ElementFactory.find(name)]
    if missing:
        raise RuntimeError('Missing GStreamer elements: ' + ', '.join(missing)
                           + '; install gstreamer1.0-pipewire and gstreamer1.0-plugins-good')
    return Gio, GLib, Gst


class PortalSession:
    def __init__(self, Gio, GLib):
        self.Gio, self.GLib = Gio, GLib
        self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self.session = None
        self.subscription = None
        self.closed = False
        self.active_loop = None

    def call(self, interface, method, parameters, path=DESKTOP_PATH):
        return self.bus.call_sync(BUS_NAME, path, interface, method, parameters,
                                  None, self.Gio.DBusCallFlags.NONE, 10000, None)

    def request(self, method, prefix_signature, prefix, options):
        token = 'shotgun_' + uuid4().hex
        options = {**options, 'handle_token': self.GLib.Variant('s', token)}
        sender = self.bus.get_unique_name()[1:].replace('.', '_')
        path = f'{DESKTOP_PATH}/request/{sender}/{token}'
        loop = self.GLib.MainLoop()
        self.active_loop = loop
        responses = []

        def response(_bus, _sender, _path, _interface, _signal, parameters, _data):
            responses.append(parameters.unpack())
            loop.quit()

        subscription = self.bus.signal_subscribe(
            BUS_NAME, REQUEST, 'Response', path, None,
            self.Gio.DBusSignalFlags.NONE, response, None,
        )
        timed_out = []

        def timeout():
            timed_out.append(True)
            loop.quit()
            return False

        timer = self.GLib.timeout_add_seconds(120, timeout)
        try:
            parameters = self.GLib.Variant('(' + prefix_signature + 'a{sv})', (*prefix, options))
            returned = self.call(SCREENCAST, method, parameters).unpack()[0]
            if returned != path:
                raise RuntimeError('Portal returned an unexpected request handle')
            if not responses and not self.closed:
                loop.run()
            if self.closed:
                raise RuntimeError('Screen sharing session closed')
            if not responses:
                raise RuntimeError('Screen sharing selection timed out')
            code, results = responses[0]
            if code != 0:
                raise RuntimeError('Screen sharing cancelled' if code == 1 else 'Screen sharing request failed')
            return results
        finally:
            if not timed_out:
                self.GLib.source_remove(timer)
            self.bus.signal_unsubscribe(subscription)
            self.active_loop = None

    def open(self):
        result = self.call('org.freedesktop.DBus.Properties', 'Get',
                           self.GLib.Variant('(ss)', (SCREENCAST, 'AvailableSourceTypes')))
        if not result.unpack()[0] & 2:
            raise RuntimeError('This portal does not support sharing individual windows')
        created = self.request('CreateSession', '', (), {
            'session_handle_token': self.GLib.Variant('s', 'shotgun_' + uuid4().hex),
        })
        self.session = created['session_handle']

        def closed(*_args):
            self.closed = True
            if self.active_loop is not None:
                self.active_loop.quit()

        self.subscription = self.bus.signal_subscribe(
            BUS_NAME, SESSION, 'Closed', self.session, None,
            self.Gio.DBusSignalFlags.NONE, closed, None,
        )
        self.request('SelectSources', 'o', (self.session,), {
            'types': self.GLib.Variant('u', 2),
            'multiple': self.GLib.Variant('b', False),
        })
        streams = self.request('Start', 'os', (self.session, ''), {})['streams']
        if len(streams) != 1 or streams[0][1].get('source_type', 2) != 2:
            raise RuntimeError('Select exactly one application window')
        result, descriptors = self.bus.call_with_unix_fd_list_sync(
            BUS_NAME, DESKTOP_PATH, SCREENCAST, 'OpenPipeWireRemote',
            self.GLib.Variant('(oa{sv})', (self.session, {})),
            self.GLib.VariantType.new('(h)'), self.Gio.DBusCallFlags.NONE,
            10000, None, None,
        )
        return descriptors.get(result.unpack()[0]), streams[0]

    def close(self):
        if self.subscription is not None:
            self.bus.signal_unsubscribe(self.subscription)
        if self.session and not self.closed:
            try:
                self.call(SESSION, 'Close', None, self.session)
            except Exception:
                pass


def stream(Gio, GLib, Gst):
    portal = PortalSession(Gio, GLib)
    pipeline = None
    descriptor = None
    try:
        descriptor, (node, properties) = portal.open()
        pipeline = Gst.parse_launch(
            'pipewiresrc name=source do-timestamp=true ! '
            'videorate drop-only=true max-rate=5 ! video/x-raw,framerate=5/1 ! '
            'videoconvert ! video/x-raw,format=RGB ! '
            'pngenc compression-level=1 ! appsink name=sink max-buffers=1 drop=true sync=false'
        )
        source = pipeline.get_by_name('source')
        source.set_property('fd', descriptor)
        if properties.get('pipewire-serial') is not None and source.find_property('target-object'):
            source.set_property('target-object', str(properties['pipewire-serial']))
        else:
            source.set_property('path', str(node))
        sink = pipeline.get_by_name('sink')
        loop = GLib.MainLoop()
        portal.active_loop = loop
        failures = []
        output_lock = threading.Lock()

        def fail(message):
            with output_lock:
                if not failures:
                    failures.append(message)
                    send(b'E', message.encode())
            GLib.idle_add(loop.quit)

        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def bus_message(_bus, message):
            if message.type == Gst.MessageType.ERROR:
                error, _debug = message.parse_error()
                fail(f'PipeWire capture failed: {error.message}')
            elif message.type == Gst.MessageType.EOS:
                fail('Screen sharing stream ended')

        bus.connect('message', bus_message)
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('Could not start PipeWire capture')
        send(b'R', json.dumps({'source_type': 'window'}).encode())

        def commands():
            try:
                while True:
                    command = sys.stdin.buffer.read(1)
                    if command in (b'', b'Q'):
                        GLib.idle_add(loop.quit)
                        return
                    if command != b'G':
                        raise RuntimeError('Invalid capture command')
                    sample = sink.emit('try-pull-sample', 5 * Gst.SECOND)
                    if sample is None:
                        raise RuntimeError('No new frame received from shared window')
                    buffer = sample.get_buffer()
                    payload = buffer.extract_dup(0, buffer.get_size())
                    with output_lock:
                        if failures:
                            return
                        send(b'F', payload)
            except Exception as exc:
                fail(str(exc))

        threading.Thread(target=commands, daemon=True).start()
        if not portal.closed:
            loop.run()
        if portal.closed:
            fail('Screen sharing session closed')
    finally:
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        portal.close()
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check dependencies without opening screen sharing')
    args = parser.parse_args()
    try:
        Gio, GLib, Gst = libraries()
        if args.check:
            print(json.dumps({'available': True, 'elements': list(ELEMENTS)}))
        else:
            stream(Gio, GLib, Gst)
    except Exception as exc:
        if args.check:
            print(str(exc), file=sys.stderr)
        else:
            send(b'E', str(exc).encode())
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
