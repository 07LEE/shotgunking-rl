"""Read consented Wayland window frames through a system-Python helper."""
import os
from pathlib import Path
import select
import struct
import subprocess
import time

import cv2
import numpy as np

MAX_PACKET = 64 * 1024 * 1024


def read_exact(pipe, size, deadline):
    chunks = []
    while size:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([pipe], [], [], max(0, remaining))[0]:
            raise OSError('Timed out waiting for screen sharing')
        chunk = os.read(pipe.fileno(), size)
        if not chunk:
            raise OSError('Screen sharing helper stopped')
        chunks.append(chunk)
        size -= len(chunk)
    return b''.join(chunks)


def read_packet(pipe, timeout):
    deadline = time.monotonic() + timeout
    header = read_exact(pipe, 5, deadline)
    kind, size = header[:1], struct.unpack('!I', header[1:])[0]
    if kind not in (b'R', b'F', b'E') or size > MAX_PACKET:
        raise OSError('Invalid screen sharing packet')
    payload = read_exact(pipe, size, deadline)
    if kind == b'E':
        raise OSError(payload.decode('utf-8', errors='replace'))
    return kind, payload


class PortalCapture:
    """Keep one window-sharing session open; each call requests a fresh frame."""

    def __init__(self, python='/usr/bin/python3'):
        self.python = python
        self.process = None

    def __enter__(self):
        try:
            self.process = subprocess.Popen(
                [self.python, '-u', str(Path(__file__).with_name('portal_helper.py'))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
            )
            kind, _payload = read_packet(self.process.stdout, 130)
            if kind != b'R':
                raise OSError('Screen sharing helper did not initialize')
            return self
        except BaseException:
            self.close()
            raise

    def __call__(self):
        if self.process is None or self.process.poll() is not None:
            raise OSError('Screen sharing is not running')
        self.process.stdin.write(b'G')
        kind, payload = read_packet(self.process.stdout, 10)
        if kind != b'F':
            raise OSError('Expected a screen sharing frame')
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError('Invalid shared-window image')
        return image

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    process.stdin.write(b'Q')
                    process.stdin.close()
                    process.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        finally:
            for pipe in (process.stdin, process.stdout):
                if pipe and not pipe.closed:
                    pipe.close()

    def __exit__(self, *_args):
        self.close()
