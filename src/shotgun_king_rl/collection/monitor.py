"""Save changed, settled game screens as unapproved collection drafts."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from uuid import uuid4

import cv2
import numpy as np

from .dataset import choose_session_split
from .images import normalize
from .prepare import COLLECTION, draft_annotation


class ChangeDetector:
    """Compare settled thumbnails with the last successfully saved screen."""

    def __init__(self, change_threshold=1.0, stability_threshold=0.3, stable_samples=2):
        if not (math.isfinite(change_threshold) and math.isfinite(stability_threshold)
                and 0 <= stability_threshold < change_threshold):
            raise ValueError('Require 0 <= stability threshold < change threshold')
        if type(stable_samples) is not int or stable_samples < 2:
            raise ValueError('stable_samples must be at least 2')
        self.change_threshold = change_threshold
        self.stability_threshold = stability_threshold
        self.stable_samples = stable_samples
        self.saved = None
        self.reset_pending()

    def reset_pending(self):
        self.candidate = None
        self.samples = 0

    @staticmethod
    def thumbnail(screen):
        small = cv2.resize(screen, (320, 180), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    @staticmethod
    def difference(first, second):
        return float(np.mean(np.abs(first - second)))

    def consider(self, screen):
        current = self.thumbnail(screen)
        if self.saved is not None and self.difference(current, self.saved) < self.change_threshold:
            self.reset_pending()
            return False
        if self.candidate is None or self.difference(current, self.candidate) > self.stability_threshold:
            self.candidate = current
            self.samples = 1
            return False
        self.samples += 1
        return self.samples >= self.stable_samples

    def mark_saved(self, screen):
        self.saved = self.thumbnail(screen)
        self.reset_pending()


def image_digest(image):
    digest = hashlib.sha256(str(image.shape).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


class DraftRecorder:
    """Append images and drafts to one session without touching the inbox."""

    def __init__(self, collection, session_id, crop=None):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', session_id):
            raise ValueError('Invalid session ID')
        self.collection = Path(collection)
        self.session_id = session_id
        self.crop = crop
        self.session = self.collection / 'sessions' / session_id
        self.split = choose_session_split(self.collection, session_id, 1)
        self.seen = set()
        for path in (self.session / 'originals').glob('*'):
            if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            image = cv2.imread(str(path))
            if image is not None:
                self.seen.add(image_digest(image))

    def save(self, image):
        digest = image_digest(image)
        if digest in self.seen:
            return None
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_') + uuid4().hex[:8] + '.png'
        data = draft_annotation(image, name, self.collection, self.session_id, self.split, self.crop)
        # Automatic suggestions never carry human approvals, including future proposers.
        data['confirmed'] = False
        state = data.setdefault('screen_state', {})
        state.pop('reviewed', None)
        state['reviews'] = {key: False for key in ('screen_kind', 'ammo', 'combat_stats', 'cards', 'soul')}
        data.setdefault('prediction', {})['human_review_required'] = True
        data['auto_capture'] = {'method': 'settled-screen-change-v1', 'pixel_sha256': digest,
                                'captured_at': datetime.now(timezone.utc).isoformat()}
        encoded_ok, encoded = cv2.imencode('.png', image)
        if not encoded_ok:
            raise OSError('Could not encode capture')
        for directory in ('originals', 'annotations', 'review'):
            (self.session / directory).mkdir(parents=True, exist_ok=True)
        original = self.session / 'originals' / name
        annotation = self.session / 'annotations' / (name + '.json')
        with original.open('xb') as output:
            output.write(encoded.tobytes())
        annotation_created = False
        try:
            with annotation.open('x', encoding='utf-8') as output:
                annotation_created = True
                output.write(json.dumps(data, indent=2) + '\n')
        except BaseException:
            if annotation_created:
                annotation.unlink()
            original.unlink()
            raise
        self.seen.add(digest)
        return annotation


def monitor(capture, recorder, *, interval=0.5, duration=3600, max_images=500,
            change_threshold=1.0, stability_threshold=0.3, stable_samples=2,
            clock=time.monotonic, sleep=time.sleep, report=print):
    """Poll via an injected capture function; never click, train or approve."""
    if not math.isfinite(interval) or interval <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Interval and duration must be finite and positive')
    if type(max_images) is not int or max_images < 1:
        raise ValueError('max_images must be positive')
    detector = ChangeDetector(change_threshold, stability_threshold, stable_samples)
    counts = dict(polls=0, saved=0, duplicates=0, capture_failures=0, unreadable_screens=0)
    start = clock()
    while clock() - start < duration and counts['saved'] < max_images:
        image = capture()
        counts['polls'] += 1
        if image is None:
            counts['capture_failures'] += 1
            detector.reset_pending()
        else:
            try:
                screen = normalize(image, recorder.crop)
            except ValueError as exc:
                counts['unreadable_screens'] += 1
                detector.reset_pending()
                report(f'Skipped screen: {exc}')
            else:
                if detector.consider(screen):
                    # Persist only after both stability and change checks pass.
                    annotation = recorder.save(image)
                    detector.mark_saved(screen)
                    if annotation is None:
                        counts['duplicates'] += 1
                    else:
                        counts['saved'] += 1
                        report(f'Draft saved: {annotation}')
        if counts['saved'] < max_images:
            sleep(interval)
    return counts


def select_backend(backend, environ=None):
    environ = os.environ if environ is None else environ
    session = environ.get('XDG_SESSION_TYPE', '').lower()
    wayland = session == 'wayland' or (not session and bool(environ.get('WAYLAND_DISPLAY')))
    if backend == 'auto':
        return 'portal' if wayland else 'x11'
    if backend == 'x11' and wayland:
        raise ValueError('X11 capture is unavailable in a Wayland session; use --capture-backend portal')
    if backend not in ('x11', 'portal'):
        raise ValueError('Unknown capture backend')
    return backend


@contextmanager
def capture_source(backend, window_title, portal_python):
    if backend == 'portal':
        from .portal import PortalCapture
        print('Select the Shotgun King application window in the screen sharing dialog. '
              'The selected window is the capture source; --window-title applies only to X11.', flush=True)
        with PortalCapture(portal_python) as capture:
            yield capture
    else:
        from capture import capture_screen
        with tempfile.TemporaryDirectory(prefix='shotgun-monitor-') as temporary:
            path = str(Path(temporary) / 'latest.png')

            def capture():
                result = capture_screen(output_path=path, window_title=window_title, activate_window=False)
                return cv2.imread(path) if result else None

            yield capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, help='One ID for the entire play session')
    parser.add_argument('--collection', type=Path, default=COLLECTION)
    parser.add_argument('--window-title', default='Shotgun King')
    parser.add_argument('--capture-backend', choices=('auto', 'x11', 'portal'), default='auto')
    parser.add_argument('--portal-python', default='/usr/bin/python3', help='System Python with Gio/GStreamer bindings')
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--duration', type=float, default=3600, help='Maximum seconds to monitor')
    parser.add_argument('--max-images', type=int, default=500)
    parser.add_argument('--change-threshold', type=float, default=1.0, help='Mean thumbnail grayscale change, 0..255')
    parser.add_argument('--stability-threshold', type=float, default=0.3)
    parser.add_argument('--stable-samples', type=int, default=2)
    parser.add_argument('--crop', type=int, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    args = parser.parse_args()
    if not args.window_title.strip():
        parser.error('A nonempty game window title is required')
    try:
        backend = select_backend(args.capture_backend)
        # Validate limits before opening the permission dialog.
        ChangeDetector(args.change_threshold, args.stability_threshold, args.stable_samples)
        if not (math.isfinite(args.interval) and args.interval > 0
                and math.isfinite(args.duration) and args.duration > 0 and args.max_images > 0):
            raise ValueError('Interval, duration and max-images must be positive and finite')
        recorder = DraftRecorder(args.collection, args.session, args.crop)
        print(f'Monitoring with {backend}; split={recorder.split}; Ctrl+C stops capture. Drafts require review.', flush=True)
        with capture_source(backend, args.window_title, args.portal_python) as capture:
            counts = monitor(capture, recorder, interval=args.interval, duration=args.duration,
                             max_images=args.max_images, change_threshold=args.change_threshold,
                             stability_threshold=args.stability_threshold, stable_samples=args.stable_samples)
        print(json.dumps(counts))
    except KeyboardInterrupt:
        print('\nMonitoring stopped; saved drafts remain unapproved.')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Monitoring failed: {exc}\n')


if __name__ == '__main__':
    main()
