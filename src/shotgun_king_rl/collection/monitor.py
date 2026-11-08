"""Save changed, settled game screens as unapproved collection drafts."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import cv2
import numpy as np

from .dataset import choose_session_split
from .images import detect_crop, normalize
from ..collection_layout import annotations_dir, originals_dir, session_images, stored_name, validate_session_id
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
        self.status = 'waiting'
        self.change = None
        self.stability = None
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
        self.change = self.difference(current, self.saved) if self.saved is not None else None
        self.stability = self.difference(current, self.candidate) if self.candidate is not None else None
        if self.change is not None and self.change < self.change_threshold:
            self.status = 'below_change_threshold'
            self.reset_pending()
            return False
        if self.candidate is None or self.stability > self.stability_threshold:
            self.status = 'waiting_for_stability'
            self.candidate = current
            self.samples = 1
            return False
        self.samples += 1
        self.status = 'settled' if self.samples >= self.stable_samples else 'waiting_for_stability'
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
        validate_session_id(session_id)
        self.collection = Path(collection)
        self.session_id = session_id
        self.crop = crop
        self.split = choose_session_split(self.collection, session_id, 1)
        self.seen = set()
        for path in session_images(self.collection, session_id):
            image = cv2.imread(str(path))
            if image is not None:
                self.seen.add(image_digest(image))

    def save(self, image, reasons=None):
        source_crop = self.crop if self.crop is not None else detect_crop(image)
        # Validate before slicing, retaining the original viewport pixels.
        normalize(image, source_crop)
        left, top, right, bottom = source_crop
        image = image[top:bottom, left:right].copy()
        height, width = image.shape[:2]
        stored_crop = [0, 0, width, height]
        digest = image_digest(image)
        if digest in self.seen:
            return None
        name = stored_name(self.session_id, datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_') + uuid4().hex[:8] + '.png')
        data = draft_annotation(image, name, self.collection, self.session_id, self.split, stored_crop)
        # Automatic suggestions never carry human approvals, including future proposers.
        data['confirmed'] = False
        state = data.setdefault('screen_state', {})
        state.pop('reviewed', None)
        state['reviews'] = {key: False for key in ('screen_kind', 'ammo', 'combat_stats', 'cards', 'soul')}
        data.setdefault('prediction', {})['human_review_required'] = True
        data['auto_capture'] = {'method': 'settled-screen-change-v1', 'pixel_sha256': digest,
                                'captured_at': datetime.now(timezone.utc).isoformat(),
                                'source_crop': list(source_crop)}
        if reasons is not None:
            data['auto_capture']['reasons'] = list(reasons)
        encoded_ok, encoded = cv2.imencode('.png', image)
        if not encoded_ok:
            raise OSError('Could not encode capture')
        for directory in (originals_dir(self.collection), annotations_dir(self.collection)):
            directory.mkdir(parents=True, exist_ok=True)
        original = originals_dir(self.collection) / name
        annotation = annotations_dir(self.collection) / (name + '.json')
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


def monitor(capture, recorder, *, interval=0.2, duration=3600, max_images=500,
            change_threshold=1.0, stability_threshold=0.3, stable_samples=2, selector=None,
            clock=time.monotonic, sleep=time.sleep, report=print):
    """Poll via an injected capture function; never click, train or approve.

    A selector returns reasons for keeping a settled screen; without reasons
    the screen becomes the new baseline but is not written.
    """
    if not math.isfinite(interval) or interval <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Interval and duration must be finite and positive')
    if type(max_images) is not int or max_images < 1:
        raise ValueError('max_images must be positive')
    detector = ChangeDetector(change_threshold, stability_threshold, stable_samples)
    counts = dict(polls=0, saved=0, duplicates=0, skipped=0, capture_failures=0, unreadable_screens=0)
    start = clock()
    last_report = start
    previous_frame = None
    frame_delta = None
    raw_changes = 0
    status = 'waiting'
    crop_failure_reported = False
    while clock() - start < duration and counts['saved'] < max_images:
        poll_start = clock()
        image = capture()
        counts['polls'] += 1
        if image is None:
            status = 'capture_failed'
            counts['capture_failures'] += 1
            detector.reset_pending()
        else:
            thumbnail = ChangeDetector.thumbnail(image)
            if previous_frame is not None:
                frame_delta = ChangeDetector.difference(thumbnail, previous_frame)
                raw_changes += int(frame_delta > 0)
            previous_frame = thumbnail
            try:
                screen = normalize(image, recorder.crop)
            except ValueError as exc:
                status = 'crop_failed'
                counts['unreadable_screens'] += 1
                detector.reset_pending()
                if not crop_failure_reported:
                    directory = recorder.collection / 'diagnostics'
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f'{recorder.session_id}__crop_failure_{uuid4().hex[:8]}.png'
                    if not cv2.imwrite(str(path), image):
                        raise OSError(f'Could not save diagnostic capture: {path}')
                    height, width = image.shape[:2]
                    report(f'Skipped screen: {exc}; capture={width}x{height}; '
                           f'raw frame saved: {path}. Repeated crop failures are suppressed.')
                    crop_failure_reported = True
            else:
                settled = detector.consider(screen)
                status = detector.status
                if settled:
                    # Persist only after both stability and change checks pass.
                    reasons = selector(screen) if selector is not None else None
                    annotation = recorder.save(image, reasons) if reasons is None or reasons else False
                    detector.mark_saved(screen)
                    if annotation is False:
                        status = 'gap_filter_skipped'
                        counts['skipped'] += 1
                    elif annotation is None:
                        status = 'duplicate'
                        counts['duplicates'] += 1
                    else:
                        status = 'saved'
                        counts['saved'] += 1
                        detail = f' ({", ".join(reasons)})' if reasons else ''
                        report(f'Draft saved: {annotation}{detail}')
        now = clock()
        if now - last_report >= 2:
            def metric(value):
                return 'n/a' if value is None else f'{value:.4f}'
            report(f'Capture status: {status}; polls={counts["polls"]}; '
                   f'saved={counts["saved"]}; filtered={counts["skipped"]}; '
                   f'duplicates={counts["duplicates"]}; crop_failures={counts["unreadable_screens"]}; '
                   f'capture_failures={counts["capture_failures"]}; raw_changes={raw_changes}; '
                   f'frame_delta={metric(frame_delta)}; change={metric(detector.change)}'
                   f'/{change_threshold}; stability={metric(detector.stability)}'
                   f'/{stability_threshold}')
            last_report = now
        if counts['saved'] < max_images:
            remaining = interval - (clock() - poll_start)
            if remaining > 0:
                sleep(remaining)
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
    parser.add_argument('--interval', type=float, default=0.2,
                        help='Target poll period in seconds, including processing time')
    parser.add_argument('--duration', type=float, default=3600, help='Maximum seconds to monitor')
    parser.add_argument('--max-images', type=int, default=500)
    parser.add_argument('--change-threshold', type=float, default=1.0, help='Mean thumbnail grayscale change, 0..255')
    parser.add_argument('--stability-threshold', type=float, default=0.3)
    parser.add_argument('--stable-samples', type=int, default=2)
    parser.add_argument('--crop', type=int, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    parser.add_argument('--save-all', action='store_true',
                        help='Save every settled change instead of only recognition gaps')
    parser.add_argument('--min-examples', type=int, default=3,
                        help='Values with fewer reviewed train examples count as rare')
    parser.add_argument('--max-per-reason', type=int, default=5,
                        help='Stop selecting a reason once this many drafts await review')
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
        selector = None
        if not args.save_all:
            from .selection import GapSelector
            selector = GapSelector.from_collection(args.collection, min_examples=args.min_examples,
                                                   max_per_reason=args.max_per_reason)
        mode = 'all settled changes' if selector is None else 'recognition gaps only'
        print(f'Monitoring with {backend}; split={recorder.split}; saving {mode}; '
              'Ctrl+C stops capture. Drafts require review.', flush=True)
        with capture_source(backend, args.window_title, args.portal_python) as capture:
            counts = monitor(capture, recorder, interval=args.interval, duration=args.duration,
                             max_images=args.max_images, change_threshold=args.change_threshold,
                             stability_threshold=args.stability_threshold, stable_samples=args.stable_samples,
                             selector=selector)
        print(json.dumps(counts))
    except KeyboardInterrupt:
        print('\nMonitoring stopped; saved drafts remain unapproved.')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Monitoring failed: {exc}\n')


if __name__ == '__main__':
    main()
