"""Turn settled inbox files into recoverable, unapproved review drafts."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid

import cv2
from PIL import Image

from .audit import audit_collection
from .backlog import build_backlog
from .importer import COLLECTION, import_images
from .locking import collection_lock
from .prepare import prepare_session

SUFFIXES = {'.png', '.jpg', '.jpeg'}


def images(directory):
    return sorted(p for p in directory.glob('*') if p.is_file() and p.suffix.lower() in SUFFIXES)


def signature(path):
    stat = path.stat()
    return [stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CollectionPipeline:
    def __init__(self, collection, session, *, settle=2.0, crop=None, retry_errors=False):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', session):
            raise ValueError('Invalid session ID')
        if not math.isfinite(settle) or settle < 0:
            raise ValueError('Settle time must be nonnegative')
        self.collection = Path(collection)
        self.session = session
        self.directory = self.collection / 'sessions' / session
        self.settle = settle
        self.crop = list(crop) if crop else None
        self.observed = {}
        self.inbox_names = None
        self.state_path = self.collection / '.pipeline' / (session + '.json')
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault('failed', {})
        self.state.setdefault('events', [])
        if retry_errors or self.state.get('crop') != self.crop:
            self.state['failed'] = {}
        self.state['crop'] = self.crop

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.state, indent=2) + '\n')
        temporary.replace(self.state_path)

    def fail(self, key, path, error):
        self.state['failed'][key] = {'signature': signature(path), 'error': str(error)}
        self.state['pending_report'] = True

    def unchanged_failure(self, key, path):
        return self.state['failed'].get(key, {}).get('signature') == signature(path)

    def recover_duplicate(self):
        pending = self.state.get('pending_duplicate')
        if not pending:
            return 0
        event = pending['event']
        source = self.collection / 'inbox' / event['duplicate']
        target = self.collection / event['archive']
        if target.exists():
            if digest(target) != pending['sha256']:
                raise ValueError('Duplicate archive changed; refusing to overwrite it')
        elif source.exists():
            if digest(source) != pending['sha256']:
                # The old move never happened; let the new inbox contents settle again.
                del self.state['pending_duplicate']
                self.save()
                return 0
            source.rename(target)
        else:
            raise ValueError('Pending duplicate is missing from both inbox and archive')
        if not any(item['archive'] == event['archive'] for item in self.state['events']):
            self.state['events'].append(event)
        del self.state['pending_duplicate']
        self.save()
        return 1

    def finish_report(self, report):
        self.state['pending_report'] = False
        self.state['events'] = []
        self.state.pop('report_run', None)
        self.state['last_report'] = str(report.relative_to(self.collection))
        self.save()

    def tick(self, now=None):
        """Process settled files while preserving unfinished batch state."""
        with collection_lock(self.collection):
            return self._tick(time.monotonic() if now is None else now)

    def _tick(self, now):
        # Finish an interrupted publication before accepting another batch's events.
        report = None
        if self.state.get('report_run'):
            report = self.export()
            self.finish_report(report)
        recovered_duplicates = self.recover_duplicate()
        inbox = self.collection / 'inbox'
        inbox.mkdir(parents=True, exist_ok=True)
        ready, pending = [], []
        paths = images(inbox)
        if self.inbox_names is not None:
            paths = [p for p in paths if p.name in self.inbox_names]
        active_keys = {'inbox/' + p.name for p in paths}
        originals = images(self.directory / 'originals')
        active_keys.update('originals/' + p.name for p in originals)
        self.state['failed'] = {k: v for k, v in self.state['failed'].items() if k in active_keys}
        for path in paths:
            key = 'inbox/' + path.name
            current = signature(path)
            previous, since = self.observed.get(path.name, (None, now))
            if current != previous:
                since = now
            self.observed[path.name] = (current, since)
            if self.unchanged_failure(key, path):
                continue
            self.state['failed'].pop(key, None)
            if now - since < self.settle:
                pending.append(path.name)
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                if signature(path) != current:
                    pending.append(path.name)
                    continue
                ready.append(path)
            except (OSError, ValueError, SyntaxError) as exc:
                if signature(path) != current:
                    pending.append(path.name)
                else:
                    self.fail(key, path, exc)
        self.observed = {p.name: self.observed[p.name] for p in paths}

        moved, duplicates, created = 0, recovered_duplicates, 0
        if ready:
            hashes = {digest(p): p.name for p in originals}
            for path in ready:
                key = 'inbox/' + path.name
                fingerprint = digest(path)
                if signature(path) != self.observed[path.name][0]:
                    pending.append(path.name)
                    continue
                if fingerprint in hashes:
                    archive = self.directory / 'duplicates'
                    archive.mkdir(parents=True, exist_ok=True)
                    target = archive / (uuid.uuid4().hex + path.suffix.lower())
                    event = {'duplicate': path.name, 'duplicate_of': hashes[fingerprint], 'archive': str(target.relative_to(self.collection))}
                    self.state['pending_duplicate'] = {'event': event, 'sha256': fingerprint}
                    self.state['pending_report'] = True
                    self.save()
                    duplicates += self.recover_duplicate()
                elif (self.directory / 'originals' / path.name).exists():
                    self.fail(key, path, 'Filename collision with different content; rename the inbox file')
                else:
                    self.state['pending_report'] = True
                    self.save()
                    moved += import_images(self.collection, self.session, [path.name])
                    hashes[fingerprint] = path.name
                if not path.exists():
                    self.state['failed'].pop(key, None)

        # Recover interrupted imports, while never regenerating an existing annotation.
        annotations = self.directory / 'annotations'
        known = {Path(json.loads(p.read_text()).get('image', '')).name for p in annotations.glob('*.json')}
        todo = []
        for path in images(self.directory / 'originals'):
            key = 'originals/' + path.name
            if path.name in known:
                self.state['failed'].pop(key, None)
            elif not self.unchanged_failure(key, path):
                self.state['failed'].pop(key, None)
                todo.append(path.name)
        if todo:
            self.state['pending_report'] = True
            self.save()
            # Prepare separately so one bad screenshot cannot block other drafts.
            for name in todo:
                path = self.directory / 'originals' / name
                try:
                    result = prepare_session(self.collection, self.session, self.crop, image_names=[], original_names=[name])
                    created += result['created']
                    if result['errors']:
                        self.fail('originals/' + name, path, '; '.join(result['errors']))
                except (ValueError, OSError, cv2.error) as exc:
                    self.fail('originals/' + name, path, exc)
                self.save()

        if self.state.get('pending_report'):
            report = self.export()
            self.finish_report(report)
        self.save()
        return {'session': self.session, 'moved': moved, 'created': created, 'duplicates': duplicates, 'pending': pending, 'errors': self.state['failed'], 'report': str(report) if report else None}

    def export(self):
        if not self.state.get('report_run'):
            self.state['report_run'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex
        # Persist the report identity and its events before creating any output.
        self.save()
        parent = self.collection / 'reports' / 'pipeline' / self.session
        output = parent / self.state['report_run']
        if output.exists():
            completed = json.loads((output / 'pipeline.json').read_text())
            if completed.get('status') != 'complete' or completed.get('report_id') != self.state['report_run']:
                raise ValueError('Existing report is not a completed pipeline report')
            return output
        staging = parent / ('.' + self.state['report_run'] + '.partial')
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        (staging / 'audit.json').write_text(json.dumps(audit_collection(self.collection / 'sessions'), indent=2) + '\n')
        if any((self.directory / 'annotations').glob('*.json')):
            build_backlog(self.collection, staging / 'backlog', [self.session])
        result = {'status': 'complete', 'report_id': self.state['report_run'], 'session': self.session, 'errors': self.state['failed'], 'events': self.state['events'], 'review_required': True, 'notice': 'Drafts only; no approval, dataset build, game input or training.'}
        (staging / 'pipeline.json').write_text(json.dumps(result, indent=2) + '\n')
        staging.rename(output)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, help='Use one ID for the entire play session')
    parser.add_argument('--collection', type=Path, default=COLLECTION)
    parser.add_argument('--interval', type=float, default=1.0, help='File settling check interval')
    parser.add_argument('--settle', type=float, default=2.0)
    parser.add_argument('--timeout', type=float, default=30.0, help='Maximum wait for files to settle')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--crop', nargs=4, type=int)
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (args.interval, args.settle, args.timeout)) or args.interval <= 0 or args.settle < 0 or args.timeout <= 0:
        parser.error('Use positive interval/timeout and nonnegative settle')
    try:
        with collection_lock(args.collection):
            pipeline = CollectionPipeline(args.collection, args.session, settle=args.settle, crop=args.crop, retry_errors=args.retry_errors)
            # This invocation owns a fixed batch; later arrivals wait for the next run.
            pipeline.inbox_names = {p.name for p in images(args.collection / 'inbox')}
            deadline = time.monotonic() + args.timeout
            totals = {'moved': 0, 'created': 0, 'duplicates': 0}
            while True:
                result = pipeline.tick()
                for key in totals:
                    totals[key] += result[key]
                if not result['pending'] or time.monotonic() >= deadline:
                    break
                time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
            kinds = Counter()
            for path in (pipeline.directory / 'annotations').glob('*.json'):
                data = json.loads(path.read_text())
                kinds[(data.get('screen_state') or {}).get('screen_kind') or 'unknown'] += 1
            result.update(totals)
            result['session_screen_kinds'] = dict(kinds)
            result['report'] = str(args.collection / pipeline.state['last_report']) if pipeline.state.get('last_report') else None
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result['pending'] or result['errors']:
                raise SystemExit(1)
    except KeyboardInterrupt:
        parser.exit(130, 'Batch interrupted; rerun to resume saved originals and drafts.\n')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Pipeline failed: {exc}\n')


if __name__ == '__main__':
    main()
