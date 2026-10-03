"""Prepare private screenshots and serve a local annotation review interface."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sys
import threading
import webbrowser

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
import analyzer
from build_piece_dataset import CLASSES, group_split
from import_collection import import_images

COLLECTION = ROOT / 'data/collection'


def detect_crop(image):
    if image.shape[:2] == (720, 1280):
        return [0, 0, 1280, 720]
    # Colored game content excludes gray window decorations and black margins.
    colored = (image.max(axis=2).astype(int) - image.min(axis=2).astype(int) > 8) & (image.max(axis=2) > 20)
    rows = np.flatnonzero(colored.sum(axis=1) > image.shape[1] * .5)
    cols = np.flatnonzero(colored.sum(axis=0) > image.shape[0] * .5)
    if len(rows) and len(cols):
        crop = [int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)]
        if abs((crop[2] - crop[0]) / (crop[3] - crop[1]) - 16 / 9) < .08:
            return crop
    mask = (image.max(axis=2) > 20).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > image.shape[1] * .5 and abs(w / h - 16 / 9) < .08:
            candidates.append((w * h, [x, y, x + w, y + h]))
    if not candidates:
        raise ValueError('Game area not detected; supply --crop LEFT TOP RIGHT BOTTOM')
    return max(candidates)[1]


def normalize(image, crop):
    if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError('Crop must contain four integer coordinates')
    left, top, right, bottom = crop
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
        raise ValueError('Crop exceeds image bounds')
    return cv2.resize(image[top:bottom, left:right], (1280, 720), interpolation=cv2.INTER_AREA)


def prepare_session(collection, session_id, crop=None):
    moved = import_images(collection, session_id)
    session = collection / 'sessions' / session_id
    created = 0
    errors = []
    for image_path in sorted((session / 'originals').iterdir()):
        if image_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        annotation = session / 'annotations' / (image_path.name + '.json')
        # Preserve earlier annotations, including names used before this tool.
        existing = [json.loads(p.read_text()) for p in (session / 'annotations').glob('*.json')]
        if any(Path(a.get('image', '')).name == image_path.name for a in existing):
            continue
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError('Image cannot be read')
            game_crop = crop or detect_crop(image)
            screen = normalize(image, game_crop)
            analyzer.reset_analyzer_cache()
            ids = analyzer.get_state_matrix(screen)
            names = ['empty', 'player_king', 'pawn', 'knight', 'bishop', 'rook', 'white_king']
            data = {'schema': 'piece-cells-v1', 'image': '../originals/' + image_path.name, 'session_id': session_id, 'floor': None, 'split': group_split(session_id), 'confirmed': False, 'game_crop': game_crop, 'exclude_cells': [], 'annotation_source': 'Automatic template predictions; human review required; ID 6 provisionally mapped to white_king', 'board': [[names[int(v)] for v in row] for row in ids]}
            annotation.write_text(json.dumps(data, indent=2) + '\n')
            created += 1
        except ValueError as exc:
            errors.append(f'{image_path.name}: {exc}')
    return {'moved': moved, 'created': created, 'errors': errors}


def annotation_paths(collection):
    return sorted((collection / 'sessions').glob('*/annotations/*.json'))


def load_item(collection, relative):
    path = (collection / relative).resolve()
    if path not in [p.resolve() for p in annotation_paths(collection)]:
        raise ValueError('Unknown annotation')
    raw = path.read_bytes()
    data = json.loads(raw)
    image_path = (path.parent / data['image']).resolve()
    originals = (path.parent.parent / 'originals').resolve()
    if image_path.parent != originals:
        raise ValueError('Image must be a session original')
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Original image cannot be read')
    return path, data, image, hashlib.sha256(raw).hexdigest()


def save_item(collection, payload):
    path, current, image, revision = load_item(collection, payload['id'])
    if payload.get('revision') != revision:
        raise ValueError('Annotation changed; reload before saving')
    board = payload['board']
    if not isinstance(board, list) or len(board) != 8 or any(not isinstance(row, list) or len(row) != 8 or any(v not in CLASSES for v in row) for row in board):
        raise ValueError('Board must contain 8x8 valid labels')
    floor = payload.get('floor')
    if floor is not None and (type(floor) is not int or floor < 1):
        raise ValueError('Floor must be a positive integer')
    split = payload['split']
    if split not in ('train', 'val', 'test'):
        raise ValueError('Invalid split')
    if type(payload.get('confirmed')) is not bool:
        raise ValueError('Invalid approval flag')
    if payload['confirmed'] and floor is None:
        raise ValueError('Set the floor before approval')
    normalize(image, payload['game_crop'])
    excluded = payload['exclude_cells']
    if not isinstance(excluded, list) or any(not isinstance(cell, list) or len(cell) != 2 or any(type(v) is not int or not 0 <= v < 8 for v in cell) for cell in excluded):
        raise ValueError('Invalid excluded cells')
    siblings = list(path.parent.glob('*.json'))
    for sibling in siblings:
        other = json.loads(sibling.read_text())
        if sibling != path and other.get('split', group_split(current['session_id'])) != split:
            raise ValueError('Keep the session split unchanged; change the entire session together')
    current.update(board=board, floor=floor, split=split, confirmed=payload['confirmed'], game_crop=payload['game_crop'], exclude_cells=excluded)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(current, indent=2) + '\n')
    temporary.replace(path)
    return {'revision': hashlib.sha256(path.read_bytes()).hexdigest(), 'confirmed': current['confirmed']}


def make_server(collection, port=0):
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, kind='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == '/':
                    html = Path(__file__).with_name('review_collection.html').read_text().replace('__TOKEN__', token)
                    return self.reply(html.encode(), 'text/html; charset=utf-8')
                if self.path == '/api/items':
                    items = []
                    for path in annotation_paths(collection):
                        a = json.loads(path.read_text())
                        items.append({'id': path.relative_to(collection).as_posix(), 'session': a['session_id'], 'floor': a.get('floor'), 'confirmed': a.get('confirmed', False)})
                    return self.reply(json.dumps(items).encode())
                from urllib.parse import urlparse, parse_qs
                query = parse_qs(urlparse(self.path).query)
                path, data, image, revision = load_item(collection, query['id'][0])
                if self.path.startswith('/api/item?'):
                    return self.reply(json.dumps({'data': data, 'revision': revision}).encode())
                if self.path.startswith('/api/board?'):
                    crop = json.loads(query['crop'][0]) if 'crop' in query else data['game_crop']
                    board = analyzer.crop_chessboard(normalize(image, crop))
                    ok, encoded = cv2.imencode('.png', board)
                    if not ok:
                        raise ValueError('Image encoding failed')
                    return self.reply(encoded.tobytes(), 'image/png')
                raise ValueError('Unknown endpoint')
            except (ValueError, KeyError, OSError) as exc:
                self.reply(json.dumps({'error': str(exc)}).encode(), status=400)

        def do_POST(self):
            if self.path != '/api/save' or self.headers.get('X-Review-Token') != token:
                return self.reply(b'{"error":"Unauthorized"}', status=403)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 100000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size))
                with lock:
                    result = save_item(collection, payload)
                self.reply(json.dumps(result).encode())
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(json.dumps({'error': str(exc)}).encode(), status=400)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', help='Import inbox and prepare this session before review')
    parser.add_argument('--collection', type=Path, default=COLLECTION)
    parser.add_argument('--crop', type=int, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true', help='Open the review page in the default browser')
    args = parser.parse_args()
    try:
        if args.session:
            print(json.dumps(prepare_session(args.collection, args.session, args.crop), indent=2), flush=True)
        if args.prepare_only:
            return
        server = make_server(args.collection, args.port)
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'Review: {url} (Ctrl+C to stop)', flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Review failed: {exc}\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
