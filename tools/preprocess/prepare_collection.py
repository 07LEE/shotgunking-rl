"""Import inbox screenshots and generate provisional annotations for review."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import analyzer
from build_piece_dataset import group_split
from import_collection import import_images
from screen_state import propose

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
            data.update(propose(screen, collection))
            annotation.write_text(json.dumps(data, indent=2) + '\n')
            created += 1
        except ValueError as exc:
            errors.append(f'{image_path.name}: {exc}')
    return {'moved': moved, 'created': created, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--collection', type=Path, default=COLLECTION)
    parser.add_argument('--crop', type=int, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    args = parser.parse_args()
    try:
        result = prepare_session(args.collection, args.session, args.crop)
        print(json.dumps(result, indent=2))
        if result['errors']:
            parser.exit(1, 'Some originals need crop correction; existing annotations were preserved.\n')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
