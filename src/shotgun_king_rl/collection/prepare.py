"""Import inbox screenshots and generate provisional annotations for review."""

import argparse
import json
from pathlib import Path

import cv2

from .dataset import choose_session_split
from .images import detect_crop, normalize
from .importer import import_images
from .screen_state import propose
from ..vision import analyzer

ROOT = Path(__file__).resolve().parents[3]

COLLECTION = ROOT / 'data/collection'


def draft_annotation(image, image_name, collection, session_id, split, crop=None):
    """Propose one unapproved annotation without modifying existing data."""
    game_crop = crop or detect_crop(image)
    screen = normalize(image, game_crop)
    analyzer.reset_analyzer_cache()
    ids = analyzer.get_state_matrix(screen, include_special_knights=True)
    names = ['empty', 'player_king', 'pawn', 'knight', 'bishop', 'rook', 'white_king', 'special_knight']
    data = {'schema': 'piece-cells-v1', 'image': '../originals/' + image_name, 'session_id': session_id, 'floor': None, 'split': split, 'confirmed': False, 'game_crop': game_crop, 'exclude_cells': [], 'annotation_source': 'Automatic template predictions; human review required; ID 6 provisionally mapped to white_king', 'board': [[names[int(v)] for v in row] for row in ids]}
    data.update(propose(screen, collection))
    return data


def prepare_session(collection, session_id, crop=None):
    moved = import_images(collection, session_id)
    session = collection / 'sessions' / session_id
    image_paths = [
        path for path in sorted((session / 'originals').iterdir())
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ]
    split = choose_session_split(collection, session_id, len(image_paths))
    created = 0
    errors = []
    for image_path in image_paths:
        annotation = session / 'annotations' / (image_path.name + '.json')
        # Preserve earlier annotations, including names used before this tool.
        existing = [json.loads(p.read_text()) for p in (session / 'annotations').glob('*.json')]
        if any(Path(a.get('image', '')).name == image_path.name for a in existing):
            continue
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError('Image cannot be read')
            data = draft_annotation(image, image_path.name, collection, session_id, split, crop)
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
