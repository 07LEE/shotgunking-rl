"""Conservative context lookup for explicitly reviewed board snapshots.

This bootstrap reader accepts only an identical normalized board image. It is
not a general side/card classifier: unseen positions, overlays and rendering
changes remain unknown until separately validated recognition is available.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..collection_layout import annotation_paths
from ..state_schema import SIDE_ALLY, SIDE_EMPTY, SIDE_ENEMY
from ..vision.layout import normalize_game_screen
from .frame import PreparedFrame
from .types import BoardContextReading, Prediction


class BoardContextReader:
    source = 'board-context-reviewed-snapshots:v1'

    def __init__(self, references=(), provenance=()):
        self._references = tuple((image.copy(), BoardContextReading(
            context.piece_sides.copy(), context.moat_active)) for image, context in references)
        self._provenance = tuple(dict(item) for item in provenance)

    @classmethod
    def from_collection(cls, collection):
        from ..collection.board_metadata import board_metadata

        references, provenance = [], []
        originals = (Path(collection) / 'originals').resolve()
        for path in annotation_paths(collection):
            raw = path.read_bytes()
            data = json.loads(raw)
            state = data.get('screen_state') or {}
            if (data.get('split') != 'train' or data.get('confirmed') is not True
                    or (data.get('piece_review') or {}).get('reviewed') is not True
                    or state.get('screen_kind') != 'gameplay'
                    or (state.get('reviews') or {}).get('screen_kind') is not True
                    or not isinstance(data.get('piece_sides'), list)
                    or type(data.get('moat_active')) is not bool
                    or data.get('exclude_cells')):
                continue
            board = data.get('board')
            labels = {'empty', 'player_king', 'pawn', 'knight', 'bishop', 'rook', 'queen', 'white_king'}
            if (not isinstance(board, list) or len(board) != 8 or any(
                    not isinstance(row, list) or len(row) != 8
                    or any(piece not in labels for piece in row) for row in board)):
                continue
            try:
                sides, _ = board_metadata(data)
                codes = {None: SIDE_EMPTY, 'ally': SIDE_ALLY, 'enemy': SIDE_ENEMY}
                sides = np.array([[codes[cell] for cell in row] for row in sides], dtype=np.int8)
            except (ValueError, KeyError, TypeError):
                continue
            if not isinstance(data.get('image'), str):
                continue
            image_path = (path.parent / data['image']).resolve()
            if image_path.parent != originals or not image_path.is_file():
                continue
            image_bytes = image_path.read_bytes()
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            try:
                image = normalize_game_screen(image, data.get('game_crop'))
                snapshot = PreparedFrame(image, normalized=True).board_image
            except (ValueError, TypeError, cv2.error):
                continue
            if snapshot is None:
                continue
            references.append((snapshot, BoardContextReading(sides, data['moat_active'])))
            provenance.append({'annotation': path.name,
                               'annotation_sha256': hashlib.sha256(raw).hexdigest(),
                               'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
                               'moat_active': data['moat_active']})
        return cls(references, provenance)

    def metadata(self):
        return {'source': self.source, 'matching': 'exact_normalized_board',
                'references': len(self._references),
                'provenance': [dict(item) for item in self._provenance]}

    def read(self, image):
        return self.read_prepared(PreparedFrame(image))

    def read_prepared(self, frame):
        if not self._references:
            return Prediction.unknown(source=self.source, reason='references_unavailable')
        snapshot = frame.board_image
        if snapshot is None:
            return Prediction.unknown(source=self.source, reason='invalid_frame')
        matches = [context for image, context in self._references
                   if np.array_equal(snapshot, image)]
        if not matches:
            return Prediction.unknown(source=self.source, reason='unmatched_board_snapshot')
        context = matches[0]
        if any(other.moat_active != context.moat_active
               or not np.array_equal(other.piece_sides, context.piece_sides)
               for other in matches[1:]):
            return Prediction.unknown(source=self.source, reason='conflicting_references')
        return Prediction.accepted(BoardContextReading(
            context.piece_sides.copy(), context.moat_active), source=self.source)
