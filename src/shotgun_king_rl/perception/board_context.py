"""Compose reviewed cell-side evidence and independent fixed-row moat evidence."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..collection_layout import annotation_paths
from ..state_schema import SIDE_ALLY, SIDE_EMPTY, SIDE_ENEMY
from ..vision.layout import normalize_game_screen
from .context_cells import CellSideReader, MoatRowReader
from .frame import PreparedFrame
from .types import BoardContextReading, Prediction


class BoardContextReader:
    source = 'board-context:reviewed-cells-and-moat-row-v2'

    def __init__(self, references=(), provenance=()):
        references = tuple(references)
        self.side_reader = CellSideReader((image, context.piece_sides) for image, context in references)
        self.moat_reader = MoatRowReader((image, context.moat_active) for image, context in references
                                        if type(context.moat_active) is bool)
        self._provenance = tuple(dict(item) for item in provenance)
        self._side_frames = len(references)
        self._joint_frames = sum(type(context.moat_active) is bool for _, context in references)

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
            try:
                image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                image = normalize_game_screen(image, data.get('game_crop'))
                snapshot = PreparedFrame(image, normalized=True).board_image
            except (ValueError, TypeError, cv2.error):
                continue
            if snapshot is None:
                continue
            references.append((snapshot, BoardContextReading(sides, data.get('moat_active'))))
            provenance.append({'annotation': path.name,
                               'annotation_sha256': hashlib.sha256(raw).hexdigest(),
                               'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
                               'moat_active': data.get('moat_active')})
        return cls(references, provenance)

    def metadata(self):
        return {'source': self.source, 'matching': 'exact_cells_and_fixed_moat_row',
                'references': self._joint_frames, 'side_reference_frames': self._side_frames,
                'moat_reference_frames': self.moat_reader.reference_rows,
                'side_reference_cells': self.side_reader.reference_cells,
                'side_source': self.side_reader.source, 'moat_source': self.moat_reader.source,
                'provenance': [dict(item) for item in self._provenance]}

    def read(self, image):
        return self.read_prepared(PreparedFrame(image))

    def read_prepared(self, frame):
        if not self._side_frames:
            return Prediction.unknown(source=self.source, reason='references_unavailable')
        image = frame.board_image
        if image is None:
            return Prediction.unknown(source=self.source, reason='invalid_frame')
        sides = self.side_reader.read_board(image)
        if not sides.is_accepted:
            return Prediction.unknown(source=self.source, reason=f'piece_sides:{sides.reason}')
        moat = self.moat_reader.read_board(image)
        if not moat.is_accepted:
            return Prediction.unknown(source=self.source, reason=f'moat:{moat.reason}')
        return Prediction.accepted(BoardContextReading(sides.value, moat.value), source=self.source)
