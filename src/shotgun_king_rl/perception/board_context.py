"""Compose reviewed cell-side evidence and independent fixed-row moat evidence."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..collection_layout import annotation_paths
from ..state_schema import SIDE_ALLY, SIDE_EMPTY, SIDE_ENEMY, SUPPORTED_BOARD_IDS
from ..vision.layout import normalize_game_screen
from .context_cells import CellSideReader, MoatRowReader, _valid_board_image
from .context_overlap import NeighborOverlapReader
from .frame import PreparedFrame
from .types import BoardContextReading, Prediction


def _valid_accepted_board(board):
    return (isinstance(board, np.ndarray) and board.shape == (8, 8)
            and (np.issubdtype(board.dtype, np.integer) or np.issubdtype(board.dtype, np.floating))
            and np.all(np.isin(board, SUPPORTED_BOARD_IDS)) and np.count_nonzero(board == 1) == 1)


class BoardContextReader:
    source = 'board-context:reviewed-cells-and-moat-row-v8'

    def __init__(self, references=(), provenance=()):
        references = tuple(references)
        self.side_reader = CellSideReader((image, context.piece_sides) for image, context in references)
        self.moat_reader = MoatRowReader(((image, context.moat_active) for image, context in references
                                        if type(context.moat_active) is bool),
                                       floor_references=((image, context.piece_sides, context.moat_active)
                                                         for image, context in references
                                                         if type(context.moat_active) is bool))
        self.overlap_reader = NeighborOverlapReader((image, context.piece_sides) for image, context in references)
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
        return {'source': self.source, 'matching': 'exact_bounded_or_reviewed_glyph_cells_and_fixed_moat_row',
                'rendering_limits': {'cell_thumbnail': [24, 24], 'mean_channel_error': 2, 'max_channel_error': 24},
                'glyph_limits': {'minimum_correlation': .98, 'side_margin': .04,
                                 'maximum_brightness_difference': 20, 'translation_pixels': 6,
                                 'translation_grid_pixels': .5,
                                 'stroke_minimum_overlap': .95, 'stroke_minimum_distance_score': .98,
                                 'supplies_empty_cells': False},
                'neighbor_overlap': {'source': self.overlap_reader.source, 'maximum_fragment_pixels': 400,
                                     'visible_floor_core': [22, 22, 43, 43], 'requires_occupied_neighbor': True,
                                     'floor_max_channel_error': 24, 'floor_core_mean_error': 2,
                                     'minimum_neighbor_floor_fraction': .25, 'red_channel_margin': 100},
                'moat_floor_evidence': {'minimum_visible_empty_tiles': 2, 'requires_agreement': True},
                'accepted_board_identities': {'empty': 'reviewed_cell_evidence_required', 'player_king': 'ally',
                                              'other_pieces': 'reviewed_side_evidence_required'},
                'references': self._joint_frames, 'side_reference_frames': self._side_frames,
                'moat_reference_frames': self.moat_reader.reference_rows,
                'side_reference_cells': self.side_reader.reference_cells,
                'side_source': self.side_reader.source, 'moat_source': self.moat_reader.source,
                'provenance': [dict(item) for item in self._provenance]}

    def read(self, image):
        return self.read_prepared(PreparedFrame(image))

    def read_prepared(self, frame):
        return self._read_context(frame)

    def read_with_board(self, frame, board):
        """Use only identities from a separately accepted complete piece board."""
        if not _valid_accepted_board(board):
            return Prediction.unknown(source=self.source, reason='invalid_accepted_board')
        identities = np.full((8, 8), -1, np.int8)
        identities[board == 1] = SIDE_ALLY
        return self._read_context(frame, identities, board)

    def read_side_cells(self, image, *, board=None):
        """Return every runtime side decision for read-only review exports."""
        if not _valid_board_image(image):
            raise ValueError('Expected a BGR board image')
        if board is not None and not _valid_accepted_board(board):
            raise ValueError('Expected a separately accepted piece board')
        known_identities = np.where(board == 1, SIDE_ALLY, -1) if board is not None else None
        predictions = [[self.side_reader.read_cell(
            image[y*65:(y+1)*65, x*65:(x+1)*65],
            known_identity=int(known_identities[y, x]) if known_identities is not None else -1)
            for x in range(8)] for y in range(8)]
        resolved = [row.copy() for row in predictions]
        if board is not None:
            for y in range(8):
                for x in range(8):
                    result = predictions[y][x]
                    if (not result.is_accepted and result.reason == 'unmatched_cell'
                            and self.overlap_reader.accepts_empty(image, y, x, board, predictions)):
                        resolved[y][x] = Prediction.accepted(SIDE_EMPTY, source=self.overlap_reader.source)
        return resolved

    def _read_sides(self, image, *, known_identities=None, board=None):
        """Use scene evidence only after independent cell decisions are fixed."""
        baseline = self.side_reader.read_board(image, known_identities=known_identities)
        if (baseline.is_accepted or board is None or image is None
                or baseline.reason in ('invalid_board_image', 'invalid_known_identities', 'references_unavailable')):
            return baseline
        predictions = self.read_side_cells(image, board=board)
        sides = np.zeros((8, 8), np.int8)
        for y in range(8):
            for x in range(8):
                result = predictions[y][x]
                if result.is_accepted:
                    sides[y, x] = result.value
                else:
                    return Prediction.unknown(source=self.source, reason=f'{result.reason}:{y},{x}')
        return Prediction.accepted(sides, source=self.source)

    def _read_context(self, frame, identities=None, board=None):
        if not self._side_frames:
            return Prediction.unknown(source=self.source, reason='references_unavailable')
        image = frame.board_image
        if image is None:
            return Prediction.unknown(source=self.source, reason='invalid_frame')
        sides = self._read_sides(image, known_identities=identities, board=board)
        if not sides.is_accepted:
            return Prediction.unknown(source=self.source, reason=f'piece_sides:{sides.reason}')
        moat = self.moat_reader.read_board(image, empty_cells=sides.value[4] == SIDE_EMPTY if board is not None else None)
        if not moat.is_accepted:
            return Prediction.unknown(source=self.source, reason=f'moat:{moat.reason}')
        return Prediction.accepted(BoardContextReading(sides.value, moat.value), source=self.source)
