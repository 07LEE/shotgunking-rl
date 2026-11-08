"""Independent exact-image evidence for cell sides and the fixed moat row."""

import hashlib

import cv2
import numpy as np

from ..state_schema import MOAT_ROW, SIDE_ALLY, SIDE_EMPTY, SIDE_ENEMY
from ..vision.analyzer import _extract_signal
from .numeric import overlap_score, token_score
from .types import Prediction


CELL_SIZE = 65
BOARD_SIZE = CELL_SIZE * 8


def _valid_board_image(image):
    return (isinstance(image, np.ndarray) and image.shape == (BOARD_SIZE, BOARD_SIZE, 3)
            and image.dtype == np.uint8)


def _key(image):
    return hashlib.sha256(image.tobytes()).digest()


def _appearance(image):
    """Average resampling noise while retaining every part of each cell."""
    return cv2.resize(image, (24 * (image.shape[1] // CELL_SIZE), 24), interpolation=cv2.INTER_AREA)


def _near_labels(image, appearances, labels):
    if appearances is None:
        return set()
    error = np.abs(appearances.astype(np.int16) - _appearance(image).astype(np.int16))
    axes = tuple(range(1, error.ndim))
    # Both limits apply: a low average cannot hide a changed local feature.
    matches = (error.mean(axis=axes) <= 2) & (error.max(axis=axes) <= 24)
    return {label for index in np.flatnonzero(matches) for label in labels[index]}


def _glyph(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    signal = _extract_signal(gray)
    foreground = signal > 100
    if np.count_nonzero(foreground) < 30 or signal.std() < 5:
        return None
    return signal.astype(np.float32), float(gray[10:55, 10:55][foreground].mean())


def _row_glyph(image):
    """Remove scanline color using agreeing floor lanes on both cell edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    left = np.median(gray[:, :8], axis=1)
    right = np.median(gray[:, 57:], axis=1)
    if np.any(np.abs(left-right) > 24):
        return None
    background = np.median(np.concatenate([gray[:, :8], gray[:, 57:]], axis=1), axis=1)
    signal = np.abs(gray-background[:, None])[10:55, 10:55]
    if signal.max() <= 35:
        return None
    signal = signal / signal.max() * 255
    foreground = signal > 100
    if np.count_nonzero(foreground) < 30:
        return None
    return signal, float(gray[10:55, 10:55][foreground].mean())


def has_indicator_occlusion(image):
    """Recognize a substantial central red overlay, without filling its contents."""
    pixels = image.astype(np.int16)
    red = ((pixels[:, :, 2]-pixels[:, :, 1] > 100)
           & (pixels[:, :, 2]-pixels[:, :, 0] > 100)).astype(np.uint8)
    _, components, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    central = set(components[22:43, 22:43].ravel()) - {0}
    return any(stats[label, cv2.CC_STAT_AREA] >= 30
               and stats[label, cv2.CC_STAT_WIDTH] >= 8
               and stats[label, cv2.CC_STAT_HEIGHT] >= 12 for label in central)


class CellSideReader:
    """Match complete cell images without assuming sides from positions or colors.

    Exact evidence is preferred. Bounded color/shape differences permit small
    resampling changes. Reviewed occupied glyphs can match small translations
    only with matching brightness and unambiguous side evidence. Glyphs never
    supply empty cells; unmatched shapes and conflicting labels stay unknown.
    """

    source = 'piece-sides:reviewed-cells-v6'

    def __init__(self, references=()):
        self._labels = {}
        appearances = {}
        glyphs = {}
        row_glyphs = {}
        self.reference_cells = 0
        for image, sides in references:
            if (not _valid_board_image(image) or not isinstance(sides, np.ndarray)
                    or sides.shape != (8, 8) or not np.issubdtype(sides.dtype, np.integer)
                    or not np.all(np.isin(sides, (SIDE_EMPTY, SIDE_ALLY, SIDE_ENEMY)))):
                raise ValueError('Side references require a BGR board and known 8x8 side grid')
            for row in range(8):
                for col in range(8):
                    patch = image[row*CELL_SIZE:(row+1)*CELL_SIZE, col*CELL_SIZE:(col+1)*CELL_SIZE]
                    self._labels.setdefault(_key(patch), set()).add(int(sides[row, col]))
                    appearance = _appearance(patch)
                    key = appearance.tobytes()
                    if key not in appearances:
                        appearances[key] = (appearance, set())
                    appearances[key][1].add(int(sides[row, col]))
                    if sides[row, col] != SIDE_EMPTY:
                        row_glyph = _row_glyph(patch)
                        if row_glyph is not None:
                            signal, brightness = row_glyph
                            key = (signal.tobytes(), brightness)
                            if key not in row_glyphs:
                                row_glyphs[key] = (signal, brightness, set())
                            row_glyphs[key][2].add(int(sides[row, col]))
                        glyph = _glyph(patch)
                        if glyph is not None:
                            signal, brightness = glyph
                            key = (signal.tobytes(), brightness)
                            if key not in glyphs:
                                glyphs[key] = (signal, brightness, set())
                            glyphs[key][2].add(int(sides[row, col]))
                    self.reference_cells += 1
        self._appearances = np.stack([a for a, _ in appearances.values()]) if appearances else None
        self._appearance_labels = tuple(labels for _, labels in appearances.values())
        self._glyphs = tuple(glyphs.values())
        self._row_glyphs = tuple(row_glyphs.values())
        self._subpixel_glyphs = tuple((cv2.resize(signal, (90, 90), interpolation=cv2.INTER_LINEAR), brightness, labels)
                                     for signal, brightness, labels in self._glyphs)
        self._stroke_glyphs = tuple(((signal > 100).astype(np.uint8), brightness, labels)
                                    for signal, brightness, labels in self._glyphs)

    def _glyph_side(self, image):
        glyph = _glyph(image)
        if glyph is None:
            return None
        signal, brightness = glyph
        padded = cv2.copyMakeBorder(signal, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=0)
        scores = {}
        for template, reference_brightness, labels in self._glyphs:
            if abs(brightness - reference_brightness) > 20:
                continue
            score = float(cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED).max())
            for side in labels:
                scores[side] = max(scores.get(side, -1), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if ranked and ranked[0][1] >= .98:
            if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
                return None
            return ranked[0][0]
        # Doubling both complete signals permits half-pixel translation without
        # changing scale, cropping the template, or relaxing score thresholds.
        padded = cv2.resize(padded, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        scores = {}
        for template, reference_brightness, labels in self._subpixel_glyphs:
            if abs(brightness - reference_brightness) > 20:
                continue
            score = float(cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED).max())
            for side in labels:
                scores[side] = max(scores.get(side, -1), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < .98:
            return self._stroke_side(signal, brightness, fallback=lambda: self._row_side(image))
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
            return None
        return ranked[0][0]

    def _stroke_side(self, signal, brightness, *, fallback=None):
        """Require the full foreground, not a selected matching fragment."""
        query = (signal > 100).astype(np.uint8)
        if not query.any():
            return None
        padded = cv2.copyMakeBorder(query, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=0)
        scores = {}
        for template, reference_brightness, labels in self._stroke_glyphs:
            if abs(brightness - reference_brightness) > 20:
                continue
            match = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, (x, y) = cv2.minMaxLoc(match)
            aligned = padded[y:y+45, x:x+45]
            if not aligned.any() or not template.any():
                continue
            evidence = (overlap_score(aligned, template), token_score(aligned, template))
            for side in labels:
                scores[side] = max(scores.get(side, (-1, -1)), evidence)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1][0] < .95 or ranked[0][1][1] < .98:
            return fallback() if fallback is not None else None
        if len(ranked) > 1 and ranked[0][1][0] - ranked[1][1][0] < .04:
            return None
        return ranked[0][0]

    def _row_side(self, image):
        glyph = _row_glyph(image)
        if glyph is None:
            return None
        signal, brightness = glyph
        padded = cv2.copyMakeBorder(signal, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=0)
        scores = {}
        for template, reference_brightness, labels in self._row_glyphs:
            if abs(brightness-reference_brightness) > 20:
                continue
            score = float(cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED).max())
            for side in labels:
                scores[side] = max(scores.get(side, -1), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < .98:
            return None
        if len(ranked) > 1 and ranked[0][1]-ranked[1][1] < .04:
            return None
        return ranked[0][0]

    def read_board(self, image, *, known_identities=None):
        if not _valid_board_image(image):
            return Prediction.unknown(source=self.source, reason='invalid_board_image')
        if known_identities is not None and (
                not isinstance(known_identities, np.ndarray) or known_identities.shape != (8, 8)
                or not np.issubdtype(known_identities.dtype, np.integer)
                or not np.all(np.isin(known_identities, (-1, SIDE_ALLY)))):
            return Prediction.unknown(source=self.source, reason='invalid_known_identities')
        if not self._labels:
            return Prediction.unknown(source=self.source, reason='references_unavailable')
        sides = np.zeros((8, 8), dtype=np.int8)
        for row in range(8):
            for col in range(8):
                patch = image[row*CELL_SIZE:(row+1)*CELL_SIZE, col*CELL_SIZE:(col+1)*CELL_SIZE]
                identity = int(known_identities[row, col]) if known_identities is not None else -1
                result = self.read_cell(patch, known_identity=identity)
                if not result.is_accepted:
                    return Prediction.unknown(source=self.source, reason=f'{result.reason}:{row},{col}')
                sides[row, col] = result.value
        return Prediction.accepted(sides, source=self.source)

    def read_cell(self, patch, *, known_identity=-1):
        """Expose the same decision for offline inspection of every failed cell."""
        if (not isinstance(patch, np.ndarray) or patch.shape != (CELL_SIZE, CELL_SIZE, 3)
                or patch.dtype != np.uint8):
            return Prediction.unknown(source=self.source, reason='invalid_cell_image')
        if type(known_identity) is not int or known_identity not in (-1, SIDE_ALLY):
            return Prediction.unknown(source=self.source, reason='invalid_known_identity')
        if not self._labels:
            return Prediction.unknown(source=self.source, reason='references_unavailable')
        if has_indicator_occlusion(patch):
            return Prediction.unknown(source=self.source, reason='indicator_occlusion')
        labels = self._labels.get(_key(patch))
        if labels is None:
            labels = _near_labels(patch, self._appearances, self._appearance_labels)
        if not labels:
            side = self._glyph_side(patch)
            labels = {side} if side is not None else set()
        if known_identity != -1:
            if labels and labels != {known_identity}:
                return Prediction.unknown(source=self.source, reason='identity_conflict')
            labels = {known_identity}
        if not labels:
            return Prediction.unknown(source=self.source, reason='unmatched_cell')
        if len(labels) != 1:
            return Prediction.unknown(source=self.source, reason='conflicting_cell')
        return Prediction.accepted(next(iter(labels)), source=self.source)


class MoatRowReader:
    """Require explicit known moat state for a matching fixed-row image.

    Other rows cannot determine moat state. With an accepted side board, two
    independently visible reviewed empty tiles on this row may establish its
    uniform moat state despite a changed piece arrangement.
    """

    source = 'moat:reviewed-row-v3'

    def __init__(self, references=(), floor_references=()):
        self._labels = {}
        appearances, labels = [], []
        self.reference_rows = 0
        for image, active in references:
            if not _valid_board_image(image) or type(active) is not bool:
                raise ValueError('Moat references require a BGR board and explicit boolean state')
            row = image[MOAT_ROW*CELL_SIZE:(MOAT_ROW+1)*CELL_SIZE]
            self._labels.setdefault(_key(row), set()).add(active)
            appearances.append(_appearance(row))
            labels.append({active})
            self.reference_rows += 1
        self._appearances = np.stack(appearances) if appearances else None
        self._appearance_labels = tuple(labels)
        self._floor_labels = {}
        floors, floor_labels = [], []
        for image, sides, active in floor_references:
            for col in np.flatnonzero(sides[MOAT_ROW] == SIDE_EMPTY):
                patch = image[MOAT_ROW*65:(MOAT_ROW+1)*65, col*65:(col+1)*65]
                self._floor_labels.setdefault(_key(patch), set()).add(active)
                floors.append(_appearance(patch))
                floor_labels.append({active})
        self._floor_appearances = np.stack(floors) if floors else None
        self._floor_appearance_labels = tuple(floor_labels)

    def read_board(self, image, *, empty_cells=None):
        if not _valid_board_image(image):
            return Prediction.unknown(source=self.source, reason='invalid_board_image')
        if not self._labels:
            return Prediction.unknown(source=self.source, reason='references_unavailable')
        row = image[MOAT_ROW*CELL_SIZE:(MOAT_ROW+1)*CELL_SIZE]
        labels = self._labels.get(_key(row))
        if labels is None:
            labels = _near_labels(row, self._appearances, self._appearance_labels)
        if not labels and empty_cells is not None:
            if (not isinstance(empty_cells, np.ndarray) or empty_cells.shape != (8,)
                    or empty_cells.dtype != np.bool_):
                return Prediction.unknown(source=self.source, reason='invalid_empty_cells')
            evidence = []
            for col in np.flatnonzero(empty_cells):
                patch = row[:, col*65:(col+1)*65]
                found = self._floor_labels.get(_key(patch))
                if found is None:
                    found = _near_labels(patch, self._floor_appearances, self._floor_appearance_labels)
                if len(found) > 1:
                    return Prediction.unknown(source=self.source, reason='conflicting_references')
                if found:
                    evidence.append(next(iter(found)))
            if len(evidence) >= 2:
                labels = set(evidence)
        if not labels:
            return Prediction.unknown(source=self.source, reason='unmatched_row')
        if len(labels) != 1:
            return Prediction.unknown(source=self.source, reason='conflicting_references')
        return Prediction.accepted(next(iter(labels)), source=self.source)
