"""Reader adapters for the current independent vision implementations."""

import cv2
import numpy as np

from ..vision import ammo as ammo_vision
from ..vision import analyzer
from ..vision import screens
from ..vision.layout import normalize_game_screen
from .types import AmmoReading, Prediction, ScreenKind


class TemplatePieceReader:
    """Read the board with the current per-cell template implementation."""

    source = "piece-templates:v2"

    def __init__(self, royal_templates=None):
        self.royal_templates = royal_templates
        if royal_templates is not None:
            self.source = "piece-templates:royals-v1"

    @classmethod
    def from_collection(cls, collection):
        from .royals import RoyalTemplates

        return cls(RoyalTemplates.from_collection(collection))

    def read(self, image) -> Prediction[np.ndarray]:
        if not _valid_image(image):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        image = _normalize(image)
        if image is None or analyzer.crop_chessboard(image) is None:
            return Prediction.unknown(source=self.source, reason="board_not_found")

        board = analyzer.get_state_matrix(image, include_special_knights=True)
        if self.royal_templates is not None:
            board = self.royal_templates.refine(image, board)
        king_count = int(np.count_nonzero(board == 1))
        if king_count != 1:
            return Prediction.rejected(
                board,
                source=self.source,
                reason=f"player_king_count:{king_count}",
            )
        if np.any(board == 7):
            return Prediction.rejected(
                board, source=self.source, reason="unsupported_runtime_piece:special_knight"
            )
        if np.any(np.isin(board, (8, 9))):
            return Prediction.rejected(
                board, source=self.source, reason="unsupported_runtime_piece:distinct_royal"
            )
        if self.royal_templates is not None and np.any(board == 6):
            return Prediction.rejected(
                board, source=self.source, reason="ambiguous_royal_kind"
            )
        return Prediction.accepted(board, source=self.source)


class RedPixelAmmoReader:
    """Read loaded and reserve ammunition from the calibrated HUD regions."""

    source = "ammo-red-slots:v1"

    def read(self, image) -> Prediction[AmmoReading]:
        image = _normalize(image)
        reading = ammo_vision.read_ammo_count(image) if image is not None else None
        if reading is None:
            return Prediction.unknown(source=self.source, reason="ammo_not_read")
        loaded, reserve = reading
        return Prediction.accepted(
            AmmoReading(loaded=loaded, reserve=reserve),
            source=self.source,
        )


class OverlayScreenReader:
    """Recognize explicit overlays and verify the gameplay board structure."""

    source = "screen-rules:v2"

    def read(self, image) -> Prediction[ScreenKind]:
        if not _valid_image(image):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        image = _normalize(image)
        if image is None:
            return Prediction.unknown(source=self.source, reason="game_area_not_found")
        if screens.check_retry_popup(image):
            return Prediction.accepted(ScreenKind.RETRY, source=self.source)
        if screens.check_card_selection_screen(image):
            return Prediction.accepted(ScreenKind.CARD_SELECTION, source=self.source)
        board = analyzer.get_state_matrix(image)
        if screens.check_gameplay_board(image) and int(np.count_nonzero(board == 1)) >= 1:
            return Prediction.accepted(ScreenKind.GAMEPLAY, source=self.source)
        return Prediction.unknown(
            source=self.source,
            reason="gameplay_or_transition_unverified",
        )


def _normalize(image):
    try:
        return normalize_game_screen(image)
    except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
        return None


def _valid_image(image) -> bool:
    return (
        isinstance(image, np.ndarray)
        and image.ndim == 3
        and image.shape[0] > 0
        and image.shape[1] > 0
        and image.shape[2] >= 3
    )
