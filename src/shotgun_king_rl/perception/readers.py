"""Reader adapters for the current independent vision implementations."""

import numpy as np

from ..vision import ammo as ammo_vision
from ..vision import analyzer
from ..vision import screens
from .types import AmmoReading, Prediction, ScreenKind


class TemplatePieceReader:
    """Read the board with the current per-cell template implementation."""

    source = "piece-templates:v1"

    def read(self, image) -> Prediction[np.ndarray]:
        if not _valid_image(image):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        if analyzer.crop_chessboard(image) is None:
            return Prediction.unknown(source=self.source, reason="board_not_found")

        board = analyzer.get_state_matrix(image)
        king_count = int(np.count_nonzero(board == 1))
        if king_count != 1:
            return Prediction.rejected(
                board,
                source=self.source,
                reason=f"player_king_count:{king_count}",
            )
        return Prediction.accepted(board, source=self.source)


class RedPixelAmmoReader:
    """Read loaded and reserve ammunition from the calibrated HUD regions."""

    source = "ammo-red-slots:v1"

    def read(self, image) -> Prediction[AmmoReading]:
        reading = ammo_vision.read_ammo_count(image)
        if reading is None:
            return Prediction.unknown(source=self.source, reason="ammo_not_read")
        loaded, reserve = reading
        return Prediction.accepted(
            AmmoReading(loaded=loaded, reserve=reserve),
            source=self.source,
        )


class OverlayScreenReader:
    """Recognize explicit retry and card overlays without guessing gameplay."""

    source = "screen-overlays:v1"

    def read(self, image) -> Prediction[ScreenKind]:
        if not _valid_image(image):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        if screens.check_retry_popup(image):
            return Prediction.accepted(ScreenKind.RETRY, source=self.source)
        if screens.check_card_selection_screen(image):
            return Prediction.accepted(ScreenKind.CARD_SELECTION, source=self.source)
        return Prediction.unknown(
            source=self.source,
            reason="gameplay_or_transition_unverified",
        )


def _valid_image(image) -> bool:
    return (
        isinstance(image, np.ndarray)
        and image.ndim == 3
        and image.shape[0] > 0
        and image.shape[1] > 0
        and image.shape[2] >= 3
    )
