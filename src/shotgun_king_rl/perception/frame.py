"""Lazy preprocessing scoped to one observation, never reused across frames."""

from functools import cached_property

import cv2

from ..vision import analyzer
from ..vision.layout import normalize_game_screen


class PreparedFrame:
    def __init__(self, image, *, normalized=False):
        self.original = image
        self.normalized = normalized

    @cached_property
    def image(self):
        try:
            return self.original if self.normalized else normalize_game_screen(self.original)
        except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
            return None

    @cached_property
    def board_image(self):
        return analyzer.crop_chessboard(self.image) if self.image is not None else None

    @cached_property
    def board_gray(self):
        return cv2.cvtColor(self.board_image, cv2.COLOR_BGR2GRAY) if self.board_image is not None else None

    @cached_property
    def gray(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if self.image is not None else None

    @cached_property
    def boards(self):
        return analyzer.get_state_matrices(
            self.image, board_img=self.board_image, board_gray=self.board_gray
        )


def read_frame(reader, frame):
    """Allow third-party readers to retain their ordinary image interface."""
    prepared = getattr(reader, "read_prepared", None)
    return prepared(frame) if callable(prepared) else reader.read(frame.original)
