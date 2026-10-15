"""Normalize captured game windows without changing their originals."""

from ..vision.layout import detect_game_crop, normalize_game_screen


def detect_crop(image):
    try:
        return detect_game_crop(image)
    except ValueError as exc:
        raise ValueError("Game area not detected; supply --crop LEFT TOP RIGHT BOTTOM") from exc


def normalize(image, crop):
    return normalize_game_screen(image, crop)
