"""Recognize reviewed HUD value shapes despite small rendering differences."""

import cv2
import numpy as np

from ..perception.hud import _stat_anchor


def numeric_token(image, offset=0):
    """Normalize a complete value, retaining separators and relative spacing."""
    anchor = _stat_anchor(image)
    if anchor is None:
        return None
    top, bottom = anchor + offset - 3, anchor + offset + 30
    if top < 0 or bottom > image.shape[0]:
        return None
    mask = (image[top:bottom, 270:354].max(axis=2) > 140).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 2), np.uint8))
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = mask.shape
    width = round(width * 32 / height)
    if not 1 <= width <= 120:
        return None
    result = np.zeros((40, 128), np.uint8)
    left = (128 - width) // 2
    result[4:36, left:left + width] = cv2.resize(mask, (width, 32), interpolation=cv2.INTER_NEAREST)
    return result


def token_score(query, template):
    """Measure symmetric stroke distance; large shape differences cost more."""
    query_distance = cv2.distanceTransform(1 - query, cv2.DIST_L2, 3)
    template_distance = cv2.distanceTransform(1 - template, cv2.DIST_L2, 3)
    error = np.minimum(query_distance[template > 0], 4).mean() + np.minimum(template_distance[query > 0], 4).mean()
    return float(1 - error / 8)


def match_floor_shapes(query, examples):
    """Match normalized red floor strokes, keeping conflicting labels unknown."""
    if query is None or query.std() < 5:
        return None
    query = (query > 0).astype(np.uint8)
    scores = {}
    for value, template in examples:
        if template.shape != query.shape or template.std() < 5:
            continue
        score = token_score(query, (template > 0).astype(np.uint8))
        scores[value] = max(scores.get(value, -1), score)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < .90:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .08:
        return None
    return ranked[0]


def match_numeric(query, context, examples):
    """Require both reviewed row context and an unambiguous numeric shape."""
    if query is None or context is None:
        return None
    scores = {}
    for value, token, row in examples:
        context_score = float(cv2.matchTemplate(context, row, cv2.TM_CCOEFF_NORMED).max())
        if context_score < .90:
            continue
        scores[value] = max(scores.get(value, -1), token_score(query, token))
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < .90:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .08:
        return None
    return ranked[0]
