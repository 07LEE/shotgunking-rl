"""Separate named optional HUD rows from reviewed integer glyphs."""

import cv2

from ..perception.hud import stat_patch
from .numeric import numeric_token, token_score


def match_named_integer(screen, field, titles, integers):
    """Require the requested title and a clearly supported integer shape.

    Titles and integer examples must originate in reviewed training screens.
    The score is stroke similarity, not a probability or whole-row correlation.
    """
    candidates = []
    for offset in (192, 256, 320):
        row = stat_patch(screen, row_offset=offset, padding=8)
        if row is None:
            continue
        title_scores = {}
        for name, title in titles:
            if title.std() < 5:
                continue
            score = float(cv2.matchTemplate(row[:46], title, cv2.TM_CCOEFF_NORMED).max())
            title_scores[name] = max(title_scores.get(name, -1), score)
        ranked = sorted(title_scores.items(), key=lambda pair: pair[1], reverse=True)
        if not ranked or ranked[0][0] != field or ranked[0][1] < .90:
            continue
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
            continue
        query = numeric_token(screen, offset)
        if query is None:
            continue
        scores = {}
        for value, token in integers:
            score = token_score(query, token)
            scores[value] = max(scores.get(value, -1), score)
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        # A .90 score limits mean clipped stroke distance to 0.4 pixels;
        # a .0625 gap requires 0.25 pixels of distance separation.
        if not ranked or ranked[0][1] < .90:
            continue
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .0625:
            continue
        candidates.append(ranked[0])
    if candidates and len({value for value, _ in candidates}) == 1:
        return max(candidates, key=lambda pair: pair[1])
    return None
