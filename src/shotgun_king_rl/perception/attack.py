"""Read the attack HUD from explicitly reviewed training references."""

import json
from pathlib import Path

import cv2
import numpy as np

from ..vision.layout import normalize_game_screen
from .types import Prediction


def attack_patch(image):
    """Locate the first bright number in the vertically shifting stats panel.

    Keep its title as context so another stat cannot match by its digit alone.
    Coordinates refer to a normalized 1280x720 game viewport.
    """
    region = image[180:335, 290:340]
    bright = region.max(axis=2) > 150
    rows = np.flatnonzero(bright.sum(axis=1) >= 2)
    if not len(rows):
        return None
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 6) + 1)
    for group in groups:
        top, bottom = int(group[0]), int(group[-1]) + 1
        if bottom - top < 10 or np.count_nonzero(bright[top:bottom]) < 20:
            continue
        y = 180 + top
        # Anchor on the number, retaining the title above it and no next stat.
        return cv2.cvtColor(image[y - 35:y + 30, 270:354], cv2.COLOR_BGR2GRAY)
    return None


class AttackReader:
    """Return only attack; other combat fields remain unavailable to policy.

    This recognizes reviewed whole HUD values, not arbitrary unseen numbers.
    References must come from train; val and test are evaluation inputs only.
    """

    source = "attack-reviewed-templates:v1"

    def __init__(self, examples=()):
        self.examples = tuple((value, template.copy()) for value, template in examples)

    @classmethod
    def from_collection(cls, collection):
        examples = []
        for path in sorted((Path(collection) / "sessions").glob("*/annotations/*.json")):
            data = json.loads(path.read_text())
            state = data.get("screen_state") or {}
            reviews = state.get("reviews") or {}
            value = (state.get("combat_stats") or {}).get("attack")
            if (
                data.get("split") != "train"
                or state.get("screen_kind") != "gameplay"
                or reviews.get("screen_kind") is not True
                or reviews.get("combat_stats") is not True
                or type(value) is not int or value < 0
            ):
                continue
            image_path = (path.parent / data.get("image", "")).resolve()
            if image_path.parent != (path.parent.parent / "originals").resolve():
                raise ValueError(f"{path}: invalid image path")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"{path}: unreadable image")
            image = normalize_game_screen(image, data.get("game_crop"))
            template = attack_patch(image)
            if template is not None:
                examples.append((value, template))
        return cls(examples)

    def read(self, image) -> Prediction[int]:
        try:
            image = normalize_game_screen(image)
            query = attack_patch(image)
        except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        if query is None or query.std() < 5:
            return Prediction.unknown(source=self.source, reason="attack_not_found")
        scores = {}
        for value, template in self.examples:
            if template.shape != query.shape or template.std() < 5:
                continue
            score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            scores[value] = max(scores.get(value, -1), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < .97:
            return Prediction.unknown(source=self.source, reason="attack_no_match")
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
            return Prediction.unknown(source=self.source, reason="attack_ambiguous")
        value, score = ranked[0]
        return Prediction.accepted(value, source=self.source, confidence=score)
