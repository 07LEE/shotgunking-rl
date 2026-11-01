"""Read the attack HUD from explicitly reviewed training references."""

import json
from pathlib import Path

import cv2

from .visibility import visual_combat_stats

from ..vision.layout import normalize_game_screen
from .numeric import numeric_token, match_numeric_with_title
from .hud import stat_patch
from .types import Prediction


def attack_patch(image):
    """Return the attack title and number from the shifting HUD panel."""
    return stat_patch(image)

class AttackReader:
    """Return only attack; other combat fields remain unavailable to policy.

    Match reviewed whole rows or complete numeric shapes with matching row context.
    References must come from train; val and test are evaluation inputs only.
    """

    source = "attack-reviewed-templates:v2"

    def __init__(self, examples=(), *, numeric_examples=()):
        self.examples = tuple((value, template.copy()) for value, template in examples)
        self._template_stds = tuple(template.std() for _, template in self.examples)
        self.numeric_examples = tuple((value, token.copy(), row.copy()) for value, token, row in numeric_examples)

    @classmethod
    def from_collection(cls, collection):
        examples = []
        numeric_examples = []
        for path in sorted((Path(collection) / "sessions").glob("*/annotations/*.json")):
            data = json.loads(path.read_text())
            state = data.get("screen_state") or {}
            reviews = state.get("reviews") or {}
            value = visual_combat_stats(state).get("attack")
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
                token = numeric_token(image, 0)
                if token is not None:
                    numeric_examples.append((examples[-1][0], token, template))
        return cls(examples, numeric_examples=numeric_examples)

    def read(self, image) -> Prediction[int]:
        try:
            image = normalize_game_screen(image)
            query = attack_patch(image)
        except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        if query is None or query.std() < 5:
            return Prediction.unknown(source=self.source, reason="attack_not_found")
        scores = {}
        for (value, template), template_std in zip(self.examples, self._template_stds):
            if template.shape != query.shape or template_std < 5:
                continue
            score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            scores[value] = max(scores.get(value, -1), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < .97:
            numeric = match_numeric_with_title(numeric_token(image, 0), stat_patch(image, 0, padding=8), self.numeric_examples)
            if numeric is not None:
                value, score = numeric
                return Prediction.accepted(value, source=self.source, confidence=score)
            return Prediction.unknown(source=self.source, reason="attack_no_match")
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
            return Prediction.unknown(source=self.source, reason="attack_ambiguous")
        value, score = ranked[0]
        return Prediction.accepted(value, source=self.source, confidence=score)
