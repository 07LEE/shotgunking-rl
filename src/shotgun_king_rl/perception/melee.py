"""Read the melee damage HUD from explicitly reviewed training references."""

import json
from pathlib import Path

import cv2

from ..vision.layout import normalize_game_screen
from .hud import stat_patch
from .types import Prediction


def melee_damage_patch(image):
    """Return the melee damage title and damage from the shifting HUD panel."""
    return stat_patch(image, row_offset=192)


class MeleeDamageReader:
    """Return visible melee damage as an integer; other combat fields remain unavailable to policy.

    This recognizes reviewed whole HUD values, not arbitrary unseen numbers.
    References must come from train; val and test are evaluation inputs only.
    Only positive labels with a verified matching title become references.
    Missing rows and zero labels remain unknown; they do not prove zero damage.
    """

    source = "melee-damage-reviewed-templates:v1"

    def __init__(self, examples=()):
        self.examples = tuple((value, template.copy()) for value, template in examples
                              if type(value) is int and value > 0)
        self._template_stds = tuple(template.std() for _, template in self.examples)

    @classmethod
    def from_collection(cls, collection, *, title_reference):
        """Require a caller-verified visible melee title crop from reviewed train.

        title_reference is the top 30 rows of melee_damage_patch. It must
        identify the title, rather than the number or the whole HUD panel.
        """
        if (title_reference is None or title_reference.shape != (30, 84)
                or title_reference.std() < 5):
            raise ValueError("A visible melee title reference is required")
        examples = []
        for path in sorted((Path(collection) / "sessions").glob("*/annotations/*.json")):
            data = json.loads(path.read_text())
            state = data.get("screen_state") or {}
            reviews = state.get("reviews") or {}
            value = (state.get("combat_stats") or {}).get("melee_damage")
            if (
                data.get("split") != "train"
                or state.get("screen_kind") != "gameplay"
                or reviews.get("screen_kind") is not True
                or reviews.get("combat_stats") is not True
                or type(value) is not int or value <= 0
            ):
                continue
            image_path = (path.parent / data.get("image", "")).resolve()
            if image_path.parent != (path.parent.parent / "originals").resolve():
                raise ValueError(f"{path}: invalid image path")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"{path}: unreadable image")
            image = normalize_game_screen(image, data.get("game_crop"))
            template = melee_damage_patch(image)
            if template is not None and template[:30].std() >= 5:
                title_score = float(cv2.matchTemplate(
                    template[:30], title_reference, cv2.TM_CCOEFF_NORMED
                )[0, 0])
                if title_score >= .97:
                    examples.append((value, template))
        return cls(examples)

    def read(self, image) -> Prediction[int]:
        try:
            image = normalize_game_screen(image)
            query = melee_damage_patch(image)
        except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        if query is None or query.std() < 5:
            return Prediction.unknown(source=self.source, reason="melee_damage_not_found")
        scores = {}
        for (value, template), template_std in zip(self.examples, self._template_stds):
            if template.shape != query.shape or template_std < 5:
                continue
            score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            scores[value] = max(scores.get(value, -1), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < .97:
            return Prediction.unknown(source=self.source, reason="melee_damage_no_match")
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
            return Prediction.unknown(source=self.source, reason="melee_damage_ambiguous")
        value, score = ranked[0]
        return Prediction.accepted(value, source=self.source, confidence=score)
