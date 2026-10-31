"""Read the melee damage HUD from explicitly reviewed training references."""

import json
from pathlib import Path

import cv2

from ..vision.layout import normalize_game_screen
from .hud import reviewed_zero_score, stat_context_patch, stat_patch
from .types import Prediction


def melee_damage_patch(image):
    """Return the melee damage title and damage from the shifting HUD panel."""
    return stat_patch(image, row_offset=192)


class MeleeDamageReader:
    """Return visible melee damage as an integer; other combat fields remain unavailable to policy.

    This recognizes reviewed whole HUD values, not arbitrary unseen numbers.
    References must come from train; val and test are evaluation inputs only.
    Positive references require a verified matching title. Zero means melee is
    unavailable and requires a reviewed matching panel and exact optional row.
    A missing or unreadable row alone never proves that melee is unavailable.
    """

    source = "melee-damage-reviewed-templates:v2"

    def __init__(self, examples=(), zero_examples=()):
        self.examples = tuple((value, template.copy()) for value, template in examples
                              if type(value) is int and value > 0)
        self._template_stds = tuple(template.std() for _, template in self.examples)
        self.zero_examples = tuple((context.copy(), row.copy()) for context, row in zero_examples)
        self._zero_stds = tuple(context.std() for context, _ in self.zero_examples)

    @classmethod
    def from_collection(cls, collection, *, title_reference=None, exclude=None):
        """Load reviewed train zeros and optionally verified positive references.

        title_reference is the top 30 rows of melee_damage_patch. It must
        identify the title, rather than the number or the whole HUD panel.
        """
        if title_reference is not None and (
                title_reference.shape != (30, 84) or title_reference.std() < 5):
            raise ValueError("A visible melee title reference is required")
        examples = []
        zero_examples = []
        for path in sorted((Path(collection) / "sessions").glob("*/annotations/*.json")):
            if exclude is not None and path.resolve() == Path(exclude).resolve():
                continue
            data = json.loads(path.read_text())
            state = data.get("screen_state") or {}
            reviews = state.get("reviews") or {}
            value = (state.get("combat_stats") or {}).get("melee_damage")
            if (
                data.get("split") != "train"
                or state.get("screen_kind") != "gameplay"
                or reviews.get("screen_kind") is not True
                or reviews.get("combat_stats") is not True
                or type(value) is not int or value < 0
                or (value > 0 and title_reference is None)
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
            if value == 0:
                context = stat_context_patch(image)
                if template is not None and context is not None and context.std() >= 5:
                    zero_examples.append((context, template))
            elif template is not None and template[:30].std() >= 5:
                title_score = float(cv2.matchTemplate(
                    template[:30], title_reference, cv2.TM_CCOEFF_NORMED
                )[0, 0])
                if title_score >= .97:
                    examples.append((value, template))
        return cls(examples, zero_examples)

    def read(self, image) -> Prediction[int]:
        try:
            image = normalize_game_screen(image)
            query = melee_damage_patch(image)
        except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        has_features = query is not None and query.std() >= 5
        if not has_features and not self.zero_examples:
            return Prediction.unknown(source=self.source, reason="melee_damage_not_found")
        scores = {}
        for (value, template), template_std in zip(self.examples, self._template_stds):
            if not has_features or template.shape != query.shape or template_std < 5:
                continue
            score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            scores[value] = max(scores.get(value, -1), score)
        zero_score = reviewed_zero_score(image, query, self.zero_examples, self._zero_stds)
        if zero_score >= 0:
            scores[0] = max(scores.get(0, -1), zero_score)
        if not scores and not has_features:
            return Prediction.unknown(source=self.source, reason="melee_damage_not_found")
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < .97:
            return Prediction.unknown(source=self.source, reason="melee_damage_no_match")
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
            return Prediction.unknown(source=self.source, reason="melee_damage_ambiguous")
        value, score = ranked[0]
        return Prediction.accepted(value, source=self.source, confidence=score)
