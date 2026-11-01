"""Read the remaining grenades HUD from explicitly reviewed training references."""

import json
from pathlib import Path

import cv2

from .visibility import visual_combat_stats
import numpy as np

from ..vision.layout import normalize_game_screen
from .types import Prediction


def grenade_count_patch(image):
    """Return the two calibrated grenade slots beside the reserve ammo HUD."""
    # The bottom two rows of the old crop included pixels below the slots.
    return cv2.cvtColor(image[78:110, 592:642], cv2.COLOR_BGR2GRAY)


def grenade_context_patch(image):
    """Keep the visible ammunition HUD as evidence for reviewed empty slots."""
    return cv2.cvtColor(image[34:112, 382:706], cv2.COLOR_BGR2GRAY)


class GrenadeCountReader:
    """Return a reviewed grenade-slot count; other combat fields remain unavailable to policy.

    This recognizes reviewed whole HUD values, not arbitrary unseen numbers.
    References must come from train; val and test are evaluation inputs only.
    Featureless slots require a reviewed zero and matching visible HUD context.
    """

    source = "grenade-count-reviewed-templates:v2"

    def __init__(self, examples=(), zero_examples=()):
        self.examples = tuple((value, template.copy()) for value, template in examples)
        self._template_stds = tuple(template.std() for _, template in self.examples)
        self.zero_examples = tuple((context.copy(), slots.copy()) for context, slots in zero_examples)
        self._zero_stds = tuple(context.std() for context, _ in self.zero_examples)

    @classmethod
    def from_collection(cls, collection):
        examples = []
        zero_examples = []
        for path in sorted((Path(collection) / "sessions").glob("*/annotations/*.json")):
            data = json.loads(path.read_text())
            state = data.get("screen_state") or {}
            reviews = state.get("reviews") or {}
            value = visual_combat_stats(state).get("grenades_remaining")
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
            template = grenade_count_patch(image)
            if template is not None and template.std() >= 5:
                examples.append((value, template))
            elif value == 0 and template is not None:
                context = grenade_context_patch(image)
                if context.std() >= 5:
                    zero_examples.append((context, template))
        return cls(examples, zero_examples)

    def read(self, image) -> Prediction[int]:
        try:
            image = normalize_game_screen(image)
            query = grenade_count_patch(image)
        except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        if query is None or query.std() < 5:
            # A blank crop alone is never a zero. Require the reviewed slots
            # within one grayscale level and visible surrounding HUD from a
            # reviewed train zero. Shape changes must still stay unknown.
            context = grenade_context_patch(image)
            scores = [float(cv2.matchTemplate(context, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
                      for (reference, slots), deviation in zip(self.zero_examples, self._zero_stds)
                      if deviation >= 5 and context.shape == reference.shape
                      and context.std() >= 5 and query.shape == slots.shape
                      and np.max(np.abs(query.astype(np.int16) - slots.astype(np.int16))) <= 1]
            if scores and max(scores) >= .97:
                return Prediction.accepted(0, source=self.source, confidence=max(scores))
            return Prediction.unknown(source=self.source, reason="grenades_remaining_not_found")
        scores = {}
        for (value, template), template_std in zip(self.examples, self._template_stds):
            if template.shape != query.shape or template_std < 5:
                continue
            score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            scores[value] = max(scores.get(value, -1), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < .97:
            return Prediction.unknown(source=self.source, reason="grenades_remaining_no_match")
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
            return Prediction.unknown(source=self.source, reason="grenades_remaining_ambiguous")
        value, score = ranked[0]
        return Prediction.accepted(value, source=self.source, confidence=score)
