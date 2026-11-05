"""Read the grenade damage HUD from explicitly reviewed training references."""

import json
from pathlib import Path

import cv2

from ..collection_layout import annotation_paths
from .visibility import visual_combat_stats
from .visibility_proposals import propose_combat_visibility

from ..vision.layout import normalize_game_screen
from .hud import stat_patch
from .types import Prediction, PredictionStatus
from .frame import PreparedFrame, read_frame
from .named_stats import named_integer_candidates
from .numeric import numeric_token


def grenade_damage_patch(image, row_offset=256):
    """Return the grenade damage title and damage from the shifting HUD panel."""
    return stat_patch(image, row_offset=row_offset)


class GrenadeDamageReader:
    """Return visible grenade damage as an integer; other combat fields remain unavailable to policy.

    Match whole HUD rows or a reviewed title plus a reviewed integer shape.
    References must come from train; val and test are evaluation inputs only.
    Missing rows are unknown, never inferred as zero grenade damage.
    """

    source = "grenade-damage-reviewed-templates:v2"

    def __init__(self, examples=(), *, titles=(), integers=()):
        self.examples = tuple((value, template.copy()) for value, template in examples)
        self._template_stds = tuple(template.std() for _, template in self.examples)
        self.titles = tuple((name, title.copy()) for name, title in titles)
        self.integers = tuple((value, token.copy()) for value, token in integers)

    @classmethod
    def from_collection(cls, collection, *, exclude=None):
        examples = []
        titles = []
        integers = []
        for path in annotation_paths(collection):
            if exclude is not None and path.resolve() == Path(exclude).resolve():
                continue
            data = json.loads(path.read_text())
            state = data.get("screen_state") or {}
            reviews = state.get("reviews") or {}
            value = visual_combat_stats(state).get("grenade_damage")
            attack = visual_combat_stats(state).get("attack")
            valid_damage = type(value) is int and value >= 0
            valid_attack = type(attack) is int and attack >= 0
            if (
                data.get("split") != "train"
                or state.get("screen_kind") != "gameplay"
                or reviews.get("screen_kind") is not True
                or reviews.get("combat_stats") is not True
                or not (valid_damage or valid_attack)
            ):
                continue
            image_path = (path.parent / data.get("image", "")).resolve()
            if image_path.parent != (path.parent.parent / "originals").resolve():
                raise ValueError(f"{path}: invalid image path")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"{path}: unreadable image")
            image = normalize_game_screen(image, data.get("game_crop"))
            if valid_damage:
                template = grenade_damage_patch(image)
                if template is not None and template[:30].std() >= 5:
                    examples.append((value, template))
                    titles.append(('grenade_damage', template[:30]))
                    token = numeric_token(image, 256)
                    if token is not None:
                        integers.append((value, token))
            # Attack uses the same plain integer font. Range separators,
            # percentages and spread degrees are not integer references.
            if valid_attack:
                token = numeric_token(image)
                if token is not None:
                    integers.append((attack, token))
        return cls(examples, titles=titles, integers=integers)

    def has_visible_title(self, image):
        """Return None without title evidence; a visible unreadable value is not absence."""
        if not self.titles:
            return None
        for offset in (192, 256, 320):
            row = stat_patch(image, row_offset=offset, padding=8)
            if row is None:
                continue
            for _, title in self.titles:
                if title.std() >= 5 and float(cv2.matchTemplate(
                        row[:46], title, cv2.TM_CCOEFF_NORMED).max()) >= .90:
                    return True
        return False

    def read(self, image) -> Prediction[int]:
        try:
            image = normalize_game_screen(image)
            queries = [grenade_damage_patch(image, offset) for offset in (192, 256, 320)]
        except (AttributeError, IndexError, TypeError, ValueError, cv2.error):
            return Prediction.unknown(source=self.source, reason="invalid_image")
        queries = [query for query in queries if query is not None and query.std() >= 5]
        if not queries:
            return Prediction.unknown(source=self.source, reason="grenade_damage_not_found")
        scores = {}
        for query in queries:
            for (value, template), template_std in zip(self.examples, self._template_stds):
                if template.shape != query.shape or template_std < 5:
                    continue
                score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                scores[value] = max(scores.get(value, -1), score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if ranked and ranked[0][1] >= .97 and len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
            return Prediction.unknown(source=self.source, reason="grenade_damage_ambiguous")
        candidates = named_integer_candidates(image, 'grenade_damage', self.titles, self.integers)
        if len({value for value, _ in candidates}) > 1:
            return Prediction.unknown(source=self.source, reason="grenade_damage_ambiguous")
        numeric = max(candidates, key=lambda item: item[1]) if candidates else None
        if ranked and ranked[0][1] >= .97:
            value, score = ranked[0]
            if numeric is not None and numeric[0] != value:
                return Prediction.unknown(source=self.source, reason="grenade_damage_ambiguous")
            return Prediction.accepted(value, source=self.source, confidence=score)
        if numeric is not None:
            value, score = numeric
            return Prediction.accepted(value, source=self.source, confidence=score)
        return Prediction.unknown(source=self.source, reason="grenade_damage_no_match")


class RetainedGrenadeDamageReader:
    """Retain an observed damage value within one live game, never a preset.

    Offline readers stay stateless. Reuse requires a recognized clean panel
    with no damage row. Occluded/unknown panels return unknown without erasing
    memory; a visible unreadable damage row invalidates the remembered value.
    """

    source = 'grenade-damage:retained-v2'

    def __init__(self, reader, *, visibility_titles=()):
        self.reader = reader
        self.visibility_titles = tuple((name, title.copy()) for name, title in visibility_titles)
        self.reset_run()

    def reset_run(self):
        self.value = None
        self.observed_source = None
        self.observed_confidence = None
        self.used_memory = False

    def memory_info(self):
        return {'value': self.value, 'observed_source': self.observed_source,
                'observed_confidence': self.observed_confidence, 'used_memory': self.used_memory}

    def read(self, image):
        return self.read_prepared(PreparedFrame(image))

    def read_prepared(self, frame):
        self.used_memory = False
        result = read_frame(self.reader, frame)
        if result.is_accepted:
            if type(result.value) is not int or result.value < 0:
                self.reset_run()
                return Prediction.rejected(None, source=self.source, reason='invalid_grenade_damage')
            self.value = result.value
            self.observed_source, self.observed_confidence = result.source, result.confidence
            return result
        if result.status is PredictionStatus.REJECTED or result.reason == 'grenade_damage_ambiguous':
            self.reset_run()
            return result
        if (result.status is not PredictionStatus.UNKNOWN or frame.image is None
                or result.reason not in ('grenade_damage_not_found', 'grenade_damage_no_match')):
            return result
        visible = self.reader.has_visible_title(frame.image)
        if visible is True:
            self.reset_run()
            return result
        status = propose_combat_visibility(frame.image, {}, self.visibility_titles)['grenade_damage']
        if status == 'unreadable':
            self.reset_run()
        elif status == 'not_displayed' and self.value is not None:
            self.used_memory = True
            return Prediction.accepted(self.value, source=self.source)
        return Prediction.unknown(source=self.source, reason=f'grenade_damage_{status}')
