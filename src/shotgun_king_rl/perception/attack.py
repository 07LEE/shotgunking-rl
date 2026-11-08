"""Read the attack HUD from explicitly reviewed training references."""

import json
from pathlib import Path

import cv2
import numpy as np

from ..collection_layout import annotation_paths
from .visibility import visual_combat_stats

from ..vision.layout import normalize_game_screen
from .numeric import (
    accept_numeric, match_numeric_with_title, numeric_ambiguous, numeric_token, overlap_score,
    rank_numeric_with_title, token_score,
)
from .hud import stat_patch, _stat_anchor
from .types import Prediction


def attack_patch(image):
    """Return the attack title and number from the shifting HUD panel."""
    return stat_patch(image)


def digit_glyphs(token):
    """Split a complete reviewed value without discarding extra symbols."""
    if token is None:
        return []
    columns = np.flatnonzero(token.any(axis=0))
    groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
    result = []
    for group in groups:
        if not len(group):
            continue
        glyph = token[:, group[0]:group[-1] + 1]
        rows = np.flatnonzero(glyph.any(axis=1))
        glyph = glyph[rows[0]:rows[-1] + 1]
        width = round(glyph.shape[1] * 32 / glyph.shape[0])
        if not 1 <= width <= 120:
            return []
        normalized = np.zeros((40, 128), np.uint8)
        left = (128 - width) // 2
        normalized[4:36, left:left + width] = cv2.resize(glyph, (width, 32), interpolation=cv2.INTER_NEAREST)
        result.append(normalized)
    return result


def hud_digit_glyphs(image, offset=0, *, expected_count=None):
    anchor = _stat_anchor(image)
    if anchor is None:
        return []
    mask = (image[anchor + offset - 3:anchor + offset + 30, 270:354].max(axis=2) > 110).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    glyphs = []
    for label, (x, y, width, height, area) in sorted(enumerate(stats[1:], 1), key=lambda item: item[1][0]):
        # Short title remnants and the range separator are not digit strokes.
        if height < 8:
            continue
        component = (labels[y:y + height, x:x + width] == label).astype(np.uint8)
        if expected_count == 2 and width > height * 1.25:
            # Reviewed two-digit rows can join at a single antialiased pixel.
            # The HUD uses equal-width digits; the label supplies their count.
            middle = width // 2
            glyphs.extend(digit_glyphs(component[:, :middle]))
            glyphs.extend(digit_glyphs(component[:, middle:]))
        else:
            glyphs.extend(digit_glyphs(component))
    return glyphs

class AttackReader:
    """Return only attack; other combat fields remain unavailable to policy.

    Match reviewed whole rows or complete numeric shapes with matching row context.
    References must come from train; val and test are evaluation inputs only.
    """

    source = "attack-reviewed-templates:v5"

    def __init__(self, examples=(), *, numeric_examples=(), dim_numeric_examples=(), digit_examples=()):
        self.examples = tuple((value, template.copy()) for value, template in examples)
        self._template_stds = tuple(template.std() for _, template in self.examples)
        self.numeric_examples = tuple((value, token.copy(), row.copy()) for value, token, row in numeric_examples)

        self.dim_numeric_examples = tuple((value, token.copy(), row.copy()) for value, token, row in dim_numeric_examples)
        self.digit_examples = tuple((value, token.copy()) for value, token in digit_examples)

    @classmethod
    def from_collection(cls, collection):
        examples = []
        numeric_examples = []
        dim_numeric_examples = []
        digit_examples = []
        for path in annotation_paths(collection):
            data = json.loads(path.read_text())
            state = data.get("screen_state") or {}
            reviews = state.get("reviews") or {}
            value = visual_combat_stats(state).get("attack")
            if (data.get('split') == 'train' and state.get('screen_kind') == 'gameplay'
                    and reviews.get('screen_kind') is True and reviews.get('combat_stats') is True):
                # Other explicitly reviewed HUD rows share the same digit font.
                stats = visual_combat_stats(state)
                reference_path = (path.parent / data.get('image', '')).resolve()
                if reference_path.parent != (path.parent.parent / 'originals').resolve():
                    raise ValueError(f'{path}: invalid image path')
                reference = cv2.imread(str(reference_path))
                if reference is not None:
                    reference = normalize_game_screen(reference, data.get('game_crop'))
                    for offset, fields, separator in ((0, ('attack',), ''),
                            (64, ('range_min', 'range_max'), '-'), (128, ('spread_degrees',), '')):
                        values = [stats.get(field) for field in fields]
                        if not all(type(v) is int and v >= 0 for v in values):
                            continue
                        text = separator.join(map(str, values))
                        text = text.replace('-', '')
                        glyphs = hud_digit_glyphs(reference, offset, expected_count=len(text))
                        if len(glyphs) == len(text):
                            digit_examples.extend((int(char), glyph) for char, glyph in zip(text, glyphs))
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
                dim_token = numeric_token(image, 0, threshold=110)
                if dim_token is not None:
                    dim_numeric_examples.append((examples[-1][0], dim_token, template))
        return cls(examples, numeric_examples=numeric_examples, dim_numeric_examples=dim_numeric_examples,
                   digit_examples=digit_examples)

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
            context = stat_patch(image, 0, padding=8)
            bright = rank_numeric_with_title(numeric_token(image, 0), context, self.numeric_examples)
            # Dim strokes may recover a missing value, never overrule conflicting bright evidence.
            if numeric_ambiguous(bright):
                return Prediction.unknown(source=self.source, reason="attack_ambiguous")
            numeric = accept_numeric(bright)
            if numeric is None:
                numeric = match_numeric_with_title(
                    numeric_token(image, 0, threshold=110), context,
                    self.dim_numeric_examples, scorer=overlap_score)
            if numeric is not None:
                value, score = numeric
                return Prediction.accepted(value, source=self.source, confidence=score)
            glyphs = hud_digit_glyphs(image)
            titles = [row[:30] for _, row in self.examples if row[:30].std() >= 5]
            if (len(glyphs) == 1 and titles and context is not None
                    and max(float(cv2.matchTemplate(context[:46], title, cv2.TM_CCOEFF_NORMED).max())
                            for title in titles) >= .90):
                scores = {}
                for value, token in self.digit_examples:
                    if overlap_score(glyphs[0], token) < .75:
                        continue
                    scores[value] = max(scores.get(value, -1), token_score(glyphs[0], token))
                match = accept_numeric(sorted(scores.items(), key=lambda item: item[1], reverse=True))
                if match is not None:
                    return Prediction.accepted(match[0], source=self.source, confidence=match[1])
            return Prediction.unknown(source=self.source, reason="attack_no_match")
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < .04:
            return Prediction.unknown(source=self.source, reason="attack_ambiguous")
        value, score = ranked[0]
        return Prediction.accepted(value, source=self.source, confidence=score)
