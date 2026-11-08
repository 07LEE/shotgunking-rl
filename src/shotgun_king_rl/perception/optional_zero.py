"""Resolve absent optional stats only on a readable, complete HUD panel."""

from .frame import PreparedFrame, read_frame
from .types import Prediction, PredictionStatus
from .visibility_proposals import propose_combat_visibility


ZERO_WHEN_NOT_DISPLAYED = ('penetration_percent', 'knockback_percent', 'melee_damage')
ZERO_RULE_SOURCE = 'hud-clean-panel-optional-zero-v1'


class ZeroWhenNotDisplayedReader:
    """A missing optional stat on a readable panel is zero; an obscured stat is unknown.

    Never reinterpret failed numeric recognition of a displayed title as zero.
    Grenade damage/count are deliberately outside this field-specific rule.
    """

    def __init__(self, reader, field, *, visibility_titles=()):
        if field not in ZERO_WHEN_NOT_DISPLAYED:
            raise ValueError('Zero rule is limited to penetration, knockback and melee')
        self.reader = reader
        self.field = field
        self.visibility_titles = tuple((name, title.copy()) for name, title in visibility_titles)
        self.source = f'{field}:clean-panel-zero-v1'

    def read(self, image):
        return self.read_prepared(PreparedFrame(image))

    def read_prepared(self, frame):
        result = read_frame(self.reader, frame)
        if frame.image is None:
            return Prediction.unknown(source=self.source, reason='invalid_image')
        status = propose_combat_visibility(frame.image, {}, self.visibility_titles)[self.field]
        if status == 'occluded':
            return Prediction.unknown(source=self.source, reason=f'{self.field}_occluded')
        if result.status is PredictionStatus.REJECTED:
            return result
        if status != 'not_displayed':
            # A reviewed direct value is still usable. Unknown title/layout
            # cannot authorize a new zero, and displayed-but-unreadable stays
            # the original unknown result.
            return result
        if result.is_accepted:
            if type(result.value) is not int or result.value != 0:
                return Prediction.rejected(None, source=self.source, reason=f'{self.field}_visibility_conflict')
        elif result.reason not in (f'{self.field.removesuffix("_percent")}_not_found',
                                   f'{self.field.removesuffix("_percent")}_no_match'):
            return result
        return Prediction.accepted(0, source=self.source)


def connect_optional_zero_readers(combat, titles, metadata):
    """Use original numeric titles collected before wrapping any readers."""
    for field, name in (('penetration_percent', 'penetration_reader'),
                        ('knockback_percent', 'knockback_reader'),
                        ('melee_damage', 'melee_reader')):
        raw = getattr(combat, name)
        wrapped = ZeroWhenNotDisplayedReader(raw, field, visibility_titles=titles)
        setattr(combat, name, wrapped)
        metadata['readers'][name].update(
            numeric_source=raw.source, source=wrapped.source,
            zero_condition='verified_readable_panel_and_not_displayed',
            zero_rule_source=ZERO_RULE_SOURCE)
    return combat
