"""Compose independent HUD readings without filling missing values."""

from dataclasses import dataclass

from .frame import PreparedFrame, read_frame
from .pipeline import Reader
from .types import CombatStatsReading, Prediction, PredictionStatus, RangeReading
from .validation import valid_combat_stats


@dataclass
class CombatStatsReader:
    """Require every field, including grenade damage at zero remaining grenades.

    A missing melee reader remains unknown until a verified reader is supplied.
    Explicitly accepted zero values are retained; no preset supplies defaults.
    """

    attack_reader: Reader[int]
    range_reader: Reader[RangeReading]
    spread_reader: Reader[int]
    penetration_reader: Reader[int]
    knockback_reader: Reader[int]
    grenade_count_reader: Reader[int]
    grenade_damage_reader: Reader[int]
    melee_reader: Reader[float] | None = None

    source = "combat-stats:composed-v1"

    def read(self, image) -> Prediction[CombatStatsReading]:
        return self.read_prepared(PreparedFrame(image))

    def read_prepared(self, frame) -> Prediction[CombatStatsReading]:
        readers = (
            ("attack", self.attack_reader),
            ("range", self.range_reader),
            ("spread_degrees", self.spread_reader),
            ("penetration_percent", self.penetration_reader),
            ("knockback_percent", self.knockback_reader),
            ("grenades_remaining", self.grenade_count_reader),
            ("grenade_damage", self.grenade_damage_reader),
            ("melee_damage", self.melee_reader),
        )
        readings = {}
        issues = []
        rejected = False
        for name, reader in readers:
            reading = (
                read_frame(reader, frame) if reader is not None else
                Prediction.unknown(source="unavailable", reason="reader_unavailable")
            )
            readings[name] = reading
            if not reading.is_accepted:
                issues.append(f"{name}:{reading.reason or reading.status.value}@{reading.source}")
                rejected |= reading.status is PredictionStatus.REJECTED
        if issues:
            reason = ";".join(issues)
            if rejected:
                return Prediction.rejected(None, source=self.source, reason=reason)
            return Prediction.unknown(source=self.source, reason=reason)

        firing_range = readings["range"].value
        if not isinstance(firing_range, RangeReading):
            return Prediction.rejected(None, source=self.source, reason="range:invalid_values")
        values = {name: reading.value for name, reading in readings.items() if name != "range"}
        stats = CombatStatsReading(
            range_min=firing_range.range_min, range_max=firing_range.range_max, **values
        )
        if not valid_combat_stats(stats):
            return Prediction.rejected(None, source=self.source, reason="invalid_values")
        # The component scores need not be calibrated on the same scale.
        return Prediction.accepted(stats, source=self.source)
