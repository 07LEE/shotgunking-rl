"""Synthetic board context, valid combat values and reader doubles."""

import numpy as np

from shotgun_king_rl.perception import BoardContextReading, CombatStatsReading, CombatStatsReader, Prediction, RangeReading


class NormalContextReader:
    def __init__(self, board_source):
        self.board_source = board_source

    def read(self, image):
        board = self.board_source() if callable(self.board_source) else self.board_source
        sides = np.where(board == 0, 0, np.where(board == 1, 1, 2)).astype(np.int8)
        return Prediction.accepted(BoardContextReading(sides, False), source='context:test')


VALID_COMBAT_STATS = CombatStatsReading(
    attack=4,
    range_min=3,
    range_max=5,
    spread_degrees=65,
    penetration_percent=0,
    knockback_percent=0,
    grenades_remaining=0,
    grenade_damage=0,
    melee_damage=0,
)


class FakeReader:
    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = 0

    def read(self, image):
        self.calls += 1
        return self.prediction


class StubReader:
    def __init__(self, value):
        self.prediction = Prediction.accepted(value, source="stub:test")
        self.frames = []

    def read_prepared(self, frame):
        self.frames.append(frame)
        return self.prediction


def make_reader():
    return CombatStatsReader(
        StubReader(4), StubReader(RangeReading(3, 5)), StubReader(65),
        StubReader(0), StubReader(0), StubReader(0), StubReader(2), StubReader(0),
    )
