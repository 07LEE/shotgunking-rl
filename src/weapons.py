"""Weapon preset definitions for Shotgun King reinforcement learning.

Each preset maps an integer ID to a named weapon specification.
range_limit is a (falloff_start, max_range) tuple:
  - falloff_start: distance at which damage begins to decay (full damage below this)
  - max_range: maximum effective range (shots beyond this distance have no effect)
Add new weapon entries here without modifying env.py or train.py.
"""

WEAPON_PRESETS = {
    0: {"name": "Solomon", "damage": 4.0, "range_limit": (3, 5), "spread": 55.0, "max_ammo": 2, "max_reserve_ammo": 8, "pierce_chance": 0.0},
    1: {"name": "Victoria", "damage": 5.0, "range_limit": (4, 6), "spread": 45.0, "max_ammo": 1, "max_reserve_ammo": 3, "pierce_chance": 0.0}
}
