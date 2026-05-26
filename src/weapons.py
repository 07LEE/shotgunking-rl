"""Weapon preset definitions for Shotgun King reinforcement learning.

Each preset maps an integer ID to a named weapon specification.
range_limit is a (falloff_start, max_range) tuple:
  - falloff_start: distance at which damage begins to decay (full damage below this)
  - max_range: maximum effective range (shots beyond this distance have no effect)
Add new weapon entries here without modifying env.py or train.py.
"""

WEAPON_PRESETS = {
    0: {"name": "Classic", "damage": 4.0, "range_limit": (3, 5), "spread": 55.0}
}
