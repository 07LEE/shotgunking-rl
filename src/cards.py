"""Card specifications database for Shotgun King reinforcement learning.

This module defines the mapping between unique card identifiers, their Korean/English names,
the persistent database template image filenames, and the precise parameter effects they apply
to the player (buffs) and the enemies (debuffs) inside the Gymnasium environment.
"""

CARD_DATABASE = {
    "heavy_armor": {
        "id": 0,
        "name_ko": "판금갑주",
        "name_en": "Heavy Armor",
        "db_image_file": "cards/heavy_armor.png",
        "player_buffs": {
            "player_max_hp_bonus": 2,
        },
        "enemy_debuffs": {
            "all_enemy_hp_bonus": 1,
            "enemy_turn_speed_bonus": -1,
            "enemy_sword_damage_bonus": -1,
        }
    }
}
