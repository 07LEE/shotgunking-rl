"""Card specifications database for Shotgun King reinforcement learning.

This module defines the mapping between unique card identifiers, their Korean/English names,
the persistent database template image filenames, and the precise parameter effects they apply
to the player (buffs) and the enemies (debuffs) inside the Gymnasium environment.
"""

CARD_DATABASE = {
    "heavy_armor": {
        "id": 0,
        "type": "debuff",
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
    },
    "court_meeting": {
        "id": 1,
        "type": "debuff",
        "name_ko": "궁정회의",
        "name_en": "Court Meeting",
        "db_image_file": "cards/court_meeting.png",
        "player_buffs": {},
        "enemy_debuffs": {
            "enemy_knight_count_bonus": 2,
            "enemy_bishop_count_bonus": 1,
            "enemy_rook_count_bonus": 1,
            "enemy_turn_speed_bonus": -1,
        }
    },
    "petition_discrimination": {
        "id": 2,
        "type": "buff",
        "name_ko": "탄원서 차별",
        "name_en": "Petition Discrimination",
        "db_image_file": "cards/petition_discrimination.png",
        "player_buffs": {
            "max_simultaneous_enemy_types_limit": 2,
        },
        "enemy_debuffs": {}
    },
    "daring_operation": {
        "id": 3,
        "type": "buff",
        "name_ko": "대담한 작전",
        "name_en": "Daring Operation",
        "db_image_file": "cards/daring_operation.png",
        "player_buffs": {
            "allow_debuff_card_selection": True,
        },
        "enemy_debuffs": {}
    }
}
