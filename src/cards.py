"""Card specifications database for Shotgun King reinforcement learning.

This module defines the mapping between unique card identifiers, their Korean/English names,
the persistent database template image filenames, and the precise parameter effects they apply
to the player (buffs) and the enemies (debuffs) inside the Gymnasium environment.
"""

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


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
    },
    "poison": {
        "id": 4,
        "type": "buff",
        "name_ko": "암살용 독",
        "name_en": "Poison",
        "db_image_file": "cards/poison.png",
        "player_buffs": {},
        "enemy_debuffs": {
            "boss_hp_bonus": -1,
            "queen_turn_limit": 1.0,
            "queen_turn_limit_duration": 15,
        }
    },
    "countdown": {
        "id": 5,
        "type": "debuff",
        "name_ko": "카운트다운",
        "name_en": "Countdown",
        "db_image_file": "cards/countdown.png",
        "player_buffs": {},
        "enemy_debuffs": {
            "trigger_enemy_count_limit": 6,
            "defeat_turn_limit": 12,
        }
    },
    "hungry_rats": {
        "id": 6,
        "type": "buff",
        "name_ko": "굶주린 쥐떼",
        "name_en": "Hungry Rats",
        "db_image_file": "cards/hungry_rats.png",
        "player_buffs": {
            "damage_on_kill_random_enemy": 1.0,
        },
        "enemy_debuffs": {}
    },
    "royal_guard": {
        "id": 7,
        "type": "debuff",
        "name_ko": "근위병",
        "name_en": "Royal Guard",
        "db_image_file": "cards/royal_guard.png",
        "player_buffs": {},
        "enemy_debuffs": {
            "enemy_knight_hp_bonus": 1,
            "king_immortal_while_knight_present": True,
        }
    }
}


def select_best_card_pair(img, cards_dir="data/cards", rank=1, player_hp=None, stats_path="data/card_stats.json"):
    """Analyze current screen, match active cards, and return the optimal pair choice ('top' or 'bottom').

    Dynamic scoring applies three correction layers in order:
      1. Rank-based debuff penalty scaling (higher rank -> larger debuff penalty).
      2. Player HP urgency scaling (low HP -> re-weight offensive cards).
      3. Win-rate-based correction loaded from stats_path JSON (min 5 samples required).

    Args:
        img: 1280x720 BGR screen screenshot image.
        cards_dir: Path to directory containing card template PNGs.
        rank: Current story mode difficulty rank (1-based integer).
        player_hp: Current player HP. None disables urgency correction.
        stats_path: Path to JSON file storing per-card win/lose statistics.

    Returns:
        Tuple of (choice, detected_cards) where choice is 'top' or 'bottom'
        and detected_cards is a dict mapping slot names to matched card keys.
    """
    if img is None or cv2 is None or np is None:
        return "top", {}

    # Define crop coordinate boundaries for 4 card slots
    card_regions = {
        "top_left": {"x1": 540, "x2": 624, "y1": 216, "y2": 332},
        "top_right": {"x1": 651, "x2": 735, "y1": 216, "y2": 332},
        "bottom_left": {"x1": 540, "x2": 624, "y1": 416, "y2": 532},
        "bottom_right": {"x1": 651, "x2": 735, "y1": 416, "y2": 532},
    }

    # Load reference templates from cards_dir
    import os
    import json
    templates = {}
    if os.path.exists(cards_dir):
        for fn in os.listdir(cards_dir):
            if fn.endswith(".png"):
                card_key = fn.replace(".png", "")
                t_img = cv2.imread(os.path.join(cards_dir, fn), cv2.IMREAD_GRAYSCALE)
                if t_img is not None:
                    templates[card_key] = cv2.resize(t_img, (84, 116))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_cards = {}
    for name, coords in card_regions.items():
        x1, x2 = coords["x1"], coords["x2"]
        y1, y2 = coords["y1"], coords["y2"]
        patch = cv2.resize(gray[y1:y2, x1:x2], (84, 116))

        best_score = -1.0
        best_key = None
        for key, tmpl in templates.items():
            res = cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res[0][0])
            if score > best_score:
                best_score = score
                best_key = key

            detected_cards[name] = best_key if best_score >= 0.65 else None

    print(f"DQN Cards: Detected card layout -> {detected_cards}")

    # Define base heuristic score weight mapping (player buffs > 0, enemy debuffs < 0)
    card_scores = {
        "heavy_armor": -8.0,
        "court_meeting": -6.0,
        "petition_discrimination": 4.0,
        "daring_operation": 2.0,
        "poison": 5.0,
        "countdown": -4.0,
        "hungry_rats": 6.0,
        "royal_guard": -12.0,
    }

    # Stage 1: Rank-based debuff penalty scaling
    debuff_scale = 1.0 + (rank - 1) * 0.1
    for key in ("heavy_armor", "court_meeting", "countdown", "royal_guard"):
        card_scores[key] = card_scores[key] * debuff_scale
    print(f"DQN Cards: Rank {rank} -> debuff_scale={debuff_scale:.2f}")

    # Stage 2: Player HP urgency correction
    if player_hp is not None and player_hp <= 1:
        card_scores["poison"] = 8.0
        card_scores["countdown"] = -1.0
        print(f"DQN Cards: Low HP ({player_hp}) urgency correction applied.")

    # Stage 3: Win-rate-based score correction from accumulated stats file
    if os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            for key in list(card_scores.keys()):
                entry = stats.get(key, {})
                total = entry.get("win", 0) + entry.get("lose", 0)
                if total >= 5:
                    win_rate = entry["win"] / total
                    correction = (win_rate - 0.5) * 4.0
                    card_scores[key] += correction
            print(f"DQN Cards: Win-rate stats loaded from {stats_path}.")
        except Exception as e:
            print(f"DQN Cards: Failed to load card stats: {e}")

    val_top = card_scores.get(detected_cards.get("top_left"), 0.0) + card_scores.get(detected_cards.get("top_right"), 0.0)
    val_bottom = card_scores.get(detected_cards.get("bottom_left"), 0.0) + card_scores.get(detected_cards.get("bottom_right"), 0.0)
    print(f"DQN Cards: Evaluation -> Top Pair: {val_top:.2f}, Bottom Pair: {val_bottom:.2f}")

    choice = "top" if val_top >= val_bottom else "bottom"
    return choice, detected_cards


def update_card_stats(detected_cards, chosen, outcome, stats_path="data/card_stats.json"):
    """Record win/lose outcome for each card in the chosen pair to the stats file.

    Args:
        detected_cards: Dict mapping slot names ('top_left', etc.) to matched card keys.
        chosen: 'top' or 'bottom' indicating which pair was selected this episode.
        outcome: 'win' or 'lose' for the episode result.
        stats_path: Path to the JSON statistics file.
    """
    import json
    import os

    stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except Exception:
            stats = {}

    # Determine cards in the chosen pair
    if chosen == "top":
        slot_keys = ["top_left", "top_right"]
    else:
        slot_keys = ["bottom_left", "bottom_right"]

    for slot in slot_keys:
        card_key = detected_cards.get(slot)
        if card_key is None:
            continue
        if card_key not in stats:
            stats[card_key] = {"win": 0, "lose": 0}
        stats[card_key][outcome] = stats[card_key].get(outcome, 0) + 1

    try:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"DQN Cards: Updated card stats -> {stats_path} (outcome: {outcome})")
    except Exception as e:
        print(f"DQN Cards: Failed to write card stats: {e}")


