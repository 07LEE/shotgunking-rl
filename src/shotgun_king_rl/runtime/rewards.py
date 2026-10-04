"""Pure reward calculations for the legacy environment."""

from dataclasses import dataclass

import numpy as np

from .actions import ShotTarget


@dataclass(frozen=True)
class RewardResult:
    """Calculated reward and the facts used to explain it."""

    reward: float
    killed_enemies: int
    expected_damage: float
    expected_knockback_reward: float
    threatened: bool
    melee_kill: bool
    invalid_shot_attempt: bool


def _ranged_expected_damage(damage, distance, falloff_start, range_limit, spread):
    if distance <= falloff_start:
        distance_factor = 1.0
    else:
        falloff_width = float(range_limit - falloff_start)
        distance_factor = 1.0 - 0.5 * (
            float(distance - falloff_start) / falloff_width
        ) if falloff_width > 0 else 1.0
    range_value = float(range_limit)
    hit_probability = max(
        0.2,
        1.0 - (spread / 120.0) * (float(distance - 1) / range_value),
    ) if range_value > 0 else 1.0
    return damage * distance_factor * hit_probability


def _second_target_distance(board, king_position, target, range_limit):
    row_offset, col_offset = target.direction
    for distance in range(target.distance + 1, int(range_limit) + 1):
        row = king_position[0] + row_offset * distance
        col = king_position[1] + col_offset * distance
        if not (0 <= row < 8 and 0 <= col < 8):
            break
        if board[row, col] >= 2:
            return distance
        if board[row, col] == 1:
            break
    return None


def calculate_step_reward(
    original_action,
    executed_action,
    previous_enemy_count,
    shot_board,
    current_board,
    current_threat,
    king_position,
    shot_target: ShotTarget | None,
    valid_shot_target,
    shot_was_melee,
    damage,
    melee_damage,
    melee_kill_extra_turn,
    falloff_start,
    range_limit,
    spread,
    pierce_chance,
    knockback_chance,
    target_hp,
    promotion_detected,
):
    """Calculate the legacy step reward from already observed action facts."""
    current_enemy_count = int(np.sum(current_board >= 2))
    killed_enemies = max(0, int(previous_enemy_count) - current_enemy_count)
    if executed_action != 9:
        killed_enemies = 0

    expected_damage = 0.0
    knockback_reward = 0.0
    attempted_shot = original_action == 9 or executed_action == 9
    if attempted_shot and valid_shot_target and shot_target is not None:
        if shot_was_melee:
            expected_damage = melee_damage
            if melee_kill_extra_turn and melee_damage >= target_hp:
                expected_damage += 1.5
        else:
            expected_damage = _ranged_expected_damage(
                damage,
                shot_target.distance,
                falloff_start,
                range_limit,
                spread,
            )

        if pierce_chance > 0.0:
            second_distance = _second_target_distance(
                shot_board,
                king_position,
                shot_target,
                range_limit,
            )
            if second_distance is not None:
                expected_damage += pierce_chance * _ranged_expected_damage(
                    damage,
                    second_distance,
                    falloff_start,
                    range_limit,
                    spread,
                )

        if knockback_chance > 0.0:
            row_offset, col_offset = shot_target.direction
            behind_row = king_position[0] + row_offset * (shot_target.distance + 1)
            behind_col = king_position[1] + col_offset * (shot_target.distance + 1)
            out_of_bounds = not (0 <= behind_row < 8 and 0 <= behind_col < 8)
            knockback_reward = knockback_chance * (2.0 if out_of_bounds else 0.4)

    reward = -0.1
    if killed_enemies > 0:
        reward += killed_enemies * 2.0
        reward += expected_damage * 0.3
        reward += knockback_reward
    elif executed_action == 9:
        if valid_shot_target:
            reward += expected_damage * 0.3 + knockback_reward
        else:
            reward -= 0.8

    invalid_shot_attempt = (
        original_action == 9
        and executed_action != 9
        and not valid_shot_target
    )
    if invalid_shot_attempt:
        reward -= 0.8

    king_positions = np.argwhere(current_board == 1)
    threatened = bool(
        len(king_positions) > 0
        and current_threat[tuple(king_positions[0])] == 1
    )
    if threatened:
        reward -= 1.5
    if promotion_detected:
        reward -= 3.0

    return RewardResult(
        reward=reward,
        killed_enemies=killed_enemies,
        expected_damage=expected_damage,
        expected_knockback_reward=knockback_reward,
        threatened=threatened,
        melee_kill=executed_action == 9 and killed_enemies > 0 and shot_was_melee,
        invalid_shot_attempt=invalid_shot_attempt,
    )
