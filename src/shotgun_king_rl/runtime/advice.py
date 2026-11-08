"""Read-only, static-position advice without model weights or game input."""

import numpy as np

from ..perception.context import terrain_grid
from ..perception.types import ScreenKind
from .actions import action_destination, available_moves, select_shot_target
from .state import build_threat_matrix
from .suggestions import _format_board, _square_name


MOVE_NAMES = ('왼쪽 위', '위', '오른쪽 위', '왼쪽', '오른쪽', '왼쪽 아래', '아래', '오른쪽 아래')
LIMITATION = '현재 배치의 기본 공격 경로 기준입니다. 적 이동 순서·체력·카드 효과·산탄 명중은 예측하지 않습니다.'


def build_manual_advice(perception, *, max_ammo=None):
    """Rank static-safe one-square moves; never infer unknown observations."""
    if max_ammo is not None and (type(max_ammo) is not int or max_ammo < 1):
        raise ValueError('최대 장전 수는 양의 정수여야 합니다')
    result = {'blocked': True, 'issues': list(perception.issues), 'moves': [],
              'reload': None, 'shot': None, 'board_rows': [], 'limitation': LIMITATION}
    # Static attack paths and reload capacity do not use combat damage. Keep
    # the policy gate intact and expose movement only when combat is the sole
    # missing observation; shooting still requires complete combat evidence.
    movement_only = (perception.combat_stats is None and bool(perception.issues)
                     and all(issue.startswith('combat_stats:') for issue in perception.issues))
    result['shot_unavailable'] = movement_only
    if (((not perception.ready_for_policy or perception.issues) and not movement_only)
            or perception.screen_kind is not ScreenKind.GAMEPLAY
            or perception.board is None or perception.board_context is None
            or perception.ammo is None
            or (perception.combat_stats is None and not movement_only)):
        if not result['issues']:
            result['issues'] = ['관측이 완전하지 않습니다']
        return result
    board = np.asarray(perception.board)
    kings = np.argwhere(board == 1)
    if len(kings) != 1:
        result['issues'] = ['조종 왕 위치가 하나로 확정되지 않았습니다']
        return result
    ammo = perception.ammo
    if max_ammo is not None and ammo.loaded > max_ammo:
        result['issues'] = ['관측 장전 수가 지정한 최대 장전 수보다 큽니다']
        return result
    terrain = terrain_grid(perception.board_context)
    king = tuple(int(v) for v in kings[0])
    # Removing the original king reveals slider rays previously blocked by it.
    vacated = board.copy()
    vacated[king] = 0
    threat = build_threat_matrix(vacated, terrain)
    valid, _ = available_moves(board, threat, king, terrain=terrain)
    for action in valid:
        destination = action_destination(king, action)
        moved = vacated.copy()
        moved[destination] = 1
        after_threat = build_threat_matrix(moved, terrain)
        if after_threat[destination]:
            continue
        # Vacate the new king for the same ray check on subsequent exits.
        exits, _ = available_moves(moved, threat, destination, terrain=terrain)
        exit_count = sum(threat[action_destination(destination, candidate)] == 0 for candidate in exits)
        result['moves'].append({'action': action, 'square': _square_name(destination),
                                'direction': MOVE_NAMES[action], 'static_exits': int(exit_count)})
    result['moves'].sort(key=lambda row: (-row['static_exits'], row['action']))
    current_threat = bool(build_threat_matrix(board, terrain)[king])
    if (max_ammo is not None and ammo.loaded < max_ammo and ammo.reserve > 0
            and not current_threat):
        result['reload'] = {'shells': min(max_ammo - ammo.loaded, ammo.reserve),
                            'priority': ammo.loaded == 0}
    stats = perception.combat_stats
    target = select_shot_target(board, king, range_limit=stats.range_max, terrain=terrain) if stats is not None else None
    if ammo.loaded > 0 and target is not None:
        position = (king[0] + target.direction[0] * target.distance,
                    king[1] + target.direction[1] * target.distance)
        result['shot'] = {'square': _square_name(position), 'attack': stats.attack,
                          'distance': round(target.effective_distance, 2),
                          'spread': stats.spread_degrees}
    result.update(blocked=False, board_rows=list(_format_board(board)),
                  ammo={'loaded': ammo.loaded, 'reserve': ammo.reserve}, current_threat=current_threat)
    return result


def format_manual_advice(result):
    if result['blocked']:
        labels = {'board_context': '기물 소속·해자 상태 확인 필요',
                  'board': '기물 배치 확인 필요', 'ammo': '탄약 확인 필요',
                  'screen': '플레이 화면 확인 필요', 'combat_stats': '전투 수치 확인 필요'}
        messages = []
        for issue in result['issues']:
            message = labels.get(issue.split(':', 1)[0], issue)
            if message not in messages:
                messages.append(message)
        return '추천 보류: ' + ', '.join(messages)
    lines = ['현재 관측', *result['board_rows'],
             f'장전 {result["ammo"]["loaded"]} / 예비 {result["ammo"]["reserve"]}']
    if result['current_threat']:
        lines.append('현재 칸이 기본 공격 경로에 있습니다.')
    reload = result['reload']
    if reload and reload['priority']:
        lines.append(f'재장전 검토: {reload["shells"]}발 보충, 현재 칸에 기본 공격 경로 없음')
    lines.append('이동 후보 — 이동 후 기본 공격 경로가 없고, 주변 이동 후보가 많은 순서')
    for row in result['moves'][:3]:
        lines.append(f'  {row["direction"]} → {row["square"]}: 주변 후보 {row["static_exits"]}칸')
    if not result['moves']:
        lines.append('  해당 조건의 이동 후보 없음')
    if reload and not reload['priority']:
        lines.append(f'재장전 후보: {reload["shells"]}발 보충 가능')
    if result.get('shot_unavailable'):
        lines.append('사격 보류: 전투 수치 미확정. 이동·재장전 후보만 표시합니다.')
    shot = result['shot']
    if shot:
        lines.append(f'사격 검토: {shot["square"]}, 거리 {shot["distance"]}, 공격력 {shot["attack"]}, 탄퍼짐 {shot["spread"]}°')
        lines.append('  적 체력·명중·반격 결과가 없어 이동보다 유리한지는 미평가')
    lines.append(result['limitation'])
    return '\n'.join(lines)
