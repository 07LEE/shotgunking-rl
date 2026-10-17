"""Board annotations independent of legacy piece IDs and runtime observations."""

MOAT_MOVEMENT = {'knight': 'jump_across', 'other_pieces': 'stop_in_moat_before_continuing'}


def board_metadata(data):
    """Validate side/terrain grids; infer only legacy piece sides when absent.

    player_king remains the controlled king; an allied rook is rook + ally.
    Missing terrain stays unknown rather than silently claiming normal ground.
    """
    board = data['board']
    sides = data.get('piece_sides')
    if sides is None:
        sides = [[None if piece == 'empty' else 'ally' if piece == 'player_king' else 'enemy'
                  for piece in row] for row in board]
    terrain = data.get('terrain')
    if terrain is None:
        terrain = [['unknown'] * 8 for _ in range(8)]
    for name, grid, allowed in (
        ('piece sides', sides, (None, 'ally', 'enemy', 'unknown')),
        ('terrain', terrain, ('unknown', 'normal', 'moat')),
    ):
        if not isinstance(grid, list) or len(grid) != 8 or any(
            not isinstance(row, list) or len(row) != 8 or any(value not in allowed for value in row)
            for row in grid
        ):
            raise ValueError(f'Invalid {name} grid')
    for r in range(8):
        for c in range(8):
            if board[r][c] == 'empty' and sides[r][c] is not None:
                raise ValueError('Empty cells cannot have a piece side')
            if board[r][c] != 'empty' and sides[r][c] is None:
                raise ValueError('Occupied cells require a piece side')
            if board[r][c] == 'player_king' and sides[r][c] != 'ally':
                raise ValueError('Controlled player king must be allied')
    return sides, terrain
