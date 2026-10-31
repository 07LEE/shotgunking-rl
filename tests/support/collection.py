"""Synthetic collection inputs for persistence and export contract tests."""

import numpy as np


def game_screen():
    image = np.full((720, 1280, 3), (40, 70, 90), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            image[119+row*64:119+(row+1)*64, 383+col*64:383+(col+1)*64] = (
                (170, 180, 220) if (row+col) % 2 else (90, 100, 150)
            )
    return image


def piece_annotation():
    board = [['empty'] * 8 for _ in range(8)]
    board[0][0], board[0][6] = 'white_king', 'knight'
    board[1][1], board[2][2], board[7][3] = 'pawn', 'bishop', 'player_king'
    return {'schema': 'piece-cells-v1', 'image': 'screen.png', 'session_id': 'fixture',
            'split': 'train', 'confirmed': False, 'floor': None, 'exclude_cells': [],
            'game_crop': [0, 0, 1280, 720], 'board': board}
