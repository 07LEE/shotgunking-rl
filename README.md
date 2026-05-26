# Shotgun King Reinforcement Learning Project

This project implements an automated gameplay and reinforcement learning system for the game Shotgun King. It processes the live game window via screen capture to construct state representations, which are then passed to a Deep Q-Network agent to optimize keyboard and mouse control simulations.

## Project Architecture

The system operates in a closed loop across four main phases:
1. Capture: Active screen capture of the window using mss.
2. Analyze: Slicing the chessboard area, detecting pieces, and reading UI status.
3. Decide: DQN agent inputs the processed state vector and outputs an action.
4. Control: Simulating inputs through PyAutoGUI mouse clicks or key presses.

## Gymnasium Environment Specification

The custom environment ShotgunKingEnv maps game frames to the OpenAI Gymnasium standard.

- Observation Space: 133-dimensional float vector
  - Board Matrix (dimensions 0-63): 8x8 flattened grid representing piece coordinates (0: Empty, 1: Player King, 2: Pawn, 3: Knight, 4: Bishop, 5: Rook, 6: Queen/King).
  - Threat Matrix (dimensions 64-127): 8x8 flattened grid projecting active enemy attack lines.
  - Ammo Status (dimensions 128-129): Real-time sync of loaded and reserve ammo counts.
  - Weapon Specifications (dimensions 130-132): Flat values representing weapon damage, range limit, and spread.
- Action Space: 10 discrete actions
  - Actions 0-7: 8-directional movement clicks relative to the Player King.
  - Action 8: Manual reload keypress.
  - Action 9: Directed shoot click targetting the closest eligible enemy.

## Image Processing Pipeline

The analyzer module extracts state vectors from raw BGR screenshots.
- Board Localization: Crops the chessboard area using fixed screen coordinates and resizes the region to 520x520 pixels.
- Cell Slicing: Divides the board into 64 unique 65x65 pixel cell patches.
- Feature Classification: Employs template matching (resized to 45x45 grayscale signals) for high-accuracy piece identification. Falls back to contour-based geometry analysis when templates are missing.
- Ammo OCR: Counts the number of active red pixels at predefined UI locations to determine ammunition counts.

## Model Details

- Policy Network: MLP structure consisting of a 133-size input layer, a 128-unit linear layer, a 64-unit linear layer, and a 10-unit output layer.
- Optimization: Backpropagation via MSE loss and Adam optimizer. Trains on experience tuples sampled from a ReplayBuffer.

## Installation and Execution

### Prerequisites

- Display Configuration: The game must be run in windowed mode at 1280x720 resolution. Do not minimize, cover, or obstruct the game window during execution.
- Emergency Stop: Sweep the mouse pointer to the top-left corner of the screen (coordinates 0,0) to instantly interrupt training.

### Setup and Running

1. Installation: Install the required packages listed in pyproject.toml.
2. Execution: Run `src/train.py` while the game is running and visible on the screen.
