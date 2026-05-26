# Shotgun King Reinforcement Learning Project

This project implements an automated gameplay and reinforcement learning system for the game Shotgun King. It processes the game screen to represent state matrices and utilizes a Deep Q-Network agent to optimize keyboard and mouse controls.

## Key Features

- Screen Capture and State Extraction: Capture the game window in real-time, segment the chessboard, and extract 8x8 grid states and threat maps.
- Real-Time Ammo Tracking: Scan UI coordinates to extract loaded and reserve ammo counts.
- Gymnasium Environment: Provides a custom, OpenAI Gymnasium-compatible environment.
- DQN Agent Training: Train a neural network using PyTorch with automated weight saving.
- Emergency Stop: Safely interrupts training when the mouse pointer is swept to the screen corner.

## Directory Structure

- src/env.py: Main Gymnasium environment linking screen capture, state extraction, and input simulation.
- src/agent.py: DQN network, replay buffer, and decision logic.
- src/analyzer.py: Image processing algorithms for grid detection, piece classification, and ammo sync.
- src/capture.py: Window geometry finder and screen grabber.
- src/input.py: Mouse and keyboard event simulation.
- src/train.py: Training loop coordinator.
- src/weapons.py: Weapon specifications and configurations.

## Installation and Execution

1. Dependencies: Install the required packages listed in pyproject.toml.
2. Training: Run the src/train.py script to start the DQN training process.
