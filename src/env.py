"""OpenAI Gymnasium custom environment for Shotgun King reinforcement learning.

This module combines screen capture (state observation) and input simulation
(action execution) into a unified Gymnasium-compatible learning environment.
"""

import os
import time

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

try:
    import pyautogui
except ImportError:
    pyautogui = None

from capture import capture_screen
from input import click_relative_in_window, press_key
from analyzer import get_state_matrix, check_retry_popup


class ShotgunKingEnv(gym.Env if gym is not None else object):
    """Custom Gymnasium environment for interacting with Shotgun King."""

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, window_title="Shotgun King", max_steps=100):
        """Initializes the environment state and spaces.

        Args:
            window_title: Title of the target game window.
            max_steps: Maximum steps allowed per episode before truncation.
        """
        super().__init__()
        self.window_title = window_title
        self.max_steps = max_steps
        self.current_step = 0
        self.current_state = None

        # Define Observation Space: 1D flat vector of size 133
        # 128 dimensions from board and threat matrix, plus 2 dimensions for ammo stats,
        # and 3 dimensions for weapon specifications (damage, range, spread)
        self.observation_space = spaces.Box(
            low=0, high=90, shape=(133,), dtype=np.float32
        )

        # Ammo Tracking
        self.loaded_ammo = 2
        self.reserve_ammo = 8

        # Weapon Specifications
        self.damage = 4.0
        self.range_limit = 3.0
        self.spread = 34.0

        # Define Action Space: Discrete actions
        # 0: Move Up-Left,  1: Move Up,    2: Move Up-Right
        # 3: Move Left,                     4: Move Right
        # 5: Move Down-Left,6: Move Down,  7: Move Down-Right
        # 8: Reload ('r' key)
        # 9: Shoot (center screen action click)
        self.action_space = spaces.Discrete(10)

        # Pre-calculated relative offset coordinates for 8-way directional clicks
        # Assuming a standard central region relative clicks
        self.direction_offsets = {
            0: (540, 260),  # Up-Left
            1: (640, 260),  # Up
            2: (740, 260),  # Up-Right
            3: (540, 360),  # Left
            4: (740, 360),  # Right
            5: (540, 460),  # Down-Left
            6: (640, 460),  # Down
            7: (740, 460),  # Down-Right
        }

    def _check_emergency_stop(self):
        """Checks if the mouse cursor is located in the top-left corner (0,0) of the screen.

        Raises:
            KeyboardInterrupt: If the mouse cursor is at or near (0,0).
        """
        if pyautogui is not None:
            mx, my = pyautogui.position()
            if mx <= 10 and my <= 10:
                print("\n=== DQN Emergency Stop: Mouse corner sweep detected! Halting training immediately. ===")
                raise KeyboardInterrupt("DQN Emergency Stop: User swept mouse to the corner (0, 0).")

    def _wait_for_equilibrium(self, max_wait=5.0):
        """적의 턴 애니메이션이 완료되고 체스판 상태가 완전히 고정되어 플레이어 턴이 정착될 때까지 대기합니다.

        Args:
            max_wait: 최대 대기 시간(초).

        Returns:
            정적 평형에 도달한 최종 8x8 상태 행렬.
        """
        start_time = time.time()
        prev_state = self._get_obs()
        
        while time.time() - start_time < max_wait:
            time.sleep(0.2)
            curr_state = self._get_obs()
            
            # 연속된 2개의 캡처본 매트릭스가 완벽히 일치하여 정지 상태에 도달했을 때
            if np.array_equal(prev_state, curr_state):
                return curr_state
                
            prev_state = curr_state
            
        print("Warning: Turn equilibrium detection timed out. Proceeding with current observation.")
        return prev_state

    def _get_threat_matrix(self, state):
        """Calculates an 8x8 binary threat matrix projecting all enemy check line rays.

        Args:
            state: Current 8x8 board state representation.

        Returns:
            An 8x8 numpy array where 1 represents a dangerous check/attack zone.
        """
        threat = np.zeros((8, 8), dtype=np.int32)
        
        # Directions for check rays
        straight_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        diagonal_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # Directions for Knight L-shapes
        knight_offsets = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        # Scan the board for all enemy pieces (values >= 2 represent different enemy types)
        for er in range(8):
            for ec in range(8):
                piece = state[er, ec]
                if piece < 2:
                    continue
                    
                # 1. Pawn (2): Attacks diagonally down by 1 tile (which is row + 1 on screen)
                if piece == 2:
                    for dc in [-1, 1]:
                        tr, tc = er + 1, ec + dc
                        if 0 <= tr < 8 and 0 <= tc < 8:
                            threat[tr, tc] = 1
                            
                # 2. Knight (3): Attacks 8 L-shape coordinates
                elif piece == 3:
                    for dr, dc in knight_offsets:
                        tr, tc = er + dr, ec + dc
                        if 0 <= tr < 8 and 0 <= tc < 8:
                            threat[tr, tc] = 1
                            
                # 3. Bishop (4): Radial diagonal rays (blocked by any piece)
                elif piece == 4:
                    for dr, dc in diagonal_directions:
                        for dist in range(1, 8):
                            tr, tc = er + dr * dist, ec + dc * dist
                            if 0 <= tr < 8 and 0 <= tc < 8:
                                threat[tr, tc] = 1
                                if state[tr, tc] != 0:
                                    break
                            else:
                                break
                                
                # 4. Rook (5): Radial straight rays (blocked by any piece)
                elif piece == 5:
                    for dr, dc in straight_directions:
                        for dist in range(1, 8):
                            tr, tc = er + dr * dist, ec + dc * dist
                            if 0 <= tr < 8 and 0 <= tc < 8:
                                threat[tr, tc] = 1
                                if state[tr, tc] != 0:
                                    break
                            else:
                                break
                                
                # 5. Queen / King (6): Both straight and diagonal rays (blocked by any piece)
                elif piece == 6:
                    for dr, dc in straight_directions + diagonal_directions:
                        for dist in range(1, 8):
                            tr, tc = er + dr * dist, ec + dc * dist
                            if 0 <= tr < 8 and 0 <= tc < 8:
                                threat[tr, tc] = 1
                                if state[tr, tc] != 0:
                                    break
                            else:
                                break
                                
        return threat

    def _get_obs(self):
        self._check_emergency_stop()
        """Captures screen and returns a 133-dimensional flat observation vector."""
        image_path = "data/screenshot.png"
        
        # Ensure fresh screen capture
        capture_screen(output_path=image_path, window_title=self.window_title)
        
        if cv2 is not None and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                # Call state extractor to return 8x8 chessboard array
                state = get_state_matrix(img).astype(np.float32)
                threat = self._get_threat_matrix(state).astype(np.float32)
                flat_obs = np.concatenate([state.flatten(), threat.flatten()])
                ammo_obs = np.array([self.loaded_ammo, self.reserve_ammo], dtype=np.float32)
                weapon_obs = np.array([self.damage, self.range_limit, self.spread], dtype=np.float32)
                return np.concatenate([flat_obs, ammo_obs, weapon_obs])
        
        # Fallback dummy observation if loading fails
        return np.zeros((133,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._check_emergency_stop()
        """Resets the environment for a new episode.

        Returns:
            A tuple containing (observation, info).
        """
        if gym is not None:
            super().reset(seed=seed)
        
        self.current_step = 0
        self.loaded_ammo = 2
        self.reserve_ammo = 8
        print("Resetting Shotgun King environment...")
        
        # Safety timeout: Allow user a 3.5-second window to reclaim focus or stop the loop
        time.sleep(3.5)
        
        # Trigger an active retry only if the Game Over screen is actually detected
        image_path = "data/screenshot.png"
        capture_screen(output_path=image_path, window_title=self.window_title)
        
        if cv2 is not None and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if check_retry_popup(img):
                print("DQN Penalty: Detected retry popup during reset. Clicking YES button (Multi-point click enabled).")
                # 5-point safety click to offset window scaling/borders
                for dx, dy in [(530, 410), (540, 410), (550, 410), (540, 400), (540, 420)]:
                    click_relative_in_window(self.window_title, dx, dy)
                    time.sleep(0.05)
                time.sleep(2.5)
        
        obs = self._get_obs()
        
        # In-game Start Sync Guard: Poll until Player King (1) is detected on the board
        while True:
            self._check_emergency_stop()
            board_state = obs[:64].reshape(8, 8)
            if np.any(board_state == 1):
                print("In-game Sync: Player King detected. Game play has officially started!")
                break
            print("In-game Sync: Waiting for game play to start (King not found on board)...")
            time.sleep(1.0)
            obs = self._get_obs()
            
        self.current_state = obs
        info = {}
        return obs, info

    def step(self, action):
        self._check_emergency_stop()
        """Executes a single step in the environment by applying the action.

        Args:
            action: Integer action index from the Action Space.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        self.current_step += 1
        print(f"Step {self.current_step} - Executing action: {action}")

        if self.current_state is None:
            self.current_state = self._get_obs()

        # Reconstruct 8x8 matrices from 133-dimensional flat state
        board_state = self.current_state[:64].reshape(8, 8)
        threat_state = self.current_state[64:128].reshape(8, 8)

        # Count enemies before action execution
        prev_enemies = np.sum(board_state >= 2)

        # Ammo, Range & Target Validations for Shoot Guard
        if action == 9:
            # Check range limit and target existence
            king_positions = np.argwhere(board_state == 1)
            has_valid_target = False
            min_dist = 99
            
            if len(king_positions) > 0:
                king_row, king_col = king_positions[0]
                # 8 directions to sweep radially for enemies
                directions = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)
                ]
                for r_diff, c_diff in directions:
                    for dist in range(1, 8):
                        tr = king_row + r_diff * dist
                        tc = king_col + c_diff * dist
                        if 0 <= tr < 8 and 0 <= tc < 8:
                            if board_state[tr, tc] >= 2:
                                if dist < min_dist:
                                    min_dist = dist
                                break
                            elif board_state[tr, tc] == 1:
                                break
                        else:
                            break
                if min_dist <= self.range_limit:
                    has_valid_target = True

            if self.loaded_ammo <= 0:
                if self.reserve_ammo > 0:
                    print(f"DQN Guard: Shoot action (9) requested but loaded_ammo is {self.loaded_ammo}. Overwriting to Reload (8).")
                    action = 8
                else:
                    import random
                    action = random.randint(0, 7)
                    print(f"DQN Guard: Shoot action (9) requested but ammo fully depleted. Overwriting to Random Move ({action}).")
            elif not has_valid_target:
                import random
                action = random.randint(0, 7)
                reason = f"closest target is out of range (dist: {min_dist} > limit: {self.range_limit})" if min_dist != 99 else "no enemies detected on 8-way radial paths"
                print(f"DQN Guard: Shoot action (9) requested but {reason}. Overwriting to Random Move ({action}).")
        elif action == 8 and self.loaded_ammo >= 2:
            import random
            action = random.randint(0, 7)
            print(f"DQN Guard: Reload action (8) requested but ammo already full. Overwriting to Random Move ({action}).")
        elif action == 8 and self.reserve_ammo <= 0 and self.loaded_ammo < 2:
            import random
            action = random.randint(0, 7)
            print(f"DQN Guard: Reload action (8) requested but reserve_ammo is 0. Overwriting to Random Move ({action}).")

        # Execute action simulation
        if action in range(8):
            # Target direction calculations based on King's real coordinate (value 1)
            king_positions = np.argwhere(board_state == 1)
            
            if len(king_positions) > 0:
                king_row, king_col = king_positions[0]
                
                # Direction diffs mapping: row_offset, col_offset
                direction_diffs = {
                    0: (-1, -1),  # Up-Left
                    1: (-1, 0),   # Up
                    2: (-1, 1),   # Up-Right
                    3: (0, -1),   # Left
                    4: (0, 1),    # Right
                    5: (1, -1),   # Down-Left
                    6: (1, 0),    # Down
                    7: (1, 1),    # Down-Right
                }
                
                row_offset, col_offset = direction_diffs[action]
                target_row = king_row + row_offset
                target_col = king_col + col_offset
                
                # Out-of-bounds safety check and replacement loop
                attempts = 0
                while not (0 <= target_row < 8 and 0 <= target_col < 8) and attempts < 15:
                    import random
                    action = random.randint(0, 7)
                    row_offset, col_offset = direction_diffs[action]
                    target_row = king_row + row_offset
                    target_col = king_col + col_offset
                    attempts += 1
                
                if 0 <= target_row < 8 and 0 <= target_col < 8:
                    # Precise board cell calculation based on standard coordinates:
                    # x_start=380, y_start=120, cell_size=65
                    x = 380 + target_col * 65 + 32
                    y = 120 + target_row * 65 + 32
                    print(f"Calculated target coordinate for King from ({king_row}, {king_col}) to ({target_row}, {target_col}) -> ({x}, {y}) (attempts: {attempts})")
                    click_relative_in_window(self.window_title, x, y)
                else:
                    # Absolute fallback clipping if loop somehow fails to find inside direction
                    target_row = max(0, min(7, target_row))
                    target_col = max(0, min(7, target_col))
                    x = 380 + target_col * 65 + 32
                    y = 120 + target_row * 65 + 32
                    print(f"Safety Clip target coordinate to ({target_row}, {target_col}) -> ({x}, {y}) due to out of bounds fallback.")
                    click_relative_in_window(self.window_title, x, y)
            else:
                print("King not found in board_state. Bypassing click action and waiting for turn stabilization...")
                time.sleep(1.0)

            # Move Rule: Automatically reload loaded_ammo from reserve_ammo when King moves
            needed = 2 - self.loaded_ammo
            transfer = min(needed, self.reserve_ammo)
            self.loaded_ammo += transfer
            self.reserve_ammo -= transfer
            print(f"Ammo System: King moved. Auto-reloaded {transfer} shells from reserve. (Loaded: {self.loaded_ammo}, Reserve: {self.reserve_ammo})")
                
        elif action == 8:
            # Reload
            press_key("r")
            needed = 2 - self.loaded_ammo
            transfer = min(needed, self.reserve_ammo)
            self.loaded_ammo += transfer
            self.reserve_ammo -= transfer
            print(f"Ammo System: Manual reload completed. Loaded {transfer} shells. (Loaded: {self.loaded_ammo}, Reserve: {self.reserve_ammo})")
            
        elif action == 9:
            # Shoot (Intel aimed click bypassing the 1-tile move physics rule)
            king_positions = np.argwhere(board_state == 1)
            
            if len(king_positions) > 0:
                king_row, king_col = king_positions[0]
                
                # 8 directions to sweep radially for enemies
                directions = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)
                ]
                
                min_dist = 99
                best_diff = None
                
                for r_diff, c_diff in directions:
                    for dist in range(1, 8):
                        tr = king_row + r_diff * dist
                        tc = king_col + c_diff * dist
                        if 0 <= tr < 8 and 0 <= tc < 8:
                            if board_state[tr, tc] >= 2:
                                if dist < min_dist:
                                    min_dist = dist
                                    best_diff = (r_diff, c_diff)
                                break  # Closest enemy on this ray found
                            elif board_state[tr, tc] == 1:
                                break
                        else:
                            break
                
                if best_diff is not None and min_dist <= self.range_limit:
                    r_diff, c_diff = best_diff
                    # 1-tile distance check: If target is 1-tile away, clicking it moves the King.
                    # Bypassed by shooting 2-tiles away in the same trajectory.
                    shoot_dist = 2 if min_dist == 1 else min_dist
                    
                    target_row = king_row + r_diff * shoot_dist
                    target_col = king_col + c_diff * shoot_dist
                    
                    # Precise absolute pixel conversion
                    x = 380 + target_col * 65 + 32
                    y = 120 + target_row * 65 + 32
                    print(f"Intel Shoot: Found enemy at dist {min_dist} (dir: {best_diff}). Aiming at ({target_row}, {target_col}) -> ({x}, {y})")
                    click_relative_in_window(self.window_title, x, y)
                else:
                    # Fallback if target is out of range or missing (normally filtered by action guard)
                    print(f"Intel Shoot Guard: Target out of range (dist: {min_dist} > limit: {self.range_limit}) or missing. Bypassing shot event.")
            else:
                print("Intel Shoot: King missing from state. Bypassing shoot click and waiting for turn stabilization...")
                time.sleep(1.0)

            self.loaded_ammo = max(0, self.loaded_ammo - 1)
            print(f"Ammo System: Shot fired. Loaded ammo consumed. (Loaded: {self.loaded_ammo}, Reserve: {self.reserve_ammo})")

        # Wait for turn transition and board state stabilization actively
        obs = self._wait_for_equilibrium()
        self.current_state = obs

        # Reconstruct board state for reward/evaluation after observation update
        curr_board = self.current_state[:64].reshape(8, 8)
        curr_threat = self.current_state[64:128].reshape(8, 8)

        # Calculate reward metrics
        curr_enemies = np.sum(curr_board >= 2)
        killed_enemies = max(0, prev_enemies - curr_enemies)
        
        # Base step reward (slight survival incentive)
        reward = 0.02
        
        # Major reward for killing enemies
        if killed_enemies > 0:
            reward += killed_enemies * 2.0
            print(f"DQN Reward: Killed {killed_enemies} enemy/enemies! Added +{killed_enemies * 2.0}")
            
        # Waste-shooting penalty (fired shoot action but killed no enemies)
        if action == 9 and killed_enemies == 0:
            reward -= 0.8
            print("DQN Penalty: Fired shoot action but killed no enemies. Subtracted -0.8")

        # Threat exposure evaluation (impose penalty if King stands inside enemy check lines)
        king_positions = np.argwhere(curr_board == 1)
        if len(king_positions) > 0:
            king_row, king_col = king_positions[0]
            if curr_threat[king_row, king_col] == 1:
                reward -= 0.5
                print("DQN Penalty: Exposed to enemy checkmate threat zone! Subtracted -0.5")

        # Threat Exposure Evaluation
        terminated = False
        king_present = np.any(curr_board == 1)
        
        # Capture screen and verify retry popup actively via screen analysis
        image_path = "data/screenshot.png"
        is_popup = False
        if cv2 is not None and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if check_retry_popup(img):
                is_popup = True

        if is_popup:
            reward = -5.0
            terminated = True
            print("DQN Penalty: Detected retry popup via screen analysis! Subtracted -5.0. Clicking YES button (Multi-point click enabled).")
            # 5-point safety click to offset window scaling/borders
            for dx, dy in [(530, 410), (540, 410), (550, 410), (540, 400), (540, 420)]:
                click_relative_in_window(self.window_title, dx, dy)
                time.sleep(0.05)
            time.sleep(2.5)
        elif curr_enemies == 0:
            reward += 10.0
            terminated = True
            print("DQN Reward: Congratulations! Level 1 cleared! Added +10.0. Episode terminated with victory.")
        elif not king_present:
            reward = -5.0
            terminated = True
            print("DQN Penalty: Player King missing but retry popup not yet detected. Postponing click to reset.")

        truncated = self.current_step >= self.max_steps
        
        if truncated:
            print("Episode truncated due to max steps limit.")

        info = {"step": self.current_step, "action": action}
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    print("Testing custom Gymnasium environment for Shotgun King...")
    if gym is not None:
        # Initialize environment with max steps limit of 5 for a quick test
        env = ShotgunKingEnv(window_title="Shotgun King", max_steps=5)
        
        # Perform initial reset
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Run a brief random walk
        for step_idx in range(1, 4):
            random_action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(random_action)
            print(f"Step {step_idx} - Obs shape: {obs.shape}, Reward: {reward}, Terminated: {term}, Truncated: {trunc}")
            time.sleep(0.5)
            
        print("Gymnasium environment random walk test completed successfully.")
    else:
        print("Error: Gymnasium is not available.")
