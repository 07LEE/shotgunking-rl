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
from weapons import WEAPON_PRESETS
from specs import get_enemy_specs_matrices


class ShotgunKingEnv(gym.Env if gym is not None else object):
    """Custom Gymnasium environment for interacting with Shotgun King."""

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, window_title="Shotgun King", max_steps=100, weapon_type=0, rank=1, buffs=None):
        """Initializes the environment state and spaces.

        Args:
            window_title: Title of the target game window.
            max_steps: Maximum steps allowed per episode before truncation.
            weapon_type: Integer ID representing registered weapon spec presets.
            rank: Target story mode difficulty level.
            buffs: Optional dictionary of active player buffs.
        """
        super().__init__()
        self.window_title = window_title
        self.max_steps = max_steps
        self.current_step = 0
        self.current_state = None
        self.rank = rank

        # Define Observation Space: 1D flat vector of size 281
        # 128 dimensions from board and threat matrix, plus 2 dimensions for ammo stats,
        # 3 dimensions for weapon specifications, 20 dimensions for buffs/debuffs,
        # 64 dimensions for enemy HP matrix, and 64 dimensions for enemy turn matrix
        self.observation_space = spaces.Box(
            low=0, high=90, shape=(281,), dtype=np.float32
        )

        # Weapon Specifications
        preset = WEAPON_PRESETS.get(weapon_type, WEAPON_PRESETS[0])
        self.damage = preset["damage"]
        self.falloff_start = preset["range_limit"][0]
        self.range_limit = preset["range_limit"][1]
        self.spread = preset["spread"]
        self.weapon_type = weapon_type
        self.pierce_chance = preset.get("pierce_chance", 0.0)
        self.knockback_chance = preset.get("knockback_chance", 0.0)
        
        if buffs is None:
            buffs = {}
        self.melee_damage = preset.get("melee_damage", 0.0)
        self.melee_kill_extra_turn = buffs.get("melee_kill_extra_turn", False)
        self.is_extra_turn_active = False

        self.max_ammo = preset.get("max_ammo", 2)
        self.max_reserve_ammo = preset.get("max_reserve_ammo", 8)

        # Ammo Tracking
        self.loaded_ammo = self.max_ammo
        self.reserve_ammo = self.max_reserve_ammo + (1 if self.rank >= 20 else 0)

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
        self.king_row = 7
        self.king_col = 4
        self.enemy_turns = {}

    def _check_emergency_stop(self):
        """Checks if the mouse cursor is located in the top-left corner (0,0) of the screen.

        Raises:
            KeyboardInterrupt: If the mouse cursor is at or near (0,0).
        """
        if pyautogui is not None:
            # Enforce pyautogui failsafe override configurations
            pyautogui.FAILSAFE = True
            mx, my = pyautogui.position()
            if mx <= 30 and my <= 30:
                print("\n=== DQN Emergency Stop: Mouse corner sweep detected! Halting training immediately. ===")
                raise KeyboardInterrupt("DQN Emergency Stop: User swept mouse to the corner.")

    def _wait_for_equilibrium(self, max_wait=5.0):
        """Wait until the enemy's turn animation starts and fully stabilizes visually.

        Args:
            max_wait: Maximum wait time in seconds.

        Returns:
            The final 133-dimensional observation vector in static equilibrium.
        """
        time.sleep(0.1)
        start_time = time.time()
        
        img_path = "data/screenshot.png"
        capture_screen(output_path=img_path, window_title=self.window_title)
        if not os.path.exists(img_path) or cv2 is None:
            return self._get_obs()
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return self._get_obs()
            
        # Crop to board ROI to focus on gameplay animations
        y1, y2, x1, x2 = 119, 631, 383, 895
        prev_roi = img[y1:y2, x1:x2]
        
        has_changed = False
        
        while time.time() - start_time < max_wait:
            time.sleep(0.15)
            capture_screen(output_path=img_path, window_title=self.window_title)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            curr_roi = img[y1:y2, x1:x2]
            
            diff = cv2.absdiff(prev_roi, curr_roi)
            mean_diff = np.mean(diff)
            
            # Check for visual transitions (animation start and end)
            if not has_changed:
                if mean_diff > 1.5:
                    has_changed = True
                elif time.time() - start_time > 0.6:
                    # Early exit if no animation starts after 0.6 seconds (e.g. invalid action or already game over)
                    break
            else:
                if mean_diff < 0.5:
                    break
                    
            prev_roi = curr_roi
            
        time.sleep(0.2)
        return self._get_obs()

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
        """Captures screen and returns a 281-dimensional flat observation vector.

        Returns:
            A 281-dimensional numpy float32 observation vector.
        """
        self._check_emergency_stop()
        image_path = "data/screenshot.png"
        
        # Ensure fresh screen capture
        capture_screen(output_path=image_path, window_title=self.window_title)
        
        if cv2 is not None and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                # Call state extractor to return 8x8 chessboard array
                state = get_state_matrix(img).astype(np.float32)
                threat = self._get_threat_matrix(state).astype(np.float32)
                
                # Real-time ammo sync from screenshot UI
                from analyzer import extract_ammo_count
                self.loaded_ammo, self.reserve_ammo = extract_ammo_count(img)
                # print(f"Ammo Sync: Real-time UI scan matched (Loaded: {self.loaded_ammo}, Reserve: {self.reserve_ammo})")

                # Construct HP and Turn Speed matrices for detected enemy pieces
                hp_matrix, turn_matrix = get_enemy_specs_matrices(state, self.enemy_turns, self.rank)

                flat_obs = np.concatenate([state.flatten(), threat.flatten()])
                ammo_obs = np.array([self.loaded_ammo, self.reserve_ammo], dtype=np.float32)
                weapon_obs = np.array([self.damage, self.range_limit, self.spread], dtype=np.float32)
                status_obs = np.zeros((20,), dtype=np.float32)
                if self.is_extra_turn_active:
                    status_obs[0] = 1.0
                enemy_spec_obs = np.concatenate([hp_matrix.flatten(), turn_matrix.flatten()])
                return np.concatenate([flat_obs, ammo_obs, weapon_obs, status_obs, enemy_spec_obs])
        
        # Fallback dummy observation if loading fails
        return np.zeros((281,), dtype=np.float32)

    def _get_yes_button_coords(self, img):
        """Finds the precise (x, y) coordinates of the active YES button on the retry screen dynamically.

        Args:
            img: Full 1280x720 BGR game screen screenshot.

        Returns:
            A tuple of (x, y) relative pixel coordinates of the YES button centroid.
        """
        if img is None or cv2 is None or np is None:
            return (540, 410)
        try:
            height, width, _ = img.shape
            if height != 720 or width != 1280:
                img = cv2.resize(img, (1280, 720))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Crop YES button region (Y: 380-440, X: 500-580)
            y1, y2, x1, x2 = 380, 440, 500, 580
            patch = gray[y1:y2, x1:x2]
            _, thresh = cv2.threshold(patch, 180, 255, cv2.THRESH_BINARY)
            M = cv2.moments(thresh)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (x1 + cx, y1 + cy)
            return (540, 410)
        except Exception as e:
            print(f"Failed to find YES button coordinates dynamically: {e}")
            return (540, 410)

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Optional environment configuration options.

        Returns:
            A tuple containing (observation, info).
        """
        self._check_emergency_stop()
        if gym is not None:
            super().reset(seed=seed)
        
        self.current_step = 0
        self.loaded_ammo = self.max_ammo
        self.is_extra_turn_active = False
        self.reserve_ammo = self.max_reserve_ammo + (1 if self.rank >= 20 else 0)
        print("Resetting Shotgun King environment...")
        
        # Safety timeout: Allow user a 3.5-second window to reclaim focus or stop the loop
        time.sleep(3.5)
        
        # Trigger an active retry only if the Game Over screen is actually detected
        image_path = "data/screenshot.png"
        
        # Poll for retry popup up to 10 attempts (5 seconds total)
        popup_detected = False
        for attempt in range(10):
            capture_screen(output_path=image_path, window_title=self.window_title)
            if cv2 is not None and os.path.exists(image_path):
                img = cv2.imread(image_path)
                if check_retry_popup(img):
                    popup_detected = True
                    break
            time.sleep(0.5)
            
        if popup_detected:
            print("DQN Penalty: Detected retry popup during reset. Clicking YES button (dynamic coordinates enabled).")
            img = cv2.imread(image_path) if cv2 is not None and os.path.exists(image_path) else None
            tx, ty = self._get_yes_button_coords(img)
            self._check_emergency_stop()
            click_relative_in_window(self.window_title, tx, ty)
            time.sleep(2.5)
        else:
            # Check if King is already present before attempting fallback force click
            initial_obs = self._get_obs()
            initial_board = initial_obs[:64].reshape(8, 8)
            if np.any(initial_board == 1):
                print("DQN Reset: Retry popup not detected, but Player King is already present. Bypassing force click.")
            else:
                # Fallback Force Click: If King is missing from the board state, perform force retry YES click
                # to break out of potential infinite loading sync loop
                print("DQN Reset Warning: Retry popup not verified by pixel variance, and King is missing. Performing force YES click (dynamic coordinates enabled).")
                img = cv2.imread(image_path) if cv2 is not None and os.path.exists(image_path) else None
                tx, ty = self._get_yes_button_coords(img)
                self._check_emergency_stop()
                click_relative_in_window(self.window_title, tx, ty)
                time.sleep(2.5)
        
        obs = self._get_obs()
        
        # In-game Start Sync Guard: Poll until Player King (1) is detected on the board
        while True:
            self._check_emergency_stop()
            board_state = obs[:64].reshape(8, 8)
            if np.any(board_state == 1):
                print("In-game Sync: Player King detected. Game play has officially started!")
                time.sleep(1.5)  # Allow turn intro animation to fully finish before first step
                obs = self._get_obs()
                fresh_board = obs[:64].reshape(8, 8)
                king_positions = np.argwhere(fresh_board == 1)
                if len(king_positions) > 0:
                    self.king_row, self.king_col = int(king_positions[0][0]), int(king_positions[0][1])
                break
            print("In-game Sync: Waiting for game play to start (King not found on board)...")
            time.sleep(1.0)
            obs = self._get_obs()
            
        self.enemy_turns.clear()
        initial_board = obs[:64].reshape(8, 8)
        from specs import ENEMY_SPECS
        for r in range(8):
            for c in range(8):
                val = int(initial_board[r, c])
                if val in ENEMY_SPECS:
                    self.enemy_turns[(r, c)] = ENEMY_SPECS[val]["turn"]

        self.current_state = obs
        info = {}
        return obs, info

    def step(self, action):
        """Executes a single step in the environment by applying the action.

        Args:
            action: Integer action index from the Action Space.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        self._check_emergency_stop()
        self.current_step += 1
        print(f"Step {self.current_step} - Executing action: {action}")

        original_action = action
        has_valid_target = False
        best_diff = None

        # Force fresh observation update before computing action parameters
        self.current_state = self._get_obs()

        # Reconstruct 8x8 matrices from 133-dimensional flat state
        board_state = self.current_state[:64].reshape(8, 8)
        threat_state = self.current_state[64:128].reshape(8, 8)

        # Determine Player King position (either detected or logical backup)
        king_positions = np.argwhere(board_state == 1)
        if len(king_positions) > 0:
            king_row, king_col = int(king_positions[0][0]), int(king_positions[0][1])
            self.king_row, self.king_col = king_row, king_col
        else:
            king_row, king_col = self.king_row, self.king_col
            print(f"King missing in board_state. Using logical tracked position: ({king_row}, {king_col})")

        # Count enemies before action execution
        prev_enemies = np.sum(board_state >= 2)

        # Ammo, Range & Target Validations for Shoot Guard
        if action == 9:
            # Check range limit and target existence
            has_valid_target = False
            min_dist = 99
            best_is_threat = False
            
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
                        piece = board_state[tr, tc]
                        if piece >= 2:
                            is_threat = False
                            is_diagonal = (abs(r_diff) == 1 and abs(c_diff) == 1)
                            is_straight = (r_diff == 0 or c_diff == 0)
                            
                            if piece == 2:  # Pawn: attacks diagonally down
                                if r_diff == -1 and is_diagonal and dist == 1:
                                        is_threat = True
                            elif piece == 4:  # Bishop: diagonal threat
                                if is_diagonal:
                                    is_threat = True
                            elif piece == 5:  # Rook: straight threat
                                if is_straight:
                                    is_threat = True
                            elif piece == 6:  # Queen: straight or diagonal threat
                                if is_straight or is_diagonal:
                                    is_threat = True
                            
                            if not best_is_threat and is_threat:
                                min_dist = dist
                                best_is_threat = is_threat
                            elif is_threat == best_is_threat:
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
        elif action == 8 and self.loaded_ammo >= self.max_ammo:
            import random
            action = random.randint(0, 7)
            print(f"DQN Guard: Reload action (8) requested but ammo already full. Overwriting to Random Move ({action}).")
        elif action == 8 and self.reserve_ammo <= 0 and self.loaded_ammo < self.max_ammo:
            import random
            action = random.randint(0, 7)
            print(f"DQN Guard: Reload action (8) requested but reserve_ammo is 0. Overwriting to Random Move ({action}).")

        # Execute action simulation
        if action in range(8):
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
                # x_start=390, y_start=127, cell_size=62.5
                x = int(390 + target_col * 62.5 + 31.25)
                y = int(127 + target_row * 62.5 + 31.25)
                print(f"Calculated target coordinate for King from ({king_row}, {king_col}) to ({target_row}, {target_col}) -> ({x}, {y}) (attempts: {attempts})")
                self._check_emergency_stop()
                click_relative_in_window(self.window_title, x, y)
                time.sleep(1.8)
            else:
                # Absolute fallback clipping if loop somehow fails to find inside direction
                target_row = max(0, min(7, target_row))
                target_col = max(0, min(7, target_col))
                x = int(390 + target_col * 62.5 + 31.25)
                y = int(127 + target_row * 62.5 + 31.25)
                print(f"Safety Clip target coordinate to ({target_row}, {target_col}) -> ({x}, {y}) due to out of bounds fallback.")
                self._check_emergency_stop()
                click_relative_in_window(self.window_title, x, y)
                time.sleep(1.8)

            # Update tracked position internally
            self.king_row = target_row
            self.king_col = target_col

            # Move Rule: Automatically reload loaded_ammo from reserve_ammo when King moves
            needed = max(0, self.max_ammo - self.loaded_ammo)
            transfer = min(needed, self.reserve_ammo)
            self.loaded_ammo += transfer
            self.reserve_ammo -= transfer
            print(f"Ammo System: King moved. Auto-reloaded {transfer} shells from reserve. (Loaded: {self.loaded_ammo}, Reserve: {self.reserve_ammo})")
                
        elif action == 8:
            # Reload
            # Replace keypress reload with relative click on gun UI at (640, 650)
            click_relative_in_window(self.window_title, 640, 650)
            time.sleep(1.8)
            needed = max(0, self.max_ammo - self.loaded_ammo)
            transfer = min(needed, self.reserve_ammo)
            self.loaded_ammo += transfer
            self.reserve_ammo -= transfer
            print(f"Ammo System: Manual reload completed. Loaded {transfer} shells. (Loaded: {self.loaded_ammo}, Reserve: {self.reserve_ammo})")
            
        elif action == 9:
            # Shoot (Intel aimed click bypassing the 1-tile move physics rule)
            min_dist = 99
            best_diff = None
            best_is_threat = False
            
            for r_diff, c_diff in directions:
                for dist in range(1, 8):
                    tr = king_row + r_diff * dist
                    tc = king_col + c_diff * dist
                    if 0 <= tr < 8 and 0 <= tc < 8:
                        piece = board_state[tr, tc]
                        if piece >= 2:
                            is_threat = False
                            is_diagonal = (abs(r_diff) == 1 and abs(c_diff) == 1)
                            is_straight = (r_diff == 0 or c_diff == 0)
                            
                            if piece == 2:  # Pawn: attacks diagonally down
                                if r_diff == -1 and is_diagonal and dist == 1:
                                    is_threat = True
                            elif piece == 4:  # Bishop: diagonal threat
                                if is_diagonal:
                                    is_threat = True
                            elif piece == 5:  # Rook: straight threat
                                if is_straight:
                                    is_threat = True
                            elif piece == 6:  # Queen: straight or diagonal threat
                                if is_straight or is_diagonal:
                                    is_threat = True
                            
                            if best_diff is None:
                                min_dist = dist
                                best_diff = (r_diff, c_diff)
                                best_is_threat = is_threat
                            else:
                                if is_threat and not best_is_threat:
                                    min_dist = dist
                                    best_diff = (r_diff, c_diff)
                                    best_is_threat = is_threat
                                elif is_threat == best_is_threat:
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
                # Exception: If melee_damage > 0, automatically decide between melee and ranged shooting based on ammo and damage.
                if min_dist == 1 and self.melee_damage > 0.0:
                    if self.loaded_ammo <= 0 or self.melee_damage >= self.damage:
                        shoot_dist = 1
                    else:
                        shoot_dist = 2
                else:
                    shoot_dist = 2 if min_dist == 1 else min_dist
                
                target_row = king_row + r_diff * shoot_dist
                target_col = king_col + c_diff * shoot_dist
                
                # Precise absolute pixel conversion
                x = int(390 + target_col * 62.5 + 31.25)
                y = int(127 + target_row * 62.5 + 31.25)
                print(f"Intel Shoot: Found enemy at dist {min_dist} (dir: {best_diff}). Aiming at ({target_row}, {target_col}) -> ({x}, {y})")
                time.sleep(0.4)
                self._check_emergency_stop()
                click_relative_in_window(self.window_title, x, y)
                time.sleep(1.8)
            else:
                # Fallback if target is out of range or missing (normally filtered by action guard)
                print(f"Intel Shoot Guard: Target out of range (dist: {min_dist} > limit: {self.range_limit}) or missing. Bypassing shot event.")

            self.loaded_ammo = max(0, self.loaded_ammo - 1)
            print(f"Ammo System: Shot fired. Loaded ammo consumed. (Loaded: {self.loaded_ammo}, Reserve: {self.reserve_ammo})")

        # Wait for turn transition and board state stabilization actively
        obs = self._wait_for_equilibrium()
        self.current_state = obs

        # Reconstruct board state for reward/evaluation after observation update
        curr_board = self.current_state[:64].reshape(8, 8)
        curr_threat = self.current_state[64:128].reshape(8, 8)

        # Calculate intermediate kill counts for turn skip decisions
        curr_enemies = np.sum(curr_board >= 2)
        killed_enemies = max(0, prev_enemies - curr_enemies)
        if action != 9:
            killed_enemies = 0

        # Determine if a melee kill actually occurred
        is_melee_kill = False
        if action == 9 and killed_enemies > 0:
            if min_dist == 1 and self.melee_damage > 0.0:
                if self.loaded_ammo <= 0 or self.melee_damage >= self.damage:
                    is_melee_kill = True

        # Update extra turn activation state for observation
        if is_melee_kill and self.melee_kill_extra_turn:
            self.is_extra_turn_active = True
        else:
            self.is_extra_turn_active = False

        # Update Enemy Turn Counter Simulation Logically
        new_enemy_turns = {}
        from specs import ENEMY_SPECS

        # Find matching previous positions for each active enemy piece on current board
        for cr in range(8):
            for cc in range(8):
                val = int(curr_board[cr, cc])
                if val >= 2:  # Enemy piece detected on current board
                    # Attempt to find the coordinate this piece moved from
                    prev_coords = []
                    for pr in range(8):
                        for pc in range(8):
                            if int(board_state[pr, pc]) == val:
                                prev_coords.append((pr, pc))

                    # Decide the source coordinate (closest to current coord is logical)
                    matched_prev = None
                    if len(prev_coords) > 0:
                        min_d = 99
                        for pr, pc in prev_coords:
                            d = abs(cr - pr) + abs(cc - pc)
                            if d < min_d:
                                min_d = d
                                matched_prev = (pr, pc)

                    # Determine if it moved or stood still
                    is_moved = True
                    if matched_prev is not None:
                        if matched_prev == (cr, cc):
                            is_moved = False

                    # Assign turn state
                    if is_moved or matched_prev not in self.enemy_turns:
                        new_enemy_turns[(cr, cc)] = ENEMY_SPECS[val]["turn"]
                    else:
                        prev_turn = self.enemy_turns[matched_prev]
                        if is_melee_kill and self.melee_kill_extra_turn:
                            new_enemy_turns[(cr, cc)] = prev_turn
                        else:
                            if prev_turn <= 0.0:
                                new_enemy_turns[(cr, cc)] = ENEMY_SPECS[val]["turn"]
                            else:
                                new_enemy_turns[(cr, cc)] = max(0.0, prev_turn - 1.0)

        self.enemy_turns = new_enemy_turns

        # Calculate reward metrics
        
        # Base step penalty (discourage wasting turns)
        reward = -0.1
        
        # Expected damage calculation based on distance and spread if a shoot attempt or action occurs
        expected_damage = 0.0
        if original_action == 9 or action == 9:
            if has_valid_target and min_dist <= self.range_limit:
                # Determine if melee attack was executed
                is_melee = False
                if min_dist == 1 and self.melee_damage > 0.0:
                    if self.loaded_ammo <= 0 or self.melee_damage >= self.damage:
                        is_melee = True
                
                if is_melee:
                    expected_damage = self.melee_damage
                    # If melee kill is expected, add bonus to reflect extra turn advantage
                    if self.melee_kill_extra_turn:
                        target_hp = hp_matrix[target_row, target_col]
                        if self.melee_damage >= target_hp:
                            expected_damage += 1.5
                else:
                    # 1. Distance falloff calculation
                    if min_dist <= self.falloff_start:
                        dist_factor = 1.0
                    else:
                        denom_falloff = float(self.range_limit - self.falloff_start)
                        dist_factor = 1.0 - 0.5 * (float(min_dist - self.falloff_start) / denom_falloff) if denom_falloff > 0 else 1.0
                    
                    # 2. Spread-based hit probability calculation
                    denom_range = float(self.range_limit)
                    hit_prob = max(0.2, 1.0 - (self.spread / 120.0) * (float(min_dist - 1) / denom_range)) if denom_range > 0 else 1.0
                    
                    expected_damage = self.damage * dist_factor * hit_prob

                # 3. Optional pierce damage calculation for a second target behind the first
                if self.pierce_chance > 0.0 and best_diff is not None:
                    r_diff, c_diff = best_diff
                    min_dist_2 = 99
                    for dist_2 in range(min_dist + 1, self.range_limit + 1):
                        tr = king_row + r_diff * dist_2
                        tc = king_col + c_diff * dist_2
                        if 0 <= tr < 8 and 0 <= tc < 8:
                            if board_state[tr, tc] >= 2:
                                min_dist_2 = dist_2
                                break
                            elif board_state[tr, tc] == 1:
                                break
                        else:
                            break
                    
                    if min_dist_2 <= self.range_limit:
                        if min_dist_2 <= self.falloff_start:
                            dist_factor_2 = 1.0
                        else:
                            denom_falloff_2 = float(self.range_limit - self.falloff_start)
                            dist_factor_2 = 1.0 - 0.5 * (float(min_dist_2 - self.falloff_start) / denom_falloff_2) if denom_falloff_2 > 0 else 1.0
                        
                        hit_prob_2 = max(0.2, 1.0 - (self.spread / 120.0) * (float(min_dist_2 - 1) / denom_range)) if denom_range > 0 else 1.0
                        expected_damage_2 = self.damage * dist_factor_2 * hit_prob_2
                        expected_damage += self.pierce_chance * expected_damage_2

                # 4. Optional knockback / Fall-off instant kill expectation reward
                expected_knockback_reward = 0.0
                if self.knockback_chance > 0.0 and best_diff is not None:
                    r_diff, c_diff = best_diff
                    tr_back = king_row + r_diff * (min_dist + 1)
                    tc_back = king_col + c_diff * (min_dist + 1)
                    is_out_of_bounds = not (0 <= tr_back < 8 and 0 <= tc_back < 8)
                    
                    if is_out_of_bounds:
                        expected_knockback_reward = self.knockback_chance * 2.0
                    else:
                        expected_knockback_reward = self.knockback_chance * 0.4

        # Major reward for killing enemies
        if killed_enemies > 0:
            reward += killed_enemies * 2.0
            if expected_damage > 0:
                reward += expected_damage * 0.3
            if expected_knockback_reward > 0:
                reward += expected_knockback_reward
            print(f"DQN Reward: Killed {killed_enemies} enemy/enemies! Added +{killed_enemies * 2.0 + expected_damage * 0.3 + expected_knockback_reward:.2f} (including {expected_damage * 0.3:.2f} damage, {expected_knockback_reward:.2f} knockback reward)")
            
        # Waste-shooting penalty / Encouragement reward
        if action == 9:
            if killed_enemies == 0:
                if has_valid_target:
                    reward += expected_damage * 0.3 + expected_knockback_reward
                    print(f"DQN Reward: Fired shoot action at a target with expected damage {expected_damage:.2f} but killed no enemies. Added +{expected_damage * 0.3 + expected_knockback_reward:.2f} (including {expected_knockback_reward:.2f} knockback reward)")
                else:
                    reward -= 0.8
                    print("DQN Penalty: Fired shoot action but no target was in range. Subtracted -0.8")
        elif original_action == 9 and action != 9:
            # Attempted to shoot but got overridden by DQN Guard (e.g. out of range / spread issues)
            if not has_valid_target:
                reward -= 0.8
                print("DQN Penalty: Attempted shoot action but no target was in range (Action Guarded). Subtracted -0.8")

        # Threat exposure evaluation (impose penalty if King stands inside enemy check lines)
        king_positions = np.argwhere(curr_board == 1)
        if len(king_positions) > 0:
            king_row, king_col = king_positions[0]
            if curr_threat[king_row, king_col] == 1:
                reward -= 1.5
                print("DQN Penalty: Exposed to enemy checkmate threat zone! Subtracted -1.5")

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
            reward = -15.0 - float(curr_enemies)
            terminated = True
            print(f"DQN Penalty: Detected retry popup via screen analysis! Subtracted {-15.0 - float(curr_enemies):.1f} (including {curr_enemies} enemies penalty). Clicking YES button (dynamic coordinates enabled).")
            img = cv2.imread(image_path) if cv2 is not None and os.path.exists(image_path) else None
            tx, ty = self._get_yes_button_coords(img)
            self._check_emergency_stop()
            click_relative_in_window(self.window_title, tx, ty)
            time.sleep(2.5)
        elif curr_enemies == 0:
            reward += 10.0
            terminated = True
            print("DQN Reward: Congratulations! Level 1 cleared! Added +10.0. Episode terminated with victory.")
        elif not king_present:
            reward = -15.0 - float(curr_enemies)
            terminated = True
            print(f"DQN Penalty: Player King missing but retry popup not yet detected. Subtracted {-15.0 - float(curr_enemies):.1f} (including {curr_enemies} enemies penalty). Postponing click to reset.")

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
