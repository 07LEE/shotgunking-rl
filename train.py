"""Training loop script for Shotgun King reinforcement learning.

This script coordinates the custom Gymnasium environment and the DQN agent
to run training episodes, perform replay updates, and optimize decisions.
"""

import time
import sys
import os

# Add src to python path for internal imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import numpy as np
except ImportError:
    np = None

from env import ShotgunKingEnv
from agent import DQNAgent


def train_dqn(episodes=2, batch_size=16, max_steps_per_episode=10, mode="autonomous", learning_rate=1e-3, epsilon_decay=0.95, epsilon_min=0.05):
    """Executes the DQN training loop over a specified number of episodes.

    Args:
        episodes: Total number of game episodes to run.
        batch_size: Mini-batch size for DQN experience replay.
        max_steps_per_episode: Maximum steps limit per episode.
        mode: Game control execution mode.
        learning_rate: Optimizer parameter for step adjustments.
        epsilon_decay: Exploration factor multiplier.
        epsilon_min: Minimum exploration cutoff limit.
    """
    print("Initializing Shotgun King Gymnasium Environment...")
    env = ShotgunKingEnv(
        window_title="Shotgun King",
        max_steps=max_steps_per_episode,
        weapon_type=config.get("weapon_type", 0),
        rank=config.get("rank", 1),
        buffs=config.get("buffs", None),
        verbose=config.get("verbose", True),
    )
    
    print("Initializing DQN Agent...")
    # 8x8 input flat is 64, with threat map flat is 128, plus 2 ammo dimensions is 130,
    # plus 3 weapon specs is 133, plus 20 buffs/debuffs is 153, plus 64 enemy hp is 217,
    # plus 64 enemy turn speed is 281
    agent = DQNAgent(state_size=281, action_size=env.action_space.n, lr=learning_rate)
    
    # Auto-load existing model weights if available to resume continuous learning
    agent.load("models/model.pth")
    
    # Exploration parameters
    epsilon = 1.0
    epsilon_min = epsilon_min
    epsilon_decay = epsilon_decay
    
    update_target_steps = 10
    total_step_count = 0
    
    print("\n=== STARTING SHOTGUN KING DQN REINFORCEMENT LEARNING ===")
    
    try:
        for ep in range(1, episodes + 1):
            print(f"\n--- Episode {ep}/{episodes} (Epsilon: {epsilon:.3f}) ---")
            state, info = env.reset()
            
            episode_reward = 0.0
            step_idx = 0
            
            while True:
                step_idx += 1
                total_step_count += 1
                
                # Direction diffs mapping: row_offset, col_offset
                direction_diffs = {
                    0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
                    3: (0, -1),             4: (0, 1),
                    5: (1, -1),  6: (1, 0),  7: (1, 1)
                }

                def is_action_valid(act):
                    board_state = env.current_state[:64].reshape(8, 8) if env.current_state is not None else np.zeros((8, 8))
                    if act in range(8):
                        row_offset, col_offset = direction_diffs[act]
                        target_row = env.king_row + row_offset
                        target_col = env.king_col + col_offset
                        if 0 <= target_row < 8 and 0 <= target_col < 8:
                            return board_state[target_row, target_col] == 0
                        return False
                    elif act in range(10, 18):
                        row_offset, col_offset = direction_diffs[act - 10]
                        mid_row = env.king_row + row_offset
                        mid_col = env.king_col + col_offset
                        target_row = env.king_row + row_offset * 2
                        target_col = env.king_col + col_offset * 2
                        if 0 <= target_row < 8 and 0 <= target_col < 8:
                            return board_state[mid_row, mid_col] == 0 and board_state[target_row, target_col] == 0
                        return False
                    return True

                # Calculate Q-values once per step to optimize performance by avoiding duplicate feed-forwards
                q_values = None
                if agent.policy_net is not None:
                    import torch
                    state_tensor = torch.tensor(state.astype(np.float32)).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        q_values = agent.policy_net(state_tensor).cpu().numpy()[0]

                # Select action using epsilon-greedy policy with cached Q-values
                import random
                if random.random() < epsilon or q_values is None:
                    action = random.randint(0, agent.action_size - 1)
                else:
                    action = int(np.argmax(q_values))

                # Verify and filter AI choice to ensure no out-of-bounds move
                if not is_action_valid(action):
                    if random.random() < epsilon or q_values is None:
                        # Find all valid actions
                        valid_actions = [a for a in range(env.action_space.n) if is_action_valid(a)]
                        action = random.choice(valid_actions)
                    else:
                        # Select best valid action by ranking cached Q-values
                        ranked_actions = np.argsort(q_values)[::-1]
                        for act in ranked_actions:
                            if is_action_valid(act):
                                action = int(act)
                                break

                # Handle decision suggestion and override in suggest mode
                if mode == "suggest" and q_values is not None:
                    action_names = {
                        0: "Move Up-Left", 1: "Move Up", 2: "Move Up-Right",
                        3: "Move Left", 4: "Move Right",
                        5: "Move Down-Left", 6: "Move Down", 7: "Move Down-Right",
                        8: "Reload", 9: "Shoot"
                    }
                    
                    # Sort Q-values descending for suggestions, filtering out invalid moves
                    top_actions = []
                    ranked_actions = np.argsort(q_values)[::-1]
                    for act_idx in ranked_actions:
                        if is_action_valid(act_idx):
                            top_actions.append(act_idx)
                            if len(top_actions) == 3:
                                break
                                
                    # Calculate softmax probabilities for Top 3 actions to project relative confidence %
                    exp_q = np.exp(q_values - np.max(q_values))
                    softmax_probs = exp_q / np.sum(exp_q)
                    
                    # Convert Numpad inputs to internal action index
                    numpad_map = {
                        "7": 0, "8": 1, "9": 2,
                        "4": 3, "5": 8, "6": 4,
                        "1": 5, "2": 6, "3": 7,
                        "0": 9
                    }
                    reverse_numpad_map = {v: k for k, v in numpad_map.items()}

                    def print_valid_action_hints():
                        valid_keys = []
                        check_order = [0, 1, 2, 3, 8, 4, 5, 6, 7, 9]
                        for act_i in check_order:
                            is_ok = is_action_valid(act_i)
                            if act_i == 9:
                                is_ok = is_ok and env.loaded_ammo > 0
                            elif act_i == 8:
                                is_ok = is_ok and env.loaded_ammo < env.max_ammo and env.reserve_ammo > 0
                            
                            if is_ok:
                                key_hint = reverse_numpad_map.get(act_i, "?")
                                if act_i == 9:
                                    valid_keys.append(f"{key_hint} (Shoot)")
                                elif act_i == 8:
                                    valid_keys.append(f"{key_hint} (Reload)")
                                else:
                                    valid_keys.append(key_hint)
                        print(f"Valid choices for current board state: {', '.join(valid_keys)}")

                    print("\n=== AI Decision Suggestion ===")
                    for rank, act_idx in enumerate(top_actions, 1):
                        confidence = softmax_probs[act_idx] * 100
                        key_hint = reverse_numpad_map.get(act_idx, "?")
                        print(f"Rank {rank}: {action_names[act_idx]} [Key: {key_hint}] (Confidence: {confidence:.1f}%, Q-value: {q_values[act_idx]:.4f})")

                    # Validation loop for user choice
                    while True:
                        user_choice = input(f"\nRecommended: [{action_names[action]}]. Press Enter to confirm, or enter custom action ID (0-9/Numpad): ").strip()
                        if user_choice == ".":
                            print("User flagged defeat. Forcing episode termination...")
                            agent.remember(state, action, -15.0, state, True)
                            break
                        
                        temp_action = action
                        if user_choice in numpad_map:
                            temp_action = numpad_map[user_choice]
                        elif user_choice.isdigit() and int(user_choice) in range(10):
                            temp_action = int(user_choice)
                        elif user_choice == "":
                            # Use recommended action
                            temp_action = action
                        else:
                            print("Invalid input. Please enter 0-9, Numpad keys, or '.' to exit.")
                            print_valid_action_hints()
                            continue

                        # Wall boundary and collision check details for move actions
                        if temp_action in range(8):
                            row_offset, col_offset = direction_diffs[temp_action]
                            target_row = env.king_row + row_offset
                            target_col = env.king_col + col_offset
                            
                            if not (0 <= target_row < 8 and 0 <= target_col < 8):
                                print(f"Move blocked by wall! Target ({target_row}, {target_col}) is out of bounds. Current King: ({env.king_row}, {env.king_col}). Choose another action.")
                                print_valid_action_hints()
                                continue
                                
                            board_state = env.current_state[:64].reshape(8, 8) if env.current_state is not None else np.zeros((8, 8))
                            if board_state[target_row, target_col] != 0:
                                print(f"Move blocked by piece! Target ({target_row}, {target_col}) contains a piece (Type {board_state[target_row, target_col]}). Current King: ({env.king_row}, {env.king_col}). Choose another action.")
                                print_valid_action_hints()
                                continue
                        elif temp_action == 9:
                            if env.loaded_ammo <= 0:
                                print(f"Shoot blocked by ammo! Loaded ammo is 0. Current ammo: {env.loaded_ammo}/{env.max_ammo}. Choose another action.")
                                print_valid_action_hints()
                                continue
                        elif temp_action == 8:
                            if env.loaded_ammo >= env.max_ammo:
                                print(f"Reload blocked! Loaded ammo is already full ({env.loaded_ammo}/{env.max_ammo}). Choose another action.")
                                print_valid_action_hints()
                                continue
                            if env.reserve_ammo <= 0:
                                print(f"Reload blocked by reserve! Reserve ammo is 0. Choose another action.")
                                print_valid_action_hints()
                                continue
                            
                        # If validation passed
                        action = temp_action
                        if user_choice != "":
                            if user_choice in numpad_map:
                                print(f"Action overridden by user to: {action_names[action]} (Numpad input: {user_choice})")
                            else:
                                print(f"Action overridden by user to: {action_names[action]}")
                        else:
                            print(f"Executing recommended action: {action_names[action]}")
                        break
                        
                    if user_choice == ".":
                        break
                
                # Perform action in game environment
                next_state, reward, term, trunc, info = env.step(action)
                done = term or trunc
                
                # Use corrected action from env guard if available
                actual_action = info.get("action", action)
                
                episode_reward += reward
                
                # Store experience in replay memory
                agent.remember(state, actual_action, reward, next_state, done)
                
                # Perform optimization step via replay
                loss = agent.replay(batch_size=batch_size)
                
                # Synchronize target network weights
                if total_step_count % update_target_steps == 0:
                    agent.update_target_network()
                    print("Target network weights synchronized.")
                    
                if loss > 0:
                    print(f"Step {step_idx} - Action: {action}, Reward: {reward:.2f}, Loss: {loss:.4f}")
                else:
                    print(f"Step {step_idx} - Action: {action}, Reward: {reward:.2f} (Filling Memory...)")
                    
                state = next_state
                
                if done:
                    break
                    
                time.sleep(1.0)
                
            # Decay exploration factor
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print(f"Episode {ep} Finished. Total Reward Collected: {episode_reward:.2f}")
            
            # Save model weights at the end of every episode for safety
            agent.save("models/model.pth")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model weights before exiting...")
        agent.save("models/model.pth")
        print("Model weights successfully saved after emergency interruption.")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}. Saving model weights before crash...")
        agent.save("models/model.pth")
        raise e
        
    print("\n=== SHOTGUN KING DQN TRAINING COMPLETED ===")


def load_config(config_path="config.yaml"):
    """Loads configuration options from a local YAML file.

    Args:
        config_path: System filesystem location to the configuration.
    """
    import os
    import yaml

    default_config = {
        "episodes": 50,
        "batch_size": 16,
        "max_steps_per_episode": 30,
        "mode": "autonomous",
        "learning_rate": 0.001,
        "epsilon_decay": 0.95,
        "epsilon_min": 0.05,
        "rank": 1,
        "weapon_type": 0
    }

    if not os.path.exists(config_path):
        print(f"Configuration file '{config_path}' not found. Using default parameters.")
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                print("Warning: Configuration file format invalid. Using defaults.")
                return default_config
            merged = default_config.copy()
            merged.update(cfg)
            print(f"Loaded configuration from '{config_path}': {merged}")
            return merged
    except Exception as e:
        print(f"Error loading configuration '{config_path}': {e}. Using defaults.")
        return default_config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shotgun King RL Training Loop")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the training configuration YAML file")
    parser.add_argument("--mode", type=str, default=None,
                        help="Execution mode override (autonomous or suggest)")
    args = parser.parse_args()

    if np is not None:
        config = load_config(args.config)
        if args.mode is not None:
            config["mode"] = args.mode
        train_dqn(
            episodes=config["episodes"],
            batch_size=config["batch_size"],
            max_steps_per_episode=config["max_steps_per_episode"],
            mode=config["mode"],
            learning_rate=config["learning_rate"],
            epsilon_decay=config["epsilon_decay"],
            epsilon_min=config["epsilon_min"]
        )
    else:
        print("Error: Numpy is not available.")
