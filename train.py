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
from weapons import WEAPON_PRESETS
from shotgun_king_rl.runtime.scope import (
    assess_assistance_scope,
    format_assistance_scope,
)
from shotgun_king_rl.runtime.suggestions import (
    ACTION_NAMES,
    build_suggestion_view,
    format_suggestion_view,
)
from shotgun_king_rl.runtime.tracing import ActionTraceWriter, build_action_trace
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def remember_manual_defeat(agent, state, action, action_mask):
    """Record a user-confirmed defeat with a complete replay transition."""
    agent.remember(
        state,
        action,
        -15.0,
        state,
        action_mask,
        action_mask,
        True,
    )


def train_dqn(episodes=2, batch_size=16, max_steps_per_episode=10, mode="autonomous", learning_rate=1e-3, epsilon_decay=0.95, epsilon_min=0.05, rank=1, weapon_type=0, current_floor=1, buffs=None, verbose=True, trace_directory="data/runtime_traces"):
    """Executes the DQN training loop over a specified number of episodes.

    Args:
        episodes: Total number of game episodes to run.
        batch_size: Mini-batch size for DQN experience replay.
        max_steps_per_episode: Maximum steps limit per episode.
        mode: Game control execution mode.
        learning_rate: Optimizer parameter for step adjustments.
        epsilon_decay: Exploration factor multiplier.
        epsilon_min: Minimum exploration cutoff limit.
        rank: Configured story rank shown to the operator.
        weapon_type: Configured weapon preset ID.
        current_floor: Floor recorded for operator context.
        buffs: Optional active player buffs.
        verbose: Enable environment diagnostic output.
        trace_directory: Private directory for one JSONL action trace per run.
    """
    preset = WEAPON_PRESETS.get(weapon_type)
    weapon_name = preset["name"] if preset is not None else "Unknown weapon"
    scope = assess_assistance_scope(
        rank=rank,
        weapon_type=weapon_type,
        weapon_name=weapon_name,
        floor=current_floor,
        weapon_registered=preset is not None,
    )
    print("\n=== LIVE ASSISTANCE SCOPE ===")
    print(format_assistance_scope(scope))
    if not scope.supported:
        print("Runtime stopped before capture or input because this scope is unsupported.")
        return False

    trace_writer = ActionTraceWriter(trace_directory)
    print(f"Runtime action trace: {trace_writer.path}")

    writer = SummaryWriter('runs/shotgun_king_dqn') if SummaryWriter is not None else None
    print("Initializing Shotgun King Gymnasium Environment...")
    env = ShotgunKingEnv(
        window_title="Shotgun King",
        max_steps=max_steps_per_episode,
        weapon_type=weapon_type,
        rank=rank,
        buffs=buffs,
        verbose=verbose,
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
            if not info.get("reset_ready", True):
                print(f"Environment reset failed; stopping before input: {info}")
                break
            
            episode_reward = 0.0
            step_idx = 0
            
            while True:
                step_idx += 1
                total_step_count += 1
                
                # Calculate action mask for current state
                action_mask = env.get_action_mask()

                def is_action_valid(act):
                    return bool(action_mask[act])

                # Calculate Q-values once per step to optimize performance by avoiding duplicate feed-forwards
                q_values = None
                if agent.policy_net is not None:
                    import torch
                    state_tensor = torch.tensor(state.astype(np.float32)).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        q_values = agent.policy_net(state_tensor).cpu().numpy()[0]

                decision_view = build_suggestion_view(
                    state,
                    action_mask,
                    q_values,
                    loaded_ammo=env.loaded_ammo,
                    max_ammo=env.max_ammo,
                    reserve_ammo=env.reserve_ammo,
                    range_limit=env.range_limit,
                    royal_guard_active=env.royal_guard_active,
                    melee_damage=env.melee_damage,
                )

                # Decide action based on current state and mask
                action = agent.act(state, action_mask=action_mask, epsilon=epsilon)
                selection_source = "epsilon_greedy"

                # Handle decision suggestion and override in suggest mode
                if mode == "suggest":
                    if q_values is None:
                        raise RuntimeError(
                            "Suggest mode requires model Q-values before any input"
                        )
                    view = decision_view
                    print("\n" + format_suggestion_view(view))
                    if not view.recommendations:
                        raise RuntimeError(
                            "No valid recommendation is available for the observed state"
                        )
                    action = view.recommendations[0].action

                    # Convert Numpad inputs to internal action index
                    numpad_map = {
                        "7": 0, "8": 1, "9": 2,
                        "4": 3, "5": 8, "6": 4,
                        "1": 5, "2": 6, "3": 7,
                        "0": 9
                    }

                    def print_valid_action_hints():
                        valid_choices = [
                            f"{item.key} ({item.name})"
                            for item in view.valid_actions
                        ]
                        print(
                            "Valid choices for current board state: "
                            + ", ".join(valid_choices)
                        )

                    # Validation loop for user choice
                    while True:
                        user_choice = input(f"\nRecommended: [{ACTION_NAMES[action]}]. Press Enter to confirm, or enter custom action ID (0-9/Numpad/r): ").strip()
                        if user_choice == ".":
                            print("User flagged defeat. Forcing episode termination...")
                            remember_manual_defeat(agent, state, action, action_mask)
                            trace_writer.write(build_action_trace(
                                episode=ep,
                                step=step_idx,
                                mode=mode,
                                rank=rank,
                                floor=current_floor,
                                weapon_type=weapon_type,
                                weapon_name=weapon_name,
                                observation=state,
                                action_mask=action_mask,
                                q_values=q_values,
                                recommendations=view.recommendations,
                                selected_action=action,
                                selection_source="user_reported_defeat",
                                loaded_ammo=env.loaded_ammo,
                                max_ammo=env.max_ammo,
                                reserve_ammo=env.reserve_ammo,
                                next_observation=state,
                                reward=-15.0,
                                terminated=True,
                                truncated=False,
                                info={
                                    "action": None,
                                    "input_succeeded": None,
                                    "transition_ready": True,
                                    "outcome": "user_reported_defeat",
                                },
                                accepted_for_replay=True,
                            ))
                            break
                        
                        temp_action = action
                        if user_choice.lower() == 'r':
                            temp_action = 8
                        elif user_choice in numpad_map:
                            temp_action = numpad_map[user_choice]
                        elif user_choice.isdigit() and int(user_choice) in range(10):
                            temp_action = int(user_choice)
                        elif user_choice == "":
                            # Use recommended action
                            temp_action = action
                        else:
                            print("Invalid input. Please enter 0-9, Numpad keys, 'r', or '.' to exit.")
                            print_valid_action_hints()
                            continue

                        if temp_action >= len(action_mask) or not is_action_valid(temp_action):
                            print("That action is not permitted by the displayed action mask.")
                            print_valid_action_hints()
                            continue
                            
                        # If validation passed
                        action = temp_action
                        if user_choice != "":
                            selection_source = "user_override"
                            if user_choice.lower() == 'r':
                                print(f"Action overridden by user to: {ACTION_NAMES[action]} (r key input)")
                            elif user_choice in numpad_map:
                                print(f"Action overridden by user to: {ACTION_NAMES[action]} (Numpad input: {user_choice})")
                            else:
                                print(f"Action overridden by user to: {ACTION_NAMES[action]}")
                        else:
                            selection_source = "recommendation_confirmed"
                            print(f"Executing recommended action: {ACTION_NAMES[action]}")
                        break
                        
                    if user_choice == ".":
                        break
                
                pre_action_ammo = (
                    env.loaded_ammo,
                    env.max_ammo,
                    env.reserve_ammo,
                )

                # Perform action in game environment
                next_state, reward, term, trunc, info = env.step(action)
                done = term or trunc
                accepted_for_replay = bool(
                    info.get("input_succeeded", True)
                    and info.get("transition_ready", True)
                )
                trace_writer.write(build_action_trace(
                    episode=ep,
                    step=step_idx,
                    mode=mode,
                    rank=rank,
                    floor=current_floor,
                    weapon_type=weapon_type,
                    weapon_name=weapon_name,
                    observation=state,
                    action_mask=action_mask,
                    q_values=q_values,
                    recommendations=decision_view.recommendations,
                    selected_action=action,
                    selection_source=selection_source,
                    loaded_ammo=pre_action_ammo[0],
                    max_ammo=pre_action_ammo[1],
                    reserve_ammo=pre_action_ammo[2],
                    next_observation=next_state,
                    reward=reward,
                    terminated=term,
                    truncated=trunc,
                    info=info,
                    accepted_for_replay=accepted_for_replay,
                ))

                if not info.get("input_succeeded", True):
                    print(f"Input failed; transition was not added to replay memory: {info.get('input')}")
                    state = next_state
                    if done:
                        break
                    continue

                if not info.get("transition_ready", True):
                    print(f"Action completion was not verified; transition was not added to replay memory: {info.get('completion') or info.get('observation_issue')}")
                    state = next_state
                    if done:
                        break
                    continue
                
                # Use corrected action from env guard if available
                actual_action = info.get("action", action)
                
                episode_reward += reward
                
                # Calculate next action mask for replay updates
                next_action_mask = env.get_action_mask()

                # Store experience in replay memory with masks included
                agent.remember(state, actual_action, reward, next_state, action_mask, next_action_mask, term)
                
                # Perform optimization step via replay
                loss = agent.replay(batch_size=batch_size)
                
                # Synchronize target network weights
                if total_step_count % update_target_steps == 0:
                    agent.update_target_network()
                    print("Target network weights synchronized.")
                    
                if loss > 0:
                    print(f"Step {step_idx} - Action: {action}, Reward: {reward:.2f}, Loss: {loss:.4f}")
                    if writer is not None:
                        writer.add_scalar('Loss/train', loss, total_step_count)
                else:
                    print(f"Step {step_idx} - Action: {action}, Reward: {reward:.2f} (Filling Memory...)")
                    
                state = next_state
                
                if done:
                    break
                    
                time.sleep(1.0)
                
            # Decay exploration factor
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print(f"Episode {ep} Finished. Total Reward Collected: {episode_reward:.2f}")
            if writer is not None:
                writer.add_scalar('Reward/episode', episode_reward, ep)
                writer.add_scalar('Epsilon/episode', epsilon, ep)
            
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
    finally:
        if writer is not None:
            writer.close()
            print("TensorBoard SummaryWriter closed.")
        
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
        "weapon_type": 0,
        "current_floor": 1,
        "trace_directory": "data/runtime_traces",
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
            epsilon_min=config["epsilon_min"],
            rank=config["rank"],
            weapon_type=config["weapon_type"],
            current_floor=config["current_floor"],
            buffs=config.get("buffs"),
            verbose=config.get("verbose", True),
            trace_directory=config.get("trace_directory", "data/runtime_traces")
        )
    else:
        print("Error: Numpy is not available.")
