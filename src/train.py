"""Training loop script for Shotgun King reinforcement learning.

This script coordinates the custom Gymnasium environment and the DQN agent
to run training episodes, perform replay updates, and optimize decisions.
"""

import time

try:
    import numpy as np
except ImportError:
    np = None

from env import ShotgunKingEnv
from agent import DQNAgent


def train_dqn(episodes=2, batch_size=16, max_steps_per_episode=10, mode="autonomous"):
    """Executes the DQN training loop over a specified number of episodes.

    Args:
        episodes: Total number of game episodes to run.
        batch_size: Mini-batch size for DQN experience replay.
        max_steps_per_episode: Maximum steps limit per episode.
    """
    print("Initializing Shotgun King Gymnasium Environment...")
    env = ShotgunKingEnv(window_title="Shotgun King", max_steps=max_steps_per_episode)
    
    print("Initializing DQN Agent...")
    # 8x8 input flat is 64, with threat map flat is 128, plus 2 ammo dimensions is 130, plus 3 weapon specs is 133
    agent = DQNAgent(state_size=133, action_size=10, lr=1e-3)
    
    # Auto-load existing model weights if available to resume continuous learning
    agent.load("models/model.pth")
    
    # Exploration parameters
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.95
    
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
                
                # Select action
                action = agent.act(state, epsilon=epsilon)

                # Handle decision suggestion and override in suggest mode
                if mode == "suggest" and agent.policy_net is not None:
                    import torch
                    state_tensor = torch.tensor(state.astype(np.float32)).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
                    
                    action_names = {
                        0: "Move Up-Left", 1: "Move Up", 2: "Move Up-Right",
                        3: "Move Left", 4: "Move Right",
                        5: "Move Down-Left", 6: "Move Down", 7: "Move Down-Right",
                        8: "Reload", 9: "Shoot"
                    }
                    
                    top_actions = np.argsort(q_values)[::-1][:3]
                    print("\n=== AI Decision Suggestion ===")
                    for rank, act_idx in enumerate(top_actions, 1):
                        print(f"Rank {rank}: {action_names[act_idx]} (Q-value: {q_values[act_idx]:.4f})")
                    
                    user_choice = input(f"Recommended: [{action_names[action]}]. Press Enter to confirm, or enter custom action ID (0-9/Numpad): ").strip()
                    if user_choice == ".":
                        print("User flagged defeat. Forcing episode termination...")
                        agent.remember(state, action, -15.0, state, True)
                        break
                    
                    # Convert Numpad inputs to internal action index
                    numpad_map = {
                        "7": 0, "8": 1, "9": 2,
                        "4": 3, "5": 8, "6": 4,
                        "1": 5, "2": 6, "3": 7,
                        "0": 9
                    }
                    if user_choice in numpad_map:
                        action = numpad_map[user_choice]
                        print(f"Action overridden by user to: {action_names[action]} (Numpad input: {user_choice})")
                    elif user_choice.isdigit() and int(user_choice) in range(10):
                        action = int(user_choice)
                        print(f"Action overridden by user to: {action_names[action]}")
                    else:
                        print(f"Executing recommended action: {action_names[action]}")
                
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
                    
                time.sleep(3.0)
                
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shotgun King RL Training Loop")
    parser.add_argument("--mode", type=str, default="autonomous", choices=["autonomous", "suggest"],
                        help="Execution mode: autonomous control or user-approved suggestion mode")
    parser.add_argument("--episodes", type=int, default=50, help="Total training episodes")
    args = parser.parse_args()

    if np is not None:
        train_dqn(episodes=args.episodes, batch_size=16, max_steps_per_episode=30, mode=args.mode)
    else:
        print("Error: Numpy is not available.")
