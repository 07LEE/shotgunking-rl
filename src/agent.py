"""Deep Q-Network (DQN) agent module for Shotgun King reinforcement learning.

This module defines the neural network architecture, experience replay buffer,
and the decision-making DQN agent using PyTorch.
"""

import os
import random
from collections import deque

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None


class QNetwork(nn.Module if nn is not None else object):
    """Deep Q-Network MLP neural network."""

    def __init__(self, state_size=281, action_size=10):
        """Initializes the network layers.

        Args:
            state_size: Flattened input state vector dimension (133).
            action_size: Number of discrete action choices (10).
        """
        super().__init__()
        if nn is not None:
            self.fc1 = nn.Linear(state_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, action_size)
            self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x: Input state tensor.

        Returns:
            Output action Q-values.
        """
        if nn is None:
            return x
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience Replay Buffer for stable DQN training."""

    def __init__(self, capacity=10000):
        """Initializes the buffer container.

        Args:
            capacity: Maximum number of experiences to hold.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, action_mask, next_action_mask, done):
        """Saves a single experience transition.

        Args:
            state: Current state matrix.
            action: Action index.
            reward: Reward value.
            next_state: Next state matrix.
            action_mask: Current action mask.
            next_action_mask: Next action mask.
            done: Termination flag.
        """
        self.buffer.append((state, action, reward, next_state, action_mask, next_action_mask, done))

    def sample(self, batch_size):
        """Samples a random batch of experiences.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A tuple of (states, actions, rewards, next_states, action_masks, next_action_masks, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, action_masks, next_action_masks, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(action_masks, dtype=np.float32),
            np.array(next_action_masks, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network decision-making agent."""

    def __init__(self, state_size=281, action_size=10, lr=1e-3, gamma=0.99):
        """Initializes the agent parameters, networks, and optimizer.

        Args:
            state_size: Flattened input state vector dimension (133).
            action_size: Number of discrete action choices.
            lr: Learning rate for training.
            gamma: Discount factor for future rewards.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.memory = ReplayBuffer(capacity=5000)
        
        if torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = QNetwork(state_size, action_size).to(self.device)
            self.target_net = QNetwork(state_size, action_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.criterion = nn.MSELoss()
        else:
            self.device = None
            self.policy_net = None
            self.target_net = None

    def act(self, state, action_mask=None, epsilon=0.1):
        """Chooses an action based on epsilon-greedy policy.

        Args:
            state: Current 8x8 state representation.
            action_mask: Current action mask.
            epsilon: Epsilon threshold for random exploration.

        Returns:
            Chosen action index (integer).
        """
        if action_mask is None:
            action_mask = np.ones(self.action_size, dtype=np.float32)

        if random.random() < epsilon or self.policy_net is None:
            valid_indices = np.where(action_mask == 1)[0]
            if len(valid_indices) > 0:
                return int(random.choice(valid_indices))
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.tensor(state.astype(np.float32)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            q_values[action_mask == 0] = -1e9
            return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, action_mask, next_action_mask, done):
        """Stores experience transition into replay memory."""
        self.memory.push(state, action, reward, next_state, action_mask, next_action_mask, done)

    def update_target_network(self):
        """Synchronizes the target network weights with policy network."""
        if self.target_net is not None and self.policy_net is not None:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def replay(self, batch_size=32):
        """Trains the policy network weights using sampled batch from memory.

        Args:
            batch_size: Mini-batch size.

        Returns:
            Calculated loss value (float), or 0.0 if not trained.
        """
        if len(self.memory) < batch_size or self.policy_net is None:
            return 0.0

        states, actions, rewards, next_states, action_masks, next_action_masks, dones = self.memory.sample(batch_size)

        state_t = torch.tensor(np.array(states, dtype=np.float32)).to(self.device)
        action_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_t = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_state_t = torch.tensor(np.array(next_states, dtype=np.float32)).to(self.device)
        next_action_masks_t = torch.tensor(np.array(next_action_masks, dtype=np.float32)).to(self.device)
        done_t = torch.tensor(dones).unsqueeze(1).to(self.device)

        # Calculate current predicted Q-values
        curr_q = self.policy_net(state_t).gather(1, action_t)

        with torch.no_grad():
            # Double DQN implementation
            next_q_policy = self.policy_net(next_state_t).clone()
            next_q_policy[next_action_masks_t == 0] = -1e9
            best_actions = next_q_policy.argmax(dim=1, keepdim=True)
            
            next_q_target = self.target_net(next_state_t)
            max_next_q = next_q_target.gather(1, best_actions)
            target_q = reward_t + (self.gamma * max_next_q * (1 - done_t))

        # Perform backpropagation
        loss = self.criterion(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())

    def save(self, filepath="models/model.pth"):
        """Saves the policy network weights to the specified file path."""
        if torch is not None and self.policy_net is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(self.policy_net.state_dict(), filepath)
            print(f"Successfully saved agent model weights to: '{filepath}'")

    def load(self, filepath="models/model.pth"):
        """Loads the policy network weights from the specified file path."""
        if torch is not None and self.policy_net is not None and os.path.exists(filepath):
            try:
                self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"Successfully loaded agent model weights from: '{filepath}'")
                return True
            except Exception as e:
                print(f"Warning: Failed to load existing weights due to dimension mismatch or corrupt file ({e}). Starting with fresh weights.")
                return False
        return False


if __name__ == "__main__":
    print("Testing DQNAgent initialization and forward pass...")
    if torch is not None and np is not None:
        agent = DQNAgent(state_size=281, action_size=10)
        dummy_state = np.zeros((281,), dtype=np.float32)
        
        action = agent.act(dummy_state, epsilon=0.0)
        print(f"Decided action for dummy state: {action}")
        
        # Test memory push and training step
        dummy_mask = np.ones((10,), dtype=np.float32)
        agent.remember(dummy_state, 1, 0.1, dummy_state, dummy_mask, dummy_mask, False)
        agent.remember(dummy_state, 2, 0.1, dummy_state, dummy_mask, dummy_mask, False)
        
        # Manually invoke replay with small batch size 2 for prototype validation
        loss_val = agent.replay(batch_size=2)
        print(f"DQN backpropagation prototype validation successful. Sample Loss: {loss_val}")
    else:
        print("Error: PyTorch is not available.")
