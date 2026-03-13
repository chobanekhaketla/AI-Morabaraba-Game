import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from .networks.q_network import QNetwork
from .replay_buffer import ReplayBuffer

class DQNAgent:
    """
    DQN Agent for Morabaraba.
    Handles action selection, experience storage, and training.
    """
    def __init__(self, action_size: int = 1200, buffer_capacity: int = 100000, 
                 hidden_size: int = 256, eps: float = 0.1, lr: float = 1e-4, 
                 target_update_frequency: int = 1000):
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_capacity)
        self.eps = eps
        self.target_update_frequency = target_update_frequency
        self.training_step_count = 0
        
        # Initialize Q-Networks (Online and Target)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_size=24, action_size=action_size, hidden_size=hidden_size).to(self.device)
        self.q_network_target = QNetwork(state_size=24, action_size=action_size, hidden_size=hidden_size).to(self.device)
        
        # Sync weights and set target to evaluation mode
        self.update_target_network()
        self.q_network_target.eval()
        
        # Optimizer and Loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        """Copies weights from the online network to the target network."""
        self.q_network_target.load_state_dict(self.q_network.state_dict())

    def select_action(self, state_encoding: np.ndarray, legal_mask: np.ndarray) -> int:
        """Selects an action using ε-greedy policy and legal masking."""
        legal_indices = np.where(legal_mask == 1.0)[0]
        if len(legal_indices) == 0:
            raise ValueError("No legal actions available according to the mask.")

        if random.random() < self.eps:
            return int(random.choice(legal_indices))

        state_tensor = torch.from_numpy(state_encoding).float().unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy().squeeze()
        self.q_network.train()
        
        masked_q_values = np.where(legal_mask == 1.0, q_values, -np.inf)
        return int(np.argmax(masked_q_values))

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Saves experience in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int = 64, gamma: float = 0.99) -> float:
        """
        Performs a single gradient descent step using a Target Network.
        Returns: The loss value as a float.
        """
        if not self.memory.is_ready(batch_size):
            return 0.0

        experiences = self.memory.sample(batch_size)
        
        # 1. Unpack and convert to tensors
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # 2. Get current Q-values for selected actions (Online Network)
        curr_q_values = self.q_network(states).gather(1, actions)
        
        # 3. Compute target Q-values (Bellman Equation using Target Network)
        with torch.no_grad():
            max_next_q = self.q_network_target(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + (gamma * max_next_q * (1 - dones))
            
        # 4. Compute Loss and Optimize
        loss = self.criterion(curr_q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 5. Periodic Target Update
        self.training_step_count += 1
        if self.training_step_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        return loss.item()

    def save(self, path: str):
        """Saves the online network weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str):
        """Loads weights into both online and target networks."""
        state_dict = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.q_network_target.load_state_dict(state_dict)
