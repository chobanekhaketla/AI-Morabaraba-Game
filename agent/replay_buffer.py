import numpy as np
import random
from typing import Tuple, List, Any

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training.
    Stores transitions in a circular buffer (FIFO eviction).
    """
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        """Samples a random batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Checks if the buffer has enough samples for training."""
        return len(self.buffer) >= batch_size
