import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    MLP-based Q-Network for Morabaraba DQN.
    Maps a 24-position board state to 1200 Q-values (one for each action).
    """
    def __init__(self, state_size: int = 24, action_size: int = 1200, hidden_size: int = 256, seed: int = 42):
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass.
        Args:
            state: Tensor of shape (batch_size, 24) representing the board encoding.
        Returns:
            Q-values: Tensor of shape (batch_size, 1200).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
