import unittest
import torch
import numpy as np
from agent.dqn_agent import DQNAgent

class TestPhase4Artifacts(unittest.TestCase):
    """
    Verification tests for Phase 4 artifacts: target network and stability.
    """
    
    def test_target_network_initialization(self):
        """Verify that DQNAgent has a target network with same weights as online network."""
        agent = DQNAgent()
        self.assertTrue(hasattr(agent, 'q_network_target'))
        
        # Verify weights are identical initially
        for p1, p2 in zip(agent.q_network.parameters(), agent.q_network_target.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Online and target networks should have same weights initially")

    def test_target_network_separation(self):
        """Verify that online network updates do not instantly affect target network."""
        agent = DQNAgent(lr=0.1)
        
        # Fill buffer with some dummy data
        for _ in range(10):
            agent.store_transition(np.zeros(24), 0, 1.0, np.zeros(24), False)
            
        # Initial target weights
        initial_target_weights = agent.q_network_target.fc3.weight.clone().detach()
        
        # Perform training step (updates online network)
        agent.train_step(batch_size=5)
        
        # Target weights should NOT have changed
        current_target_weights = agent.q_network_target.fc3.weight.clone().detach()
        self.assertTrue(torch.equal(initial_target_weights, current_target_weights), 
                        "Target network weights should remain frozen during training")
        
        # Online weights SHOULD have changed
        online_weights = agent.q_network.fc3.weight.clone().detach()
        self.assertFalse(torch.equal(online_weights, initial_target_weights), 
                         "Online network weights should be updated")

    def test_periodic_target_update(self):
        """Verify that target network is updated after target_update_frequency steps."""
        update_freq = 5
        agent = DQNAgent(target_update_frequency=update_freq)
        
        # Fill buffer
        for _ in range(10):
            agent.store_transition(np.zeros(24), 0, 1.0, np.zeros(24), False)
            
        # Initial weights
        initial_online_weights = agent.q_network.fc3.weight.clone().detach()
        
        # Perform 4 steps (no update yet)
        for _ in range(update_freq - 1):
            agent.train_step(batch_size=2)
            
        self.assertFalse(torch.equal(agent.q_network.fc3.weight, agent.q_network_target.fc3.weight),
                         "Target should not have updated yet")
        
        # 5th step (should trigger update)
        agent.train_step(batch_size=2)
        
        self.assertTrue(torch.equal(agent.q_network.fc3.weight, agent.q_network_target.fc3.weight),
                        "Target network should be updated after reaching frequency limit")

if __name__ == '__main__':
    unittest.main()
