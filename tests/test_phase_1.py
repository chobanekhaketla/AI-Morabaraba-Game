import unittest
import torch
import numpy as np
from agent.networks.q_network import QNetwork
from agent.dqn_agent import DQNAgent

class TestPhase1Artifacts(unittest.TestCase):
    """
    Verification tests for Phase 1 artifacts: QNetwork and DQNAgent integration.
    """
    
    def test_q_network_shapes(self):
        """Verify QNetwork input and output shapes."""
        state_size = 24
        action_size = 1200
        batch_size = 32
        
        net = QNetwork(state_size=state_size, action_size=action_size)
        dummy_input = torch.randn(batch_size, state_size)
        output = net(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, action_size), f"Expected shape (32, 1200), got {output.shape}")

    def test_q_network_determinism(self):
        """Verify QNetwork initialization is deterministic with the same seed."""
        net1 = QNetwork(seed=42)
        net2 = QNetwork(seed=42)
        
        # Check a few weights
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Network weights should be identical for the same seed")

    def test_dqn_agent_network_integration(self):
        """Verify DQNAgent initializes a QNetwork and can perform a forward pass."""
        agent = DQNAgent()
        self.assertTrue(hasattr(agent, 'q_network'), "DQNAgent should have a q_network attribute")
        self.assertIsInstance(agent.q_network, QNetwork, "dqn_agent.q_network should be an instance of QNetwork")
        
        # Test select_action forward pass
        dummy_state = np.zeros(24, dtype=np.float32)
        dummy_mask = np.ones(1200, dtype=np.float32)
        
        # Should not crash and return an int
        action = agent.select_action(dummy_state, dummy_mask)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 1200)

if __name__ == '__main__':
    unittest.main()
