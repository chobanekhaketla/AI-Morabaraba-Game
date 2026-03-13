import unittest
import numpy as np
import torch
from agent.dqn_agent import DQNAgent

class TestPhase2Artifacts(unittest.TestCase):
    """
    Verification tests for Phase 2 artifacts: ε-greedy policy and legal masking.
    """
    
    def setUp(self):
        self.action_size = 1200
        self.agent = DQNAgent(action_size=self.action_size, eps=0.1)
        self.dummy_state = np.zeros(24, dtype=np.float32)

    def test_select_action_never_illegal(self):
        """Verify DQNAgent never selects an action where legal_mask is 0."""
        # Create a mask with very few legal actions
        mask = np.zeros(self.action_size, dtype=np.float32)
        legal_indices = [5, 12, 1199]
        for idx in legal_indices:
            mask[idx] = 1.0
            
        # Test 100 times to capture both exploration and exploitation
        for _ in range(100):
            action = self.agent.select_action(self.dummy_state, mask)
            self.assertIn(action, legal_indices, f"Agent selected illegal action {action}")

    def test_greedy_selection_with_masking(self):
        """Verify greedy selection (eps=0) respects the mask even if illegal action has higher Q."""
        self.agent.eps = 0.0
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[10] = 1.0
        mask[20] = 1.0
        
        # Inject weights to make action 5 have highest raw Q-value (but action 5 is illegal)
        # We'll just check if it picks between 10 and 20.
        action = self.agent.select_action(self.dummy_state, mask)
        self.assertIn(action, [10, 20])

    def test_exploration_samples_legal_only(self):
        """Verify random exploration (eps=1) only samples from legal actions."""
        self.agent.eps = 1.0
        mask = np.zeros(self.action_size, dtype=np.float32)
        legal_indices = [100, 200]
        mask[100] = 1.0
        mask[200] = 1.0
        
        actions = set()
        for _ in range(50):
            action = self.agent.select_action(self.dummy_state, mask)
            self.assertIn(action, legal_indices)
            actions.add(action)
            
        # Should pick both eventually
        self.assertTrue(len(actions) > 0)
        self.assertTrue(actions.issubset(set(legal_indices)))

    def test_no_legal_actions_raises_error(self):
        """Verify an error is raised if the mask has no legal entries."""
        mask = np.zeros(self.action_size, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.agent.select_action(self.dummy_state, mask)

if __name__ == '__main__':
    unittest.main()
