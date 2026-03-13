import unittest
import torch
import numpy as np
from agent.dqn_agent import DQNAgent

class TestPhase3Artifacts(unittest.TestCase):
    """
    Verification tests for Phase 3 artifacts: training logic and loop.
    """
    
    def test_train_step_updates_weights(self):
        """Verify that agent.train_step actually updates the network weights."""
        agent = DQNAgent(lr=0.1) # Large LR to ensure measurable change
        
        # Collect some dummy data to fill the buffer
        for _ in range(100):
            state = np.zeros(24, dtype=np.float32)
            action = 0
            reward = 1.0
            next_state = np.zeros(24, dtype=np.float32)
            done = False
            agent.store_transition(state, action, reward, next_state, done)
            
        # Capture initial weights
        initial_weights = agent.q_network.fc3.weight.clone().detach()
        
        # Perform training step
        loss = agent.train_step(batch_size=32)
        self.assertGreater(loss, 0, "Loss should be positive for non-zero reward")
        
        # Verify weights have changed
        updated_weights = agent.q_network.fc3.weight.clone().detach()
        self.assertFalse(torch.equal(initial_weights, updated_weights), "Weights should have changed after training step")

    def test_train_dqn_script_runs(self):
        """Verify that the training script can be imported and executed at a small scale."""
        # This test ensures that the plumbing in train_dqn.py is correct
        import training.train_dqn as trainer
        
        # Test a very short run (2 episodes, small steps)
        # Instead of calling trainer.train directly (which might take time),
        # we check the logic integrity.
        try:
            trainer.train(num_episodes=1)
        except Exception as e:
            self.fail(f"train_dqn.py failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
