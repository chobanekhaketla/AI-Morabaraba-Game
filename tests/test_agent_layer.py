"""
TEST SUITE FOR THE AGENT LAYER
Verifies ReplayBuffer, DQNAgent contract, and RandomLegalAgent integration.
"""

import unittest
import numpy as np
from agent.replay_buffer import ReplayBuffer
from agent.random_agent import RandomLegalAgent
from agent.dqn_agent import DQNAgent
from engine.env import MorabarabaEnv
from engine.action_space import legal_action_mask

class TestReplayBuffer(unittest.TestCase):
    """Tests for the experience replay buffer"""
    
    def test_buffer_push_and_storage(self):
        """Verify transitions are stored correctly."""
        capacity = 10
        buffer = ReplayBuffer(capacity=capacity)
        
        state = np.zeros(24)
        action = 5
        reward = 1.0
        next_state = np.ones(24)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        
        self.assertEqual(len(buffer), 1)
        stored_state, stored_action, stored_reward, stored_next_state, stored_done = buffer.buffer[0]
        
        np.testing.assert_array_equal(stored_state, state)
        self.assertEqual(stored_action, action)
        self.assertEqual(stored_reward, reward)
        np.testing.assert_array_equal(stored_next_state, next_state)
        self.assertEqual(stored_done, done)

    def test_buffer_fifo_eviction(self):
        """Verify FIFO behavior when capacity is exceeded."""
        capacity = 5
        buffer = ReplayBuffer(capacity=capacity)
        
        # Fill buffer
        for i in range(capacity):
            buffer.push(np.zeros(24), i, 0.0, np.zeros(24), False)
        
        self.assertEqual(len(buffer), capacity)
        self.assertEqual(buffer.buffer[0][1], 0)
        
        # Add another element, should evict index 0
        buffer.push(np.zeros(24), 99, 0.0, np.zeros(24), False)
        
        self.assertEqual(len(buffer), capacity)
        self.assertEqual(buffer.buffer[0][1], 99) # Replaced at position 0
        self.assertEqual(buffer.position, 1)

    def test_buffer_sampling(self):
        """Verify random sampling returns correct batch sizes and doesn't mutate."""
        capacity = 20
        buffer = ReplayBuffer(capacity=capacity)
        
        for i in range(capacity):
            buffer.push(np.zeros(24), i, 0.0, np.zeros(24), False)
            
        batch_size = 5
        sample = buffer.sample(batch_size)
        
        self.assertEqual(len(sample), batch_size)
        self.assertEqual(len(buffer), capacity) # Buffer size remains same

class TestRandomLegalAgent(unittest.TestCase):
    """Tests for the RandomLegalAgent"""
    
    def test_select_legal_action_only(self):
        """Verify agent only selects actions that are marked as legal."""
        agent = RandomLegalAgent()
        state = np.zeros(24)
        
        # Create a specific mask with only a few legal actions
        mask = np.zeros(1200)
        legal_indices = [10, 50, 100]
        for idx in legal_indices:
            mask[idx] = 1.0
            
        for _ in range(100):
            action = agent.select_action(state, mask)
            self.assertIn(action, legal_indices)
            self.assertEqual(mask[action], 1.0)

    def test_action_variety(self):
        """Verify agent samples across different legal actions."""
        agent = RandomLegalAgent()
        state = np.zeros(24)
        mask = np.zeros(1200)
        legal_indices = [1, 2, 3, 4, 5]
        for idx in legal_indices:
            mask[idx] = 1.0
            
        actions = set()
        for _ in range(100):
            actions.add(agent.select_action(state, mask))
            
        # With 100 tries, it's statistically nearly certain all 5 actions will be picked
        self.assertEqual(len(actions), 5)

class TestDQNAgentInterface(unittest.TestCase):
    """Tests for the DQNAgent interface contract"""
    
    def test_interface_methods_exist(self):
        """Verify DQNAgent has required methods."""
        agent = DQNAgent()
        self.assertTrue(hasattr(agent, 'select_action'))
        self.assertTrue(hasattr(agent, 'store_transition'))
        self.assertTrue(hasattr(agent, 'train_step'))

    def test_store_transition_integrity(self):
        """Verify store_transition correctly populates memory."""
        agent = DQNAgent(buffer_capacity=10)
        state = np.zeros(24)
        action = 1
        reward = 1.0
        next_state = np.zeros(24)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
        self.assertEqual(len(agent.memory), 1)

class TestAgentEnvIntegration(unittest.TestCase):
    """Integration tests for agent and environment"""
    
    def test_full_random_game_rollout(self):
        """Verify RandomLegalAgent can play a full game with termination."""
        env = MorabarabaEnv(max_moves=200) # Small move count for test speed
        agent = RandomLegalAgent()
        
        state = env.reset()
        done = False
        steps = 0
        
        while not done:
            encoding = env.get_encoding()
            mask = legal_action_mask(state)
            
            action = agent.select_action(encoding, mask)
            state, reward, done, info = env.step(action)
            steps += 1
            
            if steps > 205: # Buffer of 5 above max_moves
                self.fail("Game failed to terminate within max_moves limit.")
                
        self.assertTrue(done)
        self.assertTrue(info["game_over"])
        self.assertIn(info["termination_reason"], ["win_loss", "move_limit", "stalemate", "illegal_action"])

if __name__ == '__main__':
    unittest.main()
