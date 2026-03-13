import unittest
import numpy as np
from agent.dqn_agent import DQNAgent
from agent.random_agent import RandomLegalAgent
from evaluation.evaluate_agent import evaluate_agent

class TestPhase6Artifacts(unittest.TestCase):
    """
    Verification tests for Phase 6 artifacts: evaluation and metrics.
    """
    
    def test_evaluation_stats_consistency(self):
        """Verify that evaluation stats are internally consistent."""
        agent = DQNAgent(eps=1.0) # Random behavior for speed
        num_episodes = 4
        stats = evaluate_agent(agent, num_episodes=num_episodes)
        
        # 1. Check counts sum to total
        total_counted = stats["win_p1"] + stats["win_p2"] + stats["draws"] + stats["losses"]
        self.assertEqual(total_counted, num_episodes)
        
        # 2. Check rates
        self.assertAlmostEqual(stats["win_rate"] + stats["draw_rate"] + (stats["losses"] / num_episodes), 1.0)
        
        # 3. Check step averages
        total_steps = stats["total_steps"]
        self.assertEqual(stats["avg_steps"], total_steps / num_episodes)

    def test_termination_reasons_tracking(self):
        """Verify that all matches have a termination reason logged."""
        agent = DQNAgent(eps=1.0)
        num_episodes = 2
        stats = evaluate_agent(agent, num_episodes=num_episodes)
        
        num_reasons = sum(stats["termination_reasons"].values())
        self.assertEqual(num_reasons, num_episodes)

if __name__ == '__main__':
    unittest.main()
