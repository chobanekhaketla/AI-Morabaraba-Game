import unittest
import os
import torch
import numpy as np
from agent.dqn_agent import DQNAgent
from agent.random_agent import RandomLegalAgent
from engine.env import MorabarabaEnv
from selfplay.self_play_loop import run_self_play_match

class TestPhase5Artifacts(unittest.TestCase):
    """
    Verification tests for Phase 5 artifacts: save/load and self-play.
    """
    
    def setUp(self):
        self.save_path = "tmp/test_model.pth"
        os.makedirs("tmp", exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def test_save_load_weights(self):
        """Verify that DQNAgent correctly saves and loads weight state."""
        agent1 = DQNAgent()
        agent2 = DQNAgent()
        
        # Manually perturb agent2 weights to ensure it starts differently from agent1
        # (Since QNetwork has a default seed of 42)
        with torch.no_grad():
            for p in agent2.q_network.parameters():
                p.add_(torch.randn_like(p))
                
        # Verify initial weights are now different
        is_different = False
        for p1, p2 in zip(agent1.q_network.parameters(), agent2.q_network.parameters()):
            if not torch.allclose(p1, p2):
                is_different = True
                break
        self.assertTrue(is_different, "Agent networks should be different before load")
        
        # Save agent1 and load into agent2
        agent1.save(self.save_path)
        agent2.load(self.save_path)
        
        # Verify weights are now identical
        for p1, p2 in zip(agent1.q_network.parameters(), agent2.q_network.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Loaded weights should be identical to saved weights")
            
        # Verify target network also updated on load
        for p1, p2 in zip(agent1.q_network.parameters(), agent2.q_network_target.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Target network should also be updated on load")

    def test_self_play_loop_integrity(self):
        """Verify that two agents can complete a match without errors."""
        env = MorabarabaEnv(max_moves=50) # Small limit for test
        dqn_agent = DQNAgent(eps=0.1)
        random_agent = RandomLegalAgent()
        
        try:
            # 1. DQN vs Random
            info = run_self_play_match(env, dqn_agent, random_agent)
            self.assertTrue(info["game_over"])
            
            # 2. DQN vs DQN
            info = run_self_play_match(env, dqn_agent, dqn_agent)
            self.assertTrue(info["game_over"])
        except Exception as e:
            self.fail(f"Self-play match failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
