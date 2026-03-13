import os
import sys
import numpy as np

# Add project root to path so we can import engine and agent
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from engine.env import MorabarabaEnv
from engine.action_space import legal_action_mask
from agent.random_agent import RandomLegalAgent
from agent.replay_buffer import ReplayBuffer

def run_rollouts(num_episodes: int = 10, buffer_capacity: int = 1000):
    """
    Runs several episodes of the game using a random agent 
    and fills the replay buffer.
    """
    env = MorabarabaEnv(max_moves=300)
    agent = RandomLegalAgent()
    buffer = ReplayBuffer(capacity=buffer_capacity)
    
    print(f"Starting {num_episodes} rollouts...")
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            # 1. Get state encoding and mask
            encoding = env.get_encoding()
            mask = legal_action_mask(state)
            
            # 2. Agent selects action
            action_idx = agent.select_action(encoding, mask)
            
            # 3. Environment Step
            next_state, reward, done, info = env.step(action_idx)
            next_encoding = env.get_encoding()
            
            # 4. Store in buffer
            buffer.push(encoding, action_idx, reward, next_encoding, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
        print(f"Episode {ep + 1}/{num_episodes} - Steps: {steps}, Reward: {total_reward:.1f}, Result: {info.get('termination_reason', 'win_loss')}")

    print(f"\nRollouts complete. Buffer size: {len(buffer)}")
    
if __name__ == "__main__":
    run_rollouts(num_episodes=5)
