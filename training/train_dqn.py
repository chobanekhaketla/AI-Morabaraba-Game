import os
import sys
try:
    import mpmath
    sys.modules['sympy.mpmath'] = mpmath
except ImportError:
    pass

import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.env import MorabarabaEnv
from engine.action_space import legal_action_mask
from agent.dqn_agent import DQNAgent

def train(num_episodes: int = 50, batch_size: int = 32, gamma: float = 0.99):
    """
    Runs a minimal training loop for the Morabaraba DQN agent.
    """
    env = MorabarabaEnv(max_moves=200)
    agent = DQNAgent(eps=0.2, lr=1e-3)
    
    print(f"Starting minimal training loop ({num_episodes} episodes)...")
    
    for ep in range(num_episodes):
        state_dict = env.reset()
        done = False
        total_reward = 0.0
        episode_loss = []
        steps = 0
        
        while not done:
            # 1. Prepare inputs
            state_encoding = env.get_encoding()
            mask = legal_action_mask(state_dict)
            
            # 2. Select and perform action
            action_idx = agent.select_action(state_encoding, mask)
            next_state_dict, reward, done, info = env.step(action_idx)
            next_encoding = env.get_encoding()
            
            # 3. Store transition
            agent.store_transition(state_encoding, action_idx, reward, next_encoding, done)
            
            # 4. Train step
            loss = agent.train_step(batch_size=batch_size, gamma=gamma)
            if loss > 0:
                episode_loss.append(loss)
                
            state_dict = next_state_dict
            total_reward += reward
            steps += 1
            
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        print(f"Ep {ep+1}/{num_episodes} | Steps: {steps:3d} | Reward: {total_reward:5.1f} | Loss: {avg_loss:.4f} | Result: {info['termination_reason']}")

    # Final save
    agent.save("models/dqn_final.pth")
    print("\nMinimal training complete. Saved to models/dqn_final.pth")

if __name__ == "__main__":
    train(num_episodes=20)
