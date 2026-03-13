import os
import sys
import numpy as np
import random
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.env import MorabarabaEnv
from engine.action_space import legal_action_mask
from agent.dqn_agent import DQNAgent
from agent.random_agent import RandomLegalAgent

def run_self_play_match(env: MorabarabaEnv, agent1: Any, agent2: Any, render: bool = False) -> Dict[str, Any]:
    """
    Runs a single match between two agents.
    agent1 is Player 1 (1.0), agent2 is Player 2 (-1.0).
    """
    state_dict = env.reset()
    done = False
    steps = 0
    
    agents = {1: agent1, -1: agent2}
    
    while not done:
        curr_player = env.game.current_player.value
        agent = agents[curr_player]
        
        # 1. Prepare inputs
        state_encoding = env.get_encoding()
        mask = legal_action_mask(state_dict)
        
        # 2. Select action
        # RandomLegalAgent and DQNAgent have different select_action signatures or logic
        if isinstance(agent, RandomLegalAgent):
            action_idx = agent.select_action(state_encoding, mask)
        else:
            # DQNAgent
            action_idx = agent.select_action(state_encoding, mask)
            
        # 3. Step
        state_dict, reward, done, info = env.step(action_idx)
        steps += 1
        
        if render:
            # Future: add render logic if needed
            pass
            
    return info

def tournament(num_matches: int = 10):
    """
    Runs a small tournament: DQN (eps=0) vs. Random.
    """
    env = MorabarabaEnv(max_moves=300)
    dqn_agent = DQNAgent(eps=0.0) # Greedy
    random_agent = RandomLegalAgent()
    
    results = {"dqn_wins": 0, "random_wins": 0, "draws": 0}
    
    print(f"Starting tournament ({num_matches} matches): DQN vs Random")
    
    for i in range(num_matches):
        # Alternate who goes first
        if i % 2 == 0:
            p1, p2 = dqn_agent, random_agent
            dqn_role = 1
        else:
            p1, p2 = random_agent, dqn_agent
            dqn_role = -1
            
        info = run_self_play_match(env, p1, p2)
        winner = info.get("winner")
        
        if winner == dqn_role:
            results["dqn_wins"] += 1
            res_str = "DQN WIN"
        elif winner is None:
            results["draws"] += 1
            res_str = "DRAW"
        else:
            results["random_wins"] += 1
            res_str = "RANDOM WIN"
            
        print(f"Match {i+1:2d}: {res_str:10s} | Steps: {info.get('move_count', 0):3d} | Reason: {info.get('termination_reason')}")
        
    print(f"\nResults: {results}")
    return results

if __name__ == "__main__":
    tournament(num_matches=5)
