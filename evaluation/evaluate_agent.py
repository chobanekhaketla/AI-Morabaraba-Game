import os
import sys
import numpy as np
import json
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.env import MorabarabaEnv
from agent.random_agent import RandomLegalAgent
from selfplay.self_play_loop import run_self_play_match

def evaluate_agent(agent: Any, num_episodes: int = 100, opponent: Any = None) -> Dict[str, Any]:
    """
    Evaluates an agent against an opponent (default: RandomLegalAgent).
    Tracks win rate, draw rate, reward, and episode length.
    """
    if opponent is None:
        opponent = RandomLegalAgent()
        
    env = MorabarabaEnv(max_moves=300)
    
    stats = {
        "win_p1": 0,
        "win_p2": 0,
        "draws": 0,
        "losses": 0,
        "total_reward": 0.0,
        "total_steps": 0,
        "termination_reasons": {}
    }
    
    for i in range(num_episodes):
        # Alternate roles
        if i % 2 == 0:
            p1, p2 = agent, opponent
            agent_role = 1
        else:
            p1, p2 = opponent, agent
            agent_role = -1
            
        info = run_self_play_match(env, p1, p2)
        winner = info.get("winner")
        reason = info.get("termination_reason", "unknown")
        
        # Track stats
        if winner == agent_role:
            if agent_role == 1:
                stats["win_p1"] += 1
            else:
                stats["win_p2"] += 1
        elif winner is None:
            stats["draws"] += 1
        else:
            stats["losses"] += 1
            
        stats["total_steps"] += info.get("move_count", 0)
        stats["termination_reasons"][reason] = stats["termination_reasons"].get(reason, 0) + 1
        
    # Calculate averages
    num_wins = stats["win_p1"] + stats["win_p2"]
    stats["win_rate"] = num_wins / num_episodes
    stats["draw_rate"] = stats["draws"] / num_episodes
    stats["avg_steps"] = stats["total_steps"] / num_episodes
    
    return stats

def print_report(stats: Dict[str, Any]):
    """Pretty prints the evaluation report."""
    print("\n" + "="*40)
    print(" EVALUATION REPORT")
    print("="*40)
    print(f"Total Episodes:  {stats.get('win_p1') + stats.get('win_p2') + stats.get('draws') + stats.get('losses')}")
    print(f"Win Rate:        {stats['win_rate']:.2%}")
    print(f"  - As Player 1: {stats['win_p1']}")
    print(f"  - As Player 2: {stats['win_p2']}")
    print(f"Draw Rate:       {stats['draw_rate']:.2%}")
    print(f"Loss Rate:       {(stats['losses'] / (stats['win_p1'] + stats['win_p2'] + stats['draws'] + stats['losses'])):.2%}")
    print(f"Avg Steps:       {stats['avg_steps']:.1f}")
    print("\nTermination Reasons:")
    for reason, count in stats["termination_reasons"].items():
        print(f"  - {reason:15s}: {count}")
    print("="*40 + "\n")

if __name__ == "__main__":
    from agent.dqn_agent import DQNAgent
    # Evaluate a fresh DQNAgent (random weights) vs. RandomLegalAgent
    test_agent = DQNAgent(eps=0.0) # Greedy inference
    results = evaluate_agent(test_agent, num_episodes=10)
    print_report(results)
