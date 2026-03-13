import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def verify_phase3():
    print("--- Verifying Phase 3: Persistence & Instrumentation ---")
    
    # 1. Create a new game
    print("\n1. Creating new game...")
    response = requests.post(f"{BASE_URL}/games/new")
    if response.status_code != 200:
        print(f"FAILED: {response.text}")
        return
    
    game_data = response.json()
    game_id = game_data["game_id"]
    print(f"SUCCESS: game_id = {game_id}")
    
    # 2. Make some moves and check metrics
    print("\n2. Making moves and checking metrics...")
    
    # Move 1: White places at 0
    move1 = {"type": "place", "to": 0}
    res1 = requests.post(f"{BASE_URL}/games/{game_id}/move", json=move1)
    if res1.status_code == 200:
        metrics = res1.json().get("last_move_metrics")
        print(f"Move 1 (Place White 0) Metrics: {json.dumps(metrics, indent=2)}")
    else:
        print(f"Move 1 FAILED: {res1.text}")
        return

    # Move 2: Black places at 8
    move2 = {"type": "place", "to": 8}
    res2 = requests.post(f"{BASE_URL}/games/{game_id}/move", json=move2)
    
    # Move 3: White places at 1
    move3 = {"type": "place", "to": 1}
    res3 = requests.post(f"{BASE_URL}/games/{game_id}/move", json=move3)
    
    # Move 4: Black places at 9
    move4 = {"type": "place", "to": 9}
    res4 = requests.post(f"{BASE_URL}/games/{game_id}/move", json=move4)
    
    # Move 5: White places at 2 (Forms MILL)
    print("\n3. Forming a mill...")
    move5 = {"type": "place", "to": 2}
    res5 = requests.post(f"{BASE_URL}/games/{game_id}/move", json=move5)
    if res5.status_code == 200:
        data = res5.json()
        metrics = data.get("last_move_metrics")
        print(f"Move 5 (Place White 2 - MILL) Metrics: {json.dumps(metrics, indent=2)}")
        if metrics.get("mills_formed") == 1:
            print("SUCCESS: Mill formation detected!")
        else:
            print("FAILED: Mill formation NOT detected in metrics.")
    else:
        print(f"Move 5 FAILED: {res5.text}")
        return

    # 3. Check history
    print(f"\n4. Checking history for {game_id}...")
    hist_res = requests.get(f"{BASE_URL}/games/{game_id}/history")
    if hist_res.status_code == 200:
        history = hist_res.json()
        print(f"History length: {len(history)}")
        if len(history) == 5:
            print("SUCCESS: All moves persisted in DB!")
            print("First move board state sample:", history[0]["board_state"])
        else:
            print(f"FAILED: Expected 5 moves, found {len(history)}")
    else:
        print(f"History retrieval FAILED: {hist_res.text}")

    # 4. Check stats
    print(f"\n5. Checking stats for {game_id}...")
    stats_res = requests.get(f"{BASE_URL}/games/{game_id}/stats")
    if stats_res.status_code == 200:
        stats = stats_res.json()
        print(f"Stats: {json.dumps(stats, indent=2)}")
        if stats["mills_per_player"]["WHITE"] >= 1:
            print("SUCCESS: Stats accurately reflecting mill count!")
        else:
            print(f"FAILED: Stats mill count incorrect. Got: {stats['mills_per_player']}")
    else:
        print(f"Stats retrieval FAILED: {stats_res.text}")

if __name__ == "__main__":
    verify_phase3()
