# Morabaraba DQN — Reinforcement Learning for a Traditional African Board Game

A complete Deep Q-Network (DQN) system for learning to play **Morabaraba**, a traditional
two-player African strategy game similar to Nine Men's Morris. The project is structured as a
headless game engine, a FastAPI backend, and an RL training pipeline.

---

## What is Morabaraba?

Morabaraba is a two-player strategy game played on a board of three concentric squares
connected at their mid-points (24 positions). Players alternate placing, moving, and
capturing pieces. Forming a **mill** (three pieces in a row) allows a player to remove one
of the opponent's pieces. A player loses when reduced to 2 pieces or unable to move.

**Phases:**

| Phase | Trigger | Allowed actions |
|-------|---------|-----------------|
| Placing | Start → 12 pieces each placed | Place a piece on any empty spot |
| Moving | After placing | Slide to an adjacent empty spot |
| Flying | Player has exactly 3 pieces | Jump to any empty spot |

---

## Project Structure

```
dqn_game/
├── engine/          # Headless, deterministic game engine
│   ├── board.py         # 24-position board, mill/adjacency definitions
│   ├── game.py          # Turn management, phase transitions, win detection
│   ├── rules.py         # Move validation per phase
│   ├── action_space.py  # Unified 1200-action flat space with legal masking
│   ├── env.py           # OpenAI-Gym-style RL environment wrapper
│   ├── reward_utils.py  # Deterministic per-move reward metrics
│   ├── constants.py     # Shared constants
│   └── errors.py        # Custom exception types
│
├── agent/           # DQN agent infrastructure
│   ├── dqn_agent.py     # DQN agent (action selection, learning step)
│   ├── replay_buffer.py # FIFO experience replay buffer
│   ├── random_agent.py  # Uniform-random legal-action agent (baseline)
│   └── networks/        # PyTorch Q-network definitions
│
├── app/             # FastAPI backend (game server + persistence)
│   ├── main.py          # App entry point
│   ├── routes.py        # Game management endpoints
│   ├── ai_routes.py     # AI / agent endpoints
│   ├── services.py      # Business logic bridging engine ↔ API
│   ├── schemas.py       # Pydantic request/response models
│   └── database.py      # SQLAlchemy/SQLite setup
│
├── training/        # Training orchestration
│   └── train_dqn.py
│
├── selfplay/        # Self-play loop
│   └── self_play_loop.py
│
├── evaluation/      # Agent evaluation vs baselines
│   └── evaluate_agent.py
│
├── scripts/         # Utility scripts
│   └── run_rollout.py   # Fills replay buffer & logs episode stats
│
├── static/          # Minimal web frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── game.js
│
├── tests/           # Comprehensive test suite (130+ tests)
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch (install separately; see below)

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/dqn_game.git
cd dqn_game

# Install core dependencies
pip install -r requirements.txt

# Install PyTorch (CPU build — visit pytorch.org for GPU variants)
pip install torch numpy
```

### Run Tests

```bash
pytest tests/
```

### Start the API Server

```bash
uvicorn app.main:app --reload
```

The server starts at `http://127.0.0.1:8000`. Interactive docs are at `/docs`.

### Train the DQN Agent

```bash
python training/train_dqn.py
```

Trained model weights are saved to `models/dqn_final.pth` (gitignored — reproduce locally).

### Evaluate the Agent

```bash
python evaluation/evaluate_agent.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/games/new` | Create a new game |
| `GET` | `/games/{id}` | Get current game state |
| `POST` | `/games/{id}/move` | Apply a move |
| `GET` | `/games/{id}/history` | Full move history |
| `GET` | `/games/{id}/stats` | Reward metrics & result |

---

## RL Architecture

| Component | Detail |
|-----------|--------|
| **Algorithm** | Deep Q-Learning (DQN) |
| **State** | 24-dimensional vector (1 = current player, −1 = opponent, 0 = empty) |
| **Action space** | 1 200 discrete actions covering all phases |
| **Action masking** | Strict legal-action mask applied at inference and during training |
| **Stability** | Target network synced every 1 000 steps |
| **Termination** | Episode ends on win/loss or after 500 legal moves (draw) |

### Board Layout (position indices)

```
00---------01---------02
|          |          |
|  08------09------10  |
|  |       |       |  |
|  |  16---17---18  |  |
|  |  |         |  |  |
07-15-23         19-11-03
|  |  |         |  |  |
|  |  22---21---20  |  |
|  |       |       |  |
|  14------13------12  |
|          |          |
06---------05---------04
```

---

## Tech Stack

- **Language**: Python 3.10+
- **API Framework**: FastAPI + Uvicorn
- **Validation**: Pydantic
- **Persistence**: SQLAlchemy + SQLite
- **ML**: PyTorch
- **Testing**: pytest + unittest

---

## Roadmap

- [x] Headless, deterministic game engine
- [x] Unified 1 200-action flat action space with legal masking
- [x] FastAPI backend with SQLite persistence
- [x] DQN agent skeleton + replay buffer
- [x] Random-agent baseline
- [ ] Full DQN training loop (self-play)
- [ ] Web UI for human-vs-AI play
- [ ] ELO tournament mode

---

## License

MIT
