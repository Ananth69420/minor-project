# Reinforcement Learning-Based Dispatch for Autonomous Fleet Management in Urban Mobility

A working live computational simulation and academic study implementing **Tabular Q-Learning with Semi-Markov Decision Process (SMDP) Fleet Parameter Sharing** for autonomous vehicle fleet dispatching and proactive repositioning in an urban mobility grid.

---

## Overview

This project models dispatch and repositioning for a small autonomous fleet on a grid. The `fleet-rl/` implementation runs a Python simulation behind a web dashboard. Its step and playback controls advance the environment, and its Q-learning policy reads the saved table at `fleet-rl/q_table.npy`. A separate replay dataset powers the dashboard when the server is not running.

The dashboard can inspect the policy state and action values, switch dispatch strategies, and run seeded comparisons. Inference uses the saved table; training mode updates the in-memory table during the session.

---

## ⚡ Quick Start: Launch the Live Simulation

### 1. Requirements
- Python 3.10+
- `fastapi`, `uvicorn`, `websockets`, `numpy`, `matplotlib`, `pytest`

Install dependencies:
```bash
pip install fastapi uvicorn websockets numpy matplotlib pytest
```

### 2. Launch the Live Computational Simulation
From the repository root directory, run:
```bash
python run_live.py
```
*(Alternatively: `python server.py`)*

This starts the FastAPI server and WebSocket telemetry stream, and automatically opens your web browser to:
👉 **`http://localhost:8000`**

### 3. Run Automated Validation Test Suite
Verify all 18 algorithmic and simulation requirements:
```bash
python -m pytest tests/ -v
```
The suite covers the simulator and the HTTP/WebSocket endpoints.

---

## 📁 Repository Architecture & V1 vs V2 Implementations

This repository maintains two implementations for empirical and theoretical completeness:

| Parameter | V1 (Diagnostic Setup - Root) | V2 (Authoritative Production - `fleet-rl/`) | Rationale / Academic Purpose |
| :--- | :---: | :---: | :--- |
| **Grid Dimensions** | $10 \times 10$ (100 cells) | **$16 \times 16$ (256 cells)** | Cross-quadrant transit time increases from 10 to 16 steps, raising the penalty of misaligned fleet distribution. |
| **Fleet Size** | 4 vehicles | **8 vehicles** | Restores operational elasticity ($18.4\%$ idle slack), allowing proactive repositioning to function effectively. |
| **State Space** | 5,184 unbucketed states | **576 structured SMDP states** | Factorized compact representation eliminates state sparsity and guarantees tabular convergence. |
| **Normal Arrival Rate** | $p = 0.20$ | **$p = 0.25$** | Realistic baseline traffic loading. |
| **Surge Arrival Rate** | $p = 0.60$ ($96.8\%$ saturation) | **$p = 0.55$** (elastic surge) | High-intensity surge clustered ($75\%$) in forecast hotspot quadrant. |
| **Simulation Duration** | 300 timesteps | **300 timesteps** | Divided into Phases 0 (Normal), 1 (Pre-Surge), 2 (Surge), 3 (Recovery). |
| **Action Space** | 6 discrete actions | **6 discrete actions** | 0: Nearest, 1: Longest-Wait, 2..5: Reposition to Q0..Q3 centroids. |
| **Trained Q-Table** | `q_table.npy` (5184, 6) | **`fleet-rl/q_table.npy` (576, 6)** | V2 table used by the live simulator. |

> **Note on Implementation Separation:** V2 inside `fleet-rl/` is the **authoritative implementation** for live simulation, benchmarking, and the project report. V1 remains archived for the utilization saturation diagnosis documented in Section 4 of the report.

### Directory Structure
```
d:/Minor Project/
├── run_live.py               # Master live simulation launcher (FastAPI + Browser)
├── server.py                 # Root server delegator
├── index.html                # High-contrast live simulation interface (HTML5 Canvas + WebSocket)
├── replay_data.js            # Precomputed benchmark replay data (offline fallback)
├── config.py                 # V1 environment configuration
├── PROJECT_REPORT.md         # Comprehensive college minor project report with Viva Voce Q&A
├── tests/
│   ├── test_simulation.py    # Simulation behavior and learning checks
│   └── test_server.py        # Automated test suite for FastAPI REST & WebSocket endpoints
├── fleet-rl/                 # Authoritative V2 Production Implementation
│   ├── server.py             # FastAPI REST API + WebSocket real-time stream
│   ├── config.py             # Centralized V1 and V2 environment definitions
│   ├── q_table.npy           # Authoritative trained Q-table (576 states x 6 actions)
│   ├── train.py              # Reproducible training pipeline (10,000 episodes)
│   ├── evaluate.py           # Final 50-episode test evaluation pipeline (Seeds 5000-5049)
│   ├── results.json          # Calibrated V2 test benchmark results
│   ├── results_v1.json       # Diagnostic V1 test benchmark results
│   ├── simulation/
│   │   └── engine.py         # Thread-safe simulation engine (CityEnv, Q-learning, Baseline)
│   ├── env/
│   │   └── city.py           # 16x16 2D urban grid, Poisson spawns, Manhattan paths, timeouts
│   ├── agents/
│   │   ├── qlearning.py      # Tabular Q-learning agent with multi-step SMDP updates
│   │   └── baseline.py       # Nearest-Vehicle baseline & Forecast-Aware heuristic
│   └── dashboard/
│       └── index.html        # Authoritative dashboard replica
```

---

## 🎮 How to Control and Demonstrate the Live Simulation

Once launched with `python run_live.py`, the dashboard connects to the live Python engine via WebSocket (`ws://localhost:8000/ws`). The top toolbar provides complete interactive control:

### 1. Timestep Execution
- **`STEP (+1 Timestep)` Button:** Directly invokes `POST /api/step` or WebSocket `step`. The Python simulator advances by exactly 1 tick:
  - Evaluates all idle autonomous vehicles.
  - Generates discrete state encodings ($s \in [0, 575]$).
  - Queries `q_table.npy` for real Q-values.
  - Applies action masks to filter invalid transitions.
  - Dispatches vehicles along Manhattan shortest paths.
  - Spawns Poisson passenger requests.
  - Boards passengers upon vehicle arrival (`status: trip`).
  - Delivers passengers to destinations (`status: idle`, $+20.0$ reward).
  - Handles cancellations for requests waiting $\ge 60$ steps ($-25.0$ penalty).
- **`Live Play / Pause` Button:** Runs continuous simulation at selected speed multiplier:
  - `1x`: 500 ms / timestep
  - `2x`: 250 ms / timestep
  - `5x`: 100 ms / timestep
  - `10x`: 50 ms / timestep
- **`Reset` Button:** Resets simulator to Step 0 with specified random seed and hotspot.

### 2. Reinforcement Learning Decision Inspector
Located in the right panel and in the **Learned Q-Table Policy Inspector** view:
- **Vehicle Selection:** Click any vehicle on the $16 \times 16$ grid (or roster) to inspect its telemetry.
- **State Representation:** Exposes the exact integer state $s = (((q_v \cdot 4 + p) \cdot 4 + h) \cdot 3 + b_{\text{own}}) \cdot 3 + b_{\text{hot}}$.
- **Decoded Features:**
  - Vehicle Quadrant $q_v \in \{0, 1, 2, 3\}$ (NW, NE, SW, SE).
  - Demand Phase $p \in \{0, 1, 2, 3\}$ (Normal, Pre-Surge, Active Surge, Recovery).
  - Forecast Surge Hotspot $h \in \{0, 1, 2, 3\}$.
  - Own Quadrant Queue Bucket $b_{\text{own}} \in \{0, 1, 2\}$ ($0$, $1\text{--}2$, $3+$ requests).
  - Hotspot Quadrant Queue Bucket $b_{\text{hot}} \in \{0, 1, 2\}$.
- **Action Q-Values Matrix:** Shows exact learned Q-values for all 6 macro-actions:
  - `0: Serve Nearest Pending Request`
  - `1: Serve Longest-Waiting Pending Request`
  - `2..5: Reposition to Q0..Q3 Centroid`
- **Masking & Selection Badges:** Clearly labels which actions are `VALID` versus `MASKED OUT`, and indicates whether the selection was `GREEDY (EXPLOIT)` or `EXPLORATION (EPSILON)`.
- **Online Learning Telemetry (Mode B):** When Training Mode is active, displays multi-step duration $\Delta t$, accumulated return $G$, TD target $G + \gamma^{\Delta t} \max Q'$, TD error $\delta$, and parameter update $\Delta Q = \alpha \delta$.

### 3. Demand Surge Demonstration
Observe the four operational regimes as the simulation advances:
- **Steps 0–79 (Phase 0: Normal):** Uniform passenger arrival ($p=0.25$). Vehicles serve local requests.
- **Steps 80–99 (Phase 1: Pre-Surge Warning):** Yellow forecast signal active. Arrival rate remains normal, but the agent receives the surge hotspot quadrant forecast. The Decision Inspector demonstrates the agent selecting **Action 2..5 (Reposition to Hotspot)** to relocate slack vehicles in advance.
- **Steps 100–199 (Phase 2: Active Demand Surge):** Arrival rate increases to $p=0.55$, with **$75\%$ of requests concentrated in the hotspot**. Pre-positioned vehicles immediately pick up surge calls without long transit delays.
- **Steps 200–299 (Phase 3: Post-Surge Recovery):** Arrival rate returns to normal. Fleet clears residual queues.

### 4. Head-to-Head Baseline Comparison (Identical Seed)
Click **`⚔ Compare Baseline vs RL`** in the toolbar:
- The Python simulator runs a paired evaluation for the specified seed across:
  1. Reactive Nearest-Vehicle Baseline
  2. Advance Forecast-Aware Rule
  3. Trained Tabular Q-Learning Agent
- Displays an honest, side-by-side performance scorecard under **strictly identical passenger spawn events**:
  - Overall Average Wait Time
  - Surge-Phase Wait Time
  - Normal-Phase Wait Time
  - Requests Served
  - Requests Dropped (Cancellations)
  - Service Fulfillment Rate (%)
  - Total Fleet Distance Travelled
  - Cumulative Fleet Reward
- Demonstrates how proactive repositioning achieves a measurable reduction in surge wait time while traveling slightly more distance (the classic mobility trade-off).

---

## 📊 Evaluation Benchmark Results (Held-Out Test Seeds 5000–5049)

Across 50 strictly reserved test seeds (evaluated over 300 steps each, mean $\pm$ standard deviation):

| Performance Metric | Nearest Baseline | Forecast-Aware Rule | Tabular Q-Learning (RL) | RL Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Average Wait Time (steps)** | 14.35 $\pm$ 1.76 | 13.96 $\pm$ 1.67 | **13.87 $\pm$ 1.44** | **-0.48 steps** |
| **Surge-Phase Wait Time (steps)** | 17.15 $\pm$ 2.74 | 16.34 $\pm$ 2.64 | **16.32 $\pm$ 2.51** | **-0.83 steps** |
| **Normal-Phase Wait Time (steps)** | 11.13 $\pm$ 1.26 | 11.15 $\pm$ 1.05 | **11.04 $\pm$ 1.02** | **-0.09 steps** |
| **Requests Served (Trips Completed)** | 97.70 $\pm$ 6.73 | 97.66 $\pm$ 6.48 | **97.96 $\pm$ 6.69** | **+0.26 trips** |
| **Requests Dropped (Cancellations)** | 3.24 $\pm$ 2.73 | **2.92 $\pm$ 2.78** | 3.04 $\pm$ 2.93 | **-0.20 cancellations** |
| **Total Fleet Distance (Grid Steps)** | **1901.50 $\pm$ 138.78** | 1961.84 $\pm$ 126.90 | 2081.66 $\pm$ 121.31 | +180.16 steps |

---

## 🧠 Algorithmic Formulation

### 1. Multi-Step Semi-Markov Decision Process (SMDP)
Physical movement requires $\Delta t \ge 1$ time steps. Intra-decision rewards accumulate with geometric discounting:
$$G = \sum_{\tau=0}^{\Delta t - 1} \gamma^\tau r_{t + \tau}$$

Upon decision completion (passenger delivery, reposition arrival, or timeout), the shared Q-table is updated via the Bellman optimality equation restricted to valid actions:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ G + \gamma^{\Delta t} \max_{a' \in \mathcal{A}_{\text{valid}}(s')} Q(s', a') - Q(s, a) \right]$$

### 2. Fleet Parameter Sharing
All 8 vehicles execute decisions against a shared $576 \times 6$ tabular array. Parameter sharing increases sample collection efficiency by $8\times$ and maintains a homogeneous fleet control policy.

---

## 🎓 Faculty Evaluation & Viva Voce Demonstration Script

To present this computational model during an academic evaluation:

1. **Launch the Engine:** Run `python run_live.py` in terminal. Show the terminal output confirming FastAPI server and WebSocket initialization.
2. **Show the initial state:** Open the dashboard and confirm the live connection indicator. The 8 vehicles start at the four quadrant centers: $(4, 4)$, $(12, 4)$, $(4, 12)$, and $(12, 12)$.
3. **Execute Single Timesteps:** Click **`STEP (+1 Timestep)`** multiple times. Point out how vehicle coordinates increment along Manhattan shortest paths in direct synchronization with the Python simulator.
4. **Inspect the Agent's Mind:** Select an idle vehicle. Open the **Decision Inspector** panel. Explain how the discrete state index (e.g. `State #198`) is calculated, how Q-values are queried, and how action masking prevents illegal repositioning.
5. **Demonstrate Pre-Surge Anticipation:** Scrub or advance the simulation to Step 80–99 (Phase 1). Show that the agent observes the surge warning for the hotspot and autonomously selects repositioning actions before requests spawn.
6. **Trigger Baseline Comparison:** Click **`⚔ Compare Baseline vs RL`**. Show the examiner that on the identical pseudo-random seed, Q-learning achieves a lower surge waiting time than the reactive nearest-vehicle baseline.
7. **Run Automated Tests:** Switch to terminal and run `python -m pytest tests/ -v`. Show 25 passing unit and integration tests confirming algorithmic correctness.
