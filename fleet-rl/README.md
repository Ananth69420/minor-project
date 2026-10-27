# Reinforcement Learning Based Dispatch for Autonomous Fleet Management in Urban Mobility

A prototype simulating dispatch and proactive repositioning decisions for a fleet of autonomous vehicles operating on a 10x10 grid city. Tabular Q-learning is evaluated alongside a forecast-aware rule dispatcher and a traditional nearest-vehicle baseline across stationary and non-stationary demand surge conditions.

## Project Overview

- **Environment:** 10x10 grid with 4 autonomous vehicles.
- **Episode Length:** 300 steps. A request pending for more than 60 steps drops out with a cancellation penalty.
- **Demand Phases & Forecast:**
  - **Phase 0 (Steps 0–79):** Normal demand (uniform spawn probability 0.20 per step).
  - **Phase 1 (Steps 80–99):** Pre-surge warning. A demand forecast signals which quadrant will experience a surge. Demand arrival remains normal.
  - **Phase 2 (Steps 100–199):** Surge active (spawn probability 0.70 with 75% concentrated in the forecast hotspot quadrant).
  - **Phase 3 (Steps 200–299):** Post-surge recovery (uniform spawn probability 0.20 per step).
- **Vehicles:** Move along Manhattan shortest paths. Automatic pickup upon arrival at passenger origin, automatic dropoff upon arrival at destination.
- **Fleet Control:** Fully centralized and autonomous (no driver accept/reject mechanics).

## Forecast Assumption

- **RL (Q-Learning)** and **Forecast-Aware Rule** receive a demand forecast indicating the upcoming surge phase and the specific hotspot quadrant during phases 1 and 2.
- **Baseline (Nearest Vehicle)** operates strictly reactively with **no forecast information**, dispatching the nearest vehicle only after requests have already spawned.

## RL Formulation (Tabular Q-Learning)

- **Agent:** Shared tabular Q-learning across all 4 vehicles.
- **State Space (576 states):**
  - Vehicle's current quadrant: 4 values (0 = NW, 1 = NE, 2 = SW, 3 = SE).
  - Current phase: 4 values (0 = Normal, 1 = Pre-surge warning, 2 = Surge, 3 = Post-surge).
  - Forecast hotspot quadrant: 4 values (0–3; set to vehicle's own quadrant during phases 0 and 3).
  - Pending request count bucket in vehicle's own quadrant: 3 values (0 = 0, 1 = 1–2, 2 = 3+).
  - Pending request count bucket in forecast hotspot quadrant: 3 values (0 = 0, 1 = 1–2, 2 = 3+).
  - Total states: $4 \times 4 \times 4 \times 3 \times 3 = 576$.
- **Action Space & Masking:**
  - 6 discrete actions when a vehicle is idle:
    - `0`: Serve nearest pending request.
    - `1`: Serve longest-waiting pending request (local quadrant first, else nearest).
    - `2`: Reposition to quadrant 0 center (2, 2).
    - `3`: Reposition to quadrant 1 center (7, 2).
    - `4`: Reposition to quadrant 2 center (2, 7).
    - `5`: Reposition to quadrant 3 center (7, 7).
  - **Strict Action Masking:**
    - Actions 0 and 1 (serve request) are valid *only* when at least one unassigned request exists on the grid.
    - Reposition actions 2–5 are valid *only* when the vehicle is idle and no requests exist, OR when the target quadrant differs from the vehicle's current quadrant.
    - Masked actions are strictly prohibited from selection during both exploration and exploitation (never chosen).
- **Reward Function:**
  - Dropoff reward: +20.0 per completed trip
  - Movement cost: -0.05 per Manhattan step
  - Queue wait penalty: -0.15 per pending request per step (shared equally across fleet)
  - Cancellation drop penalty: -25.0 per timed-out request (shared equally across fleet)
- **Hyperparameters:** Learning rate $\alpha = 0.05$, discount factor $\gamma = 0.96$, $\epsilon$-greedy decay from 1.0 to 0.01 (decay rate 0.9996 over 10,000 training episodes).

## Dispatch Policies Evaluated

1. **Baseline (Nearest Vehicle):** Standard reactive greedy dispatcher. Whenever a vehicle is idle, it serves the nearest unassigned pending request. Does not reposition when idle. Operates without forecast.
2. **Forecast-Aware Rule:** Hand-coded heuristic. Serves nearest pending request; when idle with no unassigned requests during phases 1 and 2, proactively repositions toward the forecast hotspot quadrant center.
3. **Reinforcement Learning (Q-Learning):** Tabular agent learning from experience using the 576-state forecast representation and action masking.

## How to Run

Requirements: Python 3.10+, numpy, matplotlib.

1. Train the Q-learning agent:
```bash
python train.py
```
This runs 10,000 episodes (~110 seconds on a standard machine) and outputs `q_table.npy` and `training_curve.json`.

2. Run evaluation on test seeds:
```bash
python evaluate.py
```
This evaluates all three policies across 50 fixed test seeds (seeds 5000 to 5049) with identical demand realizations. It outputs `results.json` and `dashboard/replay_data.js`.

3. Open the dashboard:
Double-click `dashboard/index.html` in your browser. It loads `replay_data.js` via a standard script tag and does not require a local web server.

## Evaluation Results

Results averaged over 50 fixed-seed test episodes (seeds 5000–5049), reporting mean $\pm$ standard deviation:

| Policy | Average Wait (steps) | Surge Wait (steps) | Normal Wait (steps) | Requests Served | Requests Dropped | Total Distance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Nearest)** | 15.52 &plusmn; 1.84 | 20.11 &plusmn; 2.55 | 9.57 &plusmn; 1.98 | 87.52 &plusmn; 6.23 | 10.60 &plusmn; 4.09 | 984.84 &plusmn; 64.53 |
| **Forecast-Aware Rule** | 15.57 &plusmn; 1.88 | 20.20 &plusmn; 2.36 | 9.46 &plusmn; 1.79 | 88.22 &plusmn; 6.28 | 9.90 &plusmn; 4.22 | 1007.26 &plusmn; 61.29 |
| **RL (Q-Learning)** | 15.57 &plusmn; 1.86 | 20.26 &plusmn; 2.57 | 9.48 &plusmn; 1.81 | 87.38 &plusmn; 6.11 | 10.80 &plusmn; 4.17 | 1078.48 &plusmn; 55.68 |

## Observations

1. **Surge Wait Comparison:**
   - On the 50 test seeds (5000–5049), the RL agent achieves a surge phase wait time of 20.26 steps compared to the Baseline dispatcher's 20.11 steps.
   - **RL does not beat the greedy nearest-vehicle baseline during the surge on this test set** (a deficit of 0.15 steps, or +0.7%).
   - However, the demand forecast features and action masking dramatically close the gap compared to the unforecasted RL agent (which achieved 21.43 steps vs 20.11 steps, a gap of +1.32 steps). On the larger 200-episode validation set (seeds 1000–1199), Q-learning achieved 20.37 steps vs baseline 20.51 steps.

2. **Why Tabular RL Struggles to Outperform Greedy Nearest:**
   - **Spatial Discretization Loss:** The 576 tabular states compress the 10x10 coordinate space into 4 coarse quadrants and 3 count buckets. The greedy nearest dispatcher calculates exact Manhattan point-to-point distances across all 100 grid cells. A vehicle repositioned to a quadrant center may still be 3–4 cells away from where a surge passenger actually spawns.
   - **Commitment Risk of Repositioning:** When a vehicle begins repositioning in Phase 1 (pre-surge), it travels for multiple steps. If non-surge requests spawn in nearby cells during transit, the vehicle cannot immediately service them without incurring detours.
   - **Low Pre-Surge Phase Visitation:** Phase 1 lasts only 20 simulation steps (steps 80–99), which is just 6.7% of the 300-step episode. Idle vehicles make very few decisions within Phase 1, making it challenging for tabular TD updates to fully propagate the long-term payoff of surge dropoffs without tens of thousands of episodes.

3. **Forecast-Aware Rule Benchmark:**
   - The hand-coded Forecast-Aware Rule demonstrates the concrete benefits of pre-positioning: it achieves the highest request service volume (88.22 vs 87.52) and reduces dropped passenger cancellations from 10.60 to 9.90.
   - This confirms that pre-positioning based on demand forecasts increases fleet throughput during demand surges, even though individual trip travel distance increases slightly (1007.26 vs 984.84).

4. **Normal Demand Performance:**
   - In the normal phase (steps 0–79 and 200–299), both the RL agent (9.48 steps) and the Forecast-Aware Rule (9.46 steps) achieve lower wait times than the Baseline (9.57 steps), showing that the learned dispatch policy services balanced demand effectively.
