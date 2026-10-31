"""
Configuration parameters for Reinforcement Learning Based Dispatch
for Autonomous Fleet Management in Urban Mobility.
All parameters, hyper-parameters, and seeds are centralized here.
"""

# ==========================================
# City Grid & Fleet Parameters
# ==========================================
GRID_SIZE = 10                  # 10x10 city grid
NUM_VEHICLES = 4                # Default autonomous fleet size
EPISODE_STEPS = 300             # Time steps per episode (1 step = 1 grid move)
MAX_WAIT_TIME = 60              # Steps before an unserved request drops out

# 4 Quadrants on a 10x10 grid:
# Quadrant 0: Top-Left (NW)  x in [0, 4], y in [0, 4] -> Center (2, 2)
# Quadrant 1: Top-Right (NE) x in [5, 9], y in [0, 4] -> Center (7, 2)
# Quadrant 2: Bottom-Left (SW) x in [0, 4], y in [5, 9] -> Center (2, 7)
# Quadrant 3: Bottom-Right (SE) x in [5, 9], y in [5, 9] -> Center (7, 7)
QUADRANT_CENTERS = [
    (2, 2),  # NW (0)
    (7, 2),  # NE (1)
    (2, 7),  # SW (2)
    (7, 7),  # SE (3)
]

# ==========================================
# Demand & Surge Schedule
# ==========================================
# Episode Timeline:
# Phase 1: Steps 0..99    -> Normal demand
# Phase 2: Steps 100..199 -> Surge demand (3x rate, concentrated in hotspot quadrant)
# Phase 3: Steps 200..299 -> Normal demand
NORMAL_DEMAND_PROB = 0.20       # Probability of request spawn per step in normal phase
SURGE_DEMAND_PROB = 0.60        # 3x rate in surge phase
SURGE_START_STEP = 100
PRE_SURGE_START_STEP = 80
SURGE_END_STEP = 200
SURGE_HOTSPOT_CONCENTRATION = 0.75  # 75% of surge requests originate in the surge quadrant

# ==========================================
# Reward Function
# ==========================================
REWARD_DROPOFF = 20.0           # Positive reward per successful passenger dropoff
PENALTY_WAIT_STEP = -0.10       # Negative per step for every pending request
PENALTY_DISTANCE = -0.05        # Fuel / wear penalty per unit distance travelled
PENALTY_DROPPED = -20.0         # Severe penalty for passenger cancellation / dropped request

# ==========================================
# Q-Learning Hyperparameters
# ==========================================
# State space: 4 phases * 4 hotspots * 4 vehicle quadrants * 81 demand bucket combinations = 5,184 states
# Actions: 0 = Nearest, 1 = Longest Waiting, 2..5 = Reposition to Quadrant 0..3 center
NUM_STATES = 5184
NUM_ACTIONS = 6

LEARNING_RATE = 0.10            # Alpha
DISCOUNT_FACTOR = 0.95          # Gamma
EPSILON_START = 1.00            # Initial exploration rate
EPSILON_MIN = 0.05              # Minimum exploration rate
EPSILON_DECAY = 0.9995          # Decay rate per training episode

# Training & Evaluation Settings
NUM_TRAIN_EPISODES = 10000       # Sufficient for tabular convergence (< 15 seconds)
NUM_EVAL_EPISODES = 50          # Statistical sample size for fair comparison
RANDOM_SEED = 42                # Base seed for reproducibility
