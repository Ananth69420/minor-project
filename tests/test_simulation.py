"""Checks the fleet environment, dispatcher, and learning agent.

Coverage includes:
1. City initialization.
2. Vehicle initialization.
3. Passenger generation.
4. Vehicle movement.
5. Passenger pickup.
6. Passenger drop-off.
7. Passenger cancellation.
8. Demand surge behavior.
9. Q-table loading.
10. Q-value retrieval.
11. Action selection.
12. Reward calculation.
13. Q-table updates.
14. Baseline dispatching.
15. Random seed reproducibility.
16. Single-timestep execution.
17. Reset functionality.
18. Training versus evaluation mode.
"""

import os
import sys
import copy
import pytest
import numpy as np

# Ensure fleet-rl root is on sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)
FLEET_RL_DIR = os.path.join(ROOT_DIR, "fleet-rl")
if FLEET_RL_DIR not in sys.path:
    sys.path.insert(0, FLEET_RL_DIR)

from env.city import CityEnv
from agents.qlearning import QLearningAgent
from agents.baseline import BaselineDispatcher, ForecastRuleDispatcher
from simulation.engine import SimulationEngine, ACTION_DEFINITIONS


# 1. City initialization
def test_01_city_initialization():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42, p_normal=0.25, p_surge=0.55)
    assert env.grid_size == 16
    assert env.num_vehicles == 8
    assert env.max_steps == 300
    assert env.max_wait == 60
    assert len(env.quad_centers) == 4
    assert env.quad_centers == [(4, 4), (12, 4), (4, 12), (12, 12)]
    assert env.step_num == 0


# 2. Vehicle initialization
def test_02_vehicle_initialization():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42)
    env.reset(seed=42, surge_quad=0)
    assert len(env.vehicles) == 8

    # Verify quadrant distributions and symmetric placements
    for i, v in enumerate(env.vehicles):
        assert v["id"] == i
        assert v["status"] == "idle"
        assert v["target_x"] is None
        assert v["target_y"] is None
        assert v["req_id"] is None
        assert v["total_dist"] == 0
        assert v["idle_dist"] == 0
        assert v["trips"] == 0
        expected_pos = env.quad_centers[i % 4]
        assert (v["x"], v["y"]) == expected_pos


# 3. Passenger generation
def test_03_passenger_generation():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=100, p_normal=1.0)
    env.reset(seed=100, surge_quad=1)

    # With p_normal=1.0, step must spawn a passenger
    reward, done, info = env.step()
    assert len(env.requests) >= 1
    req = env.requests[0]

    assert "id" in req
    assert "px" in req and "py" in req
    assert "dx" in req and "dy" in req
    assert "spawn_step" in req
    assert "wait" in req
    assert "status" in req
    assert req["status"] == "pending"
    assert 0 <= req["px"] < 16 and 0 <= req["py"] < 16
    assert 0 <= req["dx"] < 16 and 0 <= req["dy"] < 16
    assert not (req["px"] == req["dx"] and req["py"] == req["dy"])


# 4. Vehicle movement
def test_04_vehicle_movement():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42)
    env.reset(seed=42)

    # Manually dispatch AV-0 to reposition to target (10, 10)
    v0 = env.vehicles[0]
    initial_x, initial_y = v0["x"], v0["y"]
    v0["status"] = "repo"
    v0["target_x"] = initial_x + 3
    v0["target_y"] = initial_y + 3

    # Execute 1 step
    env.step()
    # Manhattan move: exactly 1 unit travelled
    dist_travelled = abs(v0["x"] - initial_x) + abs(v0["y"] - initial_y)
    assert dist_travelled == 1
    assert v0["total_dist"] == 1
    assert v0["idle_dist"] == 1


# 5. Passenger pickup
def test_05_passenger_pickup():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42)
    env.reset(seed=42)

    # Place a passenger at distance 1 from vehicle 0
    v0 = env.vehicles[0]
    px, py = v0["x"] + 1, v0["y"]
    dx, dy = v0["x"] + 5, v0["y"] + 5

    req = {
        "id": 999,
        "px": px,
        "py": py,
        "dx": dx,
        "dy": dy,
        "spawn_step": 0,
        "wait": 0,
        "status": "pending",
        "assigned_vehicle": None
    }
    env.requests.append(req)

    # Dispatch vehicle 0 to serve nearest
    dispatched = env.dispatch(0, 0)
    assert dispatched is True
    assert v0["status"] == "pickup"
    assert v0["req_id"] == 999
    assert req["status"] == "assigned"
    assert req["assigned_vehicle"] == 0

    # Step: vehicle moves 1 step and arrives at pickup
    env.step()
    assert v0["x"] == px and v0["y"] == py
    assert req["status"] == "in_transit"
    assert v0["status"] == "trip"
    assert v0["target_x"] == dx and v0["target_y"] == dy


# 6. Passenger drop-off
def test_06_passenger_dropoff():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42)
    env.reset(seed=42)

    v0 = env.vehicles[0]
    # Simulate vehicle already in trip at distance 1 from destination
    req = {
        "id": 888,
        "px": v0["x"],
        "py": v0["y"],
        "dx": v0["x"] + 1,
        "dy": v0["y"],
        "spawn_step": 0,
        "wait": 5,
        "status": "in_transit",
        "assigned_vehicle": 0
    }
    env.requests.append(req)
    v0["status"] = "trip"
    v0["target_x"] = req["dx"]
    v0["target_y"] = req["dy"]
    v0["req_id"] = 888

    # Step: arrives at dropoff
    reward, done, info = env.step()
    assert req["status"] == "completed"
    assert v0["status"] == "idle"
    assert v0["target_x"] is None and v0["target_y"] is None
    assert v0["req_id"] is None
    assert v0["trips"] == 1
    assert info["dropoffs"] == 1
    assert info["dropoffs_by_v"][0] is True


# 7. Passenger cancellation
def test_07_passenger_cancellation():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42)
    env.reset(seed=42)

    # Passenger whose wait time reaches 59 steps
    req = {
        "id": 777,
        "px": 10,
        "py": 10,
        "dx": 2,
        "dy": 2,
        "spawn_step": 0,
        "wait": 59,
        "status": "pending",
        "assigned_vehicle": None
    }
    env.requests.append(req)

    # Step: wait increments to 60 -> should drop
    reward, done, info = env.step()
    assert req["wait"] == 60
    assert req["status"] == "dropped"
    assert info["drops"] == 1


# 8. Demand surge behavior
def test_08_demand_surge_behavior():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42, p_normal=0.0, p_surge=1.0)
    env.reset(seed=42, surge_quad=2)  # Q2 (SW): x in [0, 7], y in [8, 15]

    # Move environment to surge phase (steps 100 to 199)
    env.step_num = 100
    assert env.get_phase() == 2

    # Run 10 steps during surge with p_surge=1.0
    surge_spawns = 0
    hotspot_spawns = 0
    for _ in range(10):
        prev_len = len(env.requests)
        env.step()
        if len(env.requests) > prev_len:
            surge_spawns += 1
            new_req = env.requests[-1]
            q = env.get_quadrant(new_req["px"], new_req["py"])
            if q == 2:
                hotspot_spawns += 1

    assert surge_spawns == 10
    # Over 10 spawns, ~75% originate in hotspot quadrant 2
    assert hotspot_spawns >= 5


# 9. Q-table loading
def test_09_qtable_loading():
    agent = QLearningAgent(num_states=576, num_actions=6)
    q_table_path = os.path.join(FLEET_RL_DIR, "q_table.npy")
    assert os.path.exists(q_table_path)
    agent.load(q_table_path)

    assert agent.q_table.shape == (576, 6)
    assert not np.isnan(agent.q_table).any()
    assert np.count_nonzero(agent.q_table) > 500


# 10. Q-value retrieval
def test_10_qvalue_retrieval():
    engine = SimulationEngine()
    # Test valid state ID 198: Q1, Phase 1 (Pre-Surge), Hotspot Q2, own queue 0, hotspot queue 0
    # (((1*4 + 1)*4 + 2)*3 + 0)*3 + 0 = 198
    q_data = engine.get_q_values(198, valid_actions=[2, 3, 4, 5])
    assert q_data["state"]["state_id"] == 198
    assert q_data["state"]["vehicle_quad"] == 1
    assert q_data["state"]["phase"] == 1
    assert q_data["state"]["hotspot_quad"] == 2
    assert q_data["state"]["own_bucket"] == 0
    assert q_data["state"]["hotspot_bucket"] == 0
    assert len(q_data["actions"]) == 6

    # Action 4 is Reposition to Q2 (the surge hotspot)
    act4 = q_data["actions"][4]
    assert act4["id"] == 4
    assert act4["is_valid"] is True
    assert isinstance(act4["q_value"], float)


# 11. Action selection
def test_11_action_selection():
    agent = QLearningAgent(num_states=576, num_actions=6)
    agent.q_table = np.zeros((576, 6))
    # Give action 4 the highest Q-value
    agent.q_table[100, 4] = 15.5
    agent.q_table[100, 2] = 5.0
    agent.q_table[100, 3] = 8.0

    # With explore=False, must select action 4 greedily
    selected = agent.select_action(state=100, valid_actions=[2, 3, 4], explore=False)
    assert selected == 4

    # If action 4 is masked out, must select action 3
    selected_masked = agent.select_action(state=100, valid_actions=[2, 3], explore=False)
    assert selected_masked == 3


def test_greedy_ties_are_repeatable():
    agent = QLearningAgent(num_states=576, num_actions=6)
    agent.q_table[100, [2, 4]] = 3.0

    choices = [agent.select_action(100, [2, 4], explore=False) for _ in range(10)]

    assert choices == [2] * 10


def test_policy_changes_discard_in_flight_training_updates():
    engine = SimulationEngine()
    engine.reset(mode="training")
    engine.agent.start_decision(v_id=0, state=10, action=2)

    engine.set_mode("inference")
    assert engine.agent.active_decisions == {}

    engine.set_mode("training")
    engine.agent.start_decision(v_id=0, state=10, action=2)
    engine.set_dispatcher("baseline")
    assert engine.agent.active_decisions == {}


def test_engine_rejects_unknown_policy_and_mode():
    engine = SimulationEngine()

    with pytest.raises(ValueError):
        engine.set_dispatcher("unknown")
    with pytest.raises(ValueError):
        engine.set_mode("unknown")
    with pytest.raises(ValueError):
        engine.reset(mode="unknown")


# 12. Reward calculation
def test_12_reward_calculation():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42)
    env.reset(seed=42)

    # Add 2 pending requests and trigger 1 dropoff
    env.requests.append({"id": 1, "px": 0, "py": 0, "dx": 1, "dy": 1, "spawn_step": 0, "wait": 0, "status": "pending", "assigned_vehicle": None})
    env.requests.append({"id": 2, "px": 5, "py": 5, "dx": 6, "dy": 6, "spawn_step": 0, "wait": 0, "status": "pending", "assigned_vehicle": None})

    reward, done, info = env.step()
    # Expected reward: dropoffs * 20.0 - pending * 0.15 - dist * 0.05 - drops * 25.0
    expected = (
        info["dropoffs"] * 20.0
        - info["pending"] * 0.15
        - info["distance"] * 0.05
        - info["drops"] * 25.0
    )
    assert abs(reward - expected) < 1e-6


# 13. Q-table updates (Semi-Markov Bellman update)
def test_13_qtable_updates():
    agent = QLearningAgent(num_states=576, num_actions=6, lr=0.1, gamma=0.95)
    agent.q_table = np.zeros((576, 6))

    # Vehicle 0 starts decision at state 50 with action 2
    agent.start_decision(v_id=0, state=50, action=2)
    # Step 1: reward -0.5
    agent.add_step_reward(v_id=0, r=-0.5)
    # Step 2: reward +20.0
    agent.add_step_reward(v_id=0, r=20.0)

    # Next state 60 has max Q-value of 10.0
    agent.q_table[60, 0] = 10.0

    # Finish decision
    tr = agent.finish_decision(v_id=0, next_state=60, next_valid_actions=[0, 1], done=False)
    assert tr is not None
    assert tr["duration"] == 2
    # Discounted reward: -0.5 + (0.95^1)*20.0 = 18.5
    assert abs(tr["discounted_reward"] - 18.5) < 1e-4

    # Target: G + (gamma^dt) * max Q(s', a') = 18.5 + (0.95^2) * 10.0 = 18.5 + 9.025 = 27.525
    expected_target = 18.5 + (0.95 ** 2) * 10.0
    assert abs(tr["target"] - expected_target) < 1e-4

    # Q update: Q += lr * (target - Q) = 0.1 * 27.525 = 2.7525
    expected_q = 0.1 * expected_target
    assert abs(agent.q_table[50, 2] - expected_q) < 1e-4


# 14. Baseline dispatching
def test_14_baseline_dispatching():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42)
    env.reset(seed=42)
    baseline = BaselineDispatcher()

    # With no requests, baseline selects None
    assert baseline.select_action(env, 0) is None

    # Add pending request
    env.requests.append({"id": 10, "px": 6, "py": 6, "dx": 1, "dy": 1, "spawn_step": 0, "wait": 0, "status": "pending", "assigned_vehicle": None})
    action = baseline.select_action(env, 0)
    assert action == 0  # Action 0 = nearest pending request


# 15. Random seed reproducibility
def test_15_seed_reproducibility():
    def run_seeded_env(seed):
        env = CityEnv(grid_size=16, num_vehicles=8, max_steps=50, max_wait=60, seed=seed)
        env.reset(seed=seed, surge_quad=0)
        baseline = BaselineDispatcher()
        for _ in range(50):
            for v in env.vehicles:
                if v["status"] == "idle":
                    a = baseline.select_action(env, v["id"])
                    if a is not None:
                        env.dispatch(v["id"], a)
            env.step()
        return env.get_metrics(), [(v["x"], v["y"]) for v in env.vehicles], [(r["id"], r["px"], r["py"]) for r in env.requests]

    m1, v_pos1, r_data1 = run_seeded_env(5000)
    m2, v_pos2, r_data2 = run_seeded_env(5000)
    m3, v_pos3, r_data3 = run_seeded_env(5001)

    assert m1 == m2
    assert v_pos1 == v_pos2
    assert r_data1 == r_data2

    # Different seeds must produce different requests
    assert r_data1 != r_data3


# 16. Single-timestep execution
def test_16_single_timestep_execution():
    engine = SimulationEngine()
    engine.reset(seed=5000)
    assert engine.step_num == 0

    st1 = engine.step()
    assert st1["step"] == 1
    assert engine.step_num == 1

    st2 = engine.step()
    assert st2["step"] == 2
    assert engine.step_num == 2


# 17. Reset functionality
def test_17_reset_functionality():
    engine = SimulationEngine()
    engine.reset(seed=5000, surge_quad=2)
    for _ in range(20):
        engine.step()
    assert engine.step_num == 20
    assert engine.cumulative_reward != 0.0

    # Reset
    reset_state = engine.reset(seed=5000, surge_quad=2)
    assert reset_state["step"] == 0
    assert engine.step_num == 0
    assert engine.cumulative_reward == 0.0
    assert len(engine.step_rewards) == 0
    assert len(reset_state["vehicles"]) == 8
    for v in reset_state["vehicles"]:
        assert v["status"] == "idle"


# 18. Training versus evaluation mode
def test_18_training_vs_evaluation_mode():
    engine = SimulationEngine()

    # Mode A: Inference (Trained Model)
    engine.reset(seed=5000, mode="inference")
    initial_q_copy = engine.agent.q_table.copy()
    assert engine.mode == "inference"
    assert engine.agent.eps == 0.0

    for _ in range(30):
        engine.step()

    # Q-table MUST NOT be modified in inference mode
    np.testing.assert_array_equal(engine.agent.q_table, initial_q_copy)

    # Mode B: Training Mode
    engine.reset(seed=5000, mode="training")
    assert engine.mode == "training"
    assert engine.agent.eps > 0.0

    # Run steps in training mode
    for _ in range(50):
        engine.step()

    # In training mode, decisions were started and finished
    assert len(engine.recent_transitions) > 0 or len(engine.agent.active_decisions) > 0


# Edge-case: No passenger assigned simultaneously to multiple vehicles
def test_no_duplicate_passenger_assignment():
    env = CityEnv(grid_size=16, num_vehicles=8, max_steps=300, max_wait=60, seed=42)
    env.reset(seed=42)

    # Add 1 request
    env.requests.append({"id": 1, "px": 5, "py": 5, "dx": 10, "dy": 10, "spawn_step": 0, "wait": 0, "status": "pending", "assigned_vehicle": None})

    # Dispatch vehicle 0 to serve nearest
    env.dispatch(0, 0)
    # Attempt to dispatch vehicle 1 to serve nearest
    env.dispatch(1, 0)

    # Vehicle 0 should be assigned, vehicle 1 must NOT be assigned to request 1
    req = env.requests[0]
    assert req["assigned_vehicle"] == 0
    assert env.vehicles[1]["req_id"] is None
