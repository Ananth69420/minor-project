"""Simulation engine for the 16x16 fleet environment."""

import os
import sys
import copy
import time
import threading
import numpy as np

# Ensure fleet-rl root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FLEET_RL_DIR = os.path.dirname(CURRENT_DIR)
if FLEET_RL_DIR not in sys.path:
    sys.path.insert(0, FLEET_RL_DIR)

from env.city import CityEnv
from agents.qlearning import QLearningAgent
from agents.baseline import BaselineDispatcher, ForecastRuleDispatcher
from config import ENV_V2

ACTION_DEFINITIONS = [
    {"id": 0, "name": "Serve Nearest", "type": "DISPATCH", "target_quad": None, "desc": "Serve Nearest Pending Request"},
    {"id": 1, "name": "Serve Longest-Waiting", "type": "DISPATCH", "target_quad": None, "desc": "Serve Longest-Waiting Request in Local Quadrant"},
    {"id": 2, "name": "Reposition Q0 (NW)", "type": "REPOSITION", "target_quad": 0, "desc": "Reposition to Quadrant 0 (North-West) Centroid (4, 4)"},
    {"id": 3, "name": "Reposition Q1 (NE)", "type": "REPOSITION", "target_quad": 1, "desc": "Reposition to Quadrant 1 (North-East) Centroid (12, 4)"},
    {"id": 4, "name": "Reposition Q2 (SW)", "type": "REPOSITION", "target_quad": 2, "desc": "Reposition to Quadrant 2 (South-West) Centroid (4, 12)"},
    {"id": 5, "name": "Reposition Q3 (SE)", "type": "REPOSITION", "target_quad": 3, "desc": "Reposition to Quadrant 3 (South-East) Centroid (12, 12)"},
]

QUADRANT_NAMES = ["Q0 (North-West)", "Q1 (North-East)", "Q2 (South-West)", "Q3 (South-East)"]
PHASE_NAMES = ["Phase 0: Normal", "Phase 1: Pre-Surge Warning", "Phase 2: Active Surge", "Phase 3: Post-Surge Recovery"]
BUCKET_NAMES = ["0 requests", "1 - 2 requests", "3+ requests"]


class SimulationEngine:
    def __init__(self, q_table_path=None):
        self.lock = threading.RLock()

        if q_table_path is None:
            q_table_path = os.path.join(FLEET_RL_DIR, "q_table.npy")
        self.q_table_path = q_table_path

        # Load pristine trained model Q-table
        if os.path.exists(self.q_table_path):
            self.q_table_trained = np.load(self.q_table_path).copy()
        else:
            self.q_table_trained = np.zeros((576, 6))

        # Initialize agents
        self.agent = QLearningAgent(num_states=576, num_actions=6, lr=0.05, gamma=0.96, eps=0.1, eps_min=0.01)
        self.agent.q_table = self.q_table_trained.copy()

        self.baseline = BaselineDispatcher()
        self.forecast_rule = ForecastRuleDispatcher()

        # Config & Environment parameters (V2 authoritative)
        self.grid_size = ENV_V2["grid_size"]
        self.num_vehicles = ENV_V2["num_vehicles"]
        self.max_steps = ENV_V2["max_steps"]
        self.max_wait = ENV_V2["max_wait"]
        self.p_normal = ENV_V2["p_normal"]
        self.p_surge = ENV_V2["p_surge"]

        self.env = CityEnv(
            grid_size=self.grid_size,
            num_vehicles=self.num_vehicles,
            max_steps=self.max_steps,
            max_wait=self.max_wait,
            p_normal=self.p_normal,
            p_surge=self.p_surge,
            seed=5000
        )

        # Simulation runtime state
        self.seed = 5000
        self.surge_quad = 0
        self.dispatcher_type = "rl"       # "rl", "baseline", "rule"
        self.mode = "inference"           # "inference" (Mode A), "training" (Mode B)
        self.is_running = False
        self.speed_multiplier = 1.0       # 1.0x, 2.0x, 5.0x, 10.0x
        self.step_num = 0
        self.cumulative_reward = 0.0
        self.step_rewards = []

        # Decision & telemetry history
        self.latest_decisions = []
        self.latest_decision_by_vehicle = {}
        self.recent_decisions = []
        self.recent_transitions = []
        self.selected_vehicle_id = 0

        # Background runner thread
        self._bg_thread = None
        self._stop_bg_thread = False

        # Initial reset
        self.reset(seed=self.seed, surge_quad=self.surge_quad, dispatcher_type=self.dispatcher_type, mode=self.mode)

    @staticmethod
    def decode_state(s: int) -> dict:
        """Decode discrete state index s in [0, 575] into its 5 semantic features."""
        s = int(s)
        b_hot = s % 3
        s1 = s // 3
        b_own = s1 % 3
        s2 = s1 // 3
        hotspot = s2 % 4
        s3 = s2 // 4
        phase = s3 % 4
        v_quad = s3 // 4

        return {
            "state_id": s,
            "vehicle_quad": v_quad,
            "vehicle_quad_name": QUADRANT_NAMES[v_quad] if 0 <= v_quad < 4 else f"Q{v_quad}",
            "phase": phase,
            "phase_name": PHASE_NAMES[phase] if 0 <= phase < 4 else f"Phase {phase}",
            "hotspot_quad": hotspot,
            "hotspot_quad_name": QUADRANT_NAMES[hotspot] if 0 <= hotspot < 4 else f"Q{hotspot}",
            "own_bucket": b_own,
            "own_bucket_name": BUCKET_NAMES[b_own] if 0 <= b_own < 3 else str(b_own),
            "hotspot_bucket": b_hot,
            "hotspot_bucket_name": BUCKET_NAMES[b_hot] if 0 <= b_hot < 3 else str(b_hot),
            "formula": f"((({v_quad}*4 + {phase})*4 + {hotspot})*3 + {b_own})*3 + {b_hot}"
        }

    def get_q_values(self, state_id: int, valid_actions: list = None) -> dict:
        """Retrieve Q-values and action mask metadata for a specific discrete state."""
        state_id = max(0, min(575, int(state_id)))
        q_vals = [float(self.agent.q_table[state_id, a]) for a in range(6)]

        if valid_actions is None:
            # Default action masking when pending queue state is unknown
            # Actions 0 & 1 require pending requests; 2..5 are repos
            valid_actions = [0, 1, 2, 3, 4, 5]

        valid_set = set(valid_actions)
        valid_q = [q_vals[a] for a in valid_actions] if valid_actions else q_vals
        max_q = max(valid_q) if valid_q else max(q_vals)
        greedy_action = None
        for a in valid_actions:
            if q_vals[a] == max_q:
                greedy_action = a
                break

        actions_data = []
        for a_def in ACTION_DEFINITIONS:
            a_id = a_def["id"]
            is_valid = a_id in valid_set
            actions_data.append({
                "id": a_id,
                "name": a_def["name"],
                "type": a_def["type"],
                "desc": a_def["desc"],
                "target_quad": a_def["target_quad"],
                "q_value": round(q_vals[a_id], 4),
                "is_valid": is_valid,
                "is_greedy": (a_id == greedy_action) if is_valid else False
            })

        return {
            "state": self.decode_state(state_id),
            "actions": actions_data,
            "greedy_action_id": greedy_action,
            "max_q": round(max_q, 4)
        }

    def reset(self, seed=None, surge_quad=None, dispatcher_type=None, mode=None) -> dict:
        """Reset the simulation environment and agent state."""
        with self.lock:
            next_dispatcher = self.dispatcher_type if dispatcher_type is None else str(dispatcher_type)
            next_mode = self.mode if mode is None else str(mode)
            if next_dispatcher not in {"rl", "baseline", "rule"}:
                raise ValueError("dispatcher_type must be 'rl', 'baseline', or 'rule'")
            if next_mode not in {"inference", "training"}:
                raise ValueError("mode must be 'inference' or 'training'")

            if seed is not None:
                self.seed = int(seed)
            if surge_quad is not None:
                self.surge_quad = int(surge_quad) % 4
            self.dispatcher_type = next_dispatcher
            self.mode = next_mode

            # In inference mode, guarantee trained Q-table integrity
            if self.mode == "inference":
                self.agent.q_table = self.q_table_trained.copy()
                self.agent.eps = 0.0
            else:
                self.agent.eps = 0.10

            self.agent.active_decisions.clear()

            self.env.reset(seed=self.seed, surge_quad=self.surge_quad)
            self.step_num = 0
            self.cumulative_reward = 0.0
            self.step_rewards = []
            self.latest_decisions = []
            self.latest_decision_by_vehicle = {}
            self.recent_decisions = []
            self.recent_transitions = []

            return self.get_state()

    def set_dispatcher(self, dispatcher_type: str) -> dict:
        """Change dispatcher: 'rl', 'baseline', or 'rule'."""
        with self.lock:
            dispatcher_type = str(dispatcher_type)
            if dispatcher_type not in {"rl", "baseline", "rule"}:
                raise ValueError("dispatcher_type must be 'rl', 'baseline', or 'rule'")
            if dispatcher_type != self.dispatcher_type:
                self.agent.active_decisions.clear()
            self.dispatcher_type = dispatcher_type
            return {"status": "ok", "dispatcher_type": self.dispatcher_type}

    def set_mode(self, mode: str) -> dict:
        """Change mode: 'inference' (Mode A) or 'training' (Mode B)."""
        with self.lock:
            mode = str(mode)
            if mode not in {"inference", "training"}:
                raise ValueError("mode must be 'inference' or 'training'")
            if mode != self.mode:
                self.agent.active_decisions.clear()
            self.mode = mode
            if self.mode == "inference":
                self.agent.q_table = self.q_table_trained.copy()
                self.agent.eps = 0.0
            else:
                self.agent.eps = 0.10
            return {"status": "ok", "mode": self.mode, "eps": self.agent.eps}

    def set_speed(self, speed: float) -> dict:
        """Set simulation speed multiplier."""
        with self.lock:
            self.speed_multiplier = max(0.2, min(20.0, float(speed)))
            return {"status": "ok", "speed_multiplier": self.speed_multiplier}

    def set_selected_vehicle(self, v_id: int) -> dict:
        """Select a vehicle for live inspector telemetry."""
        with self.lock:
            self.selected_vehicle_id = max(0, min(self.num_vehicles - 1, int(v_id)))
            return {"status": "ok", "selected_vehicle_id": self.selected_vehicle_id}

    def step(self) -> dict:
        """
        Execute exactly ONE actual simulation timestep.
        Dispatches idle vehicles according to the active policy,
        moves vehicles, spawns passenger requests, handles dropoffs and cancellations,
        and computes dynamic rewards and Q-table updates.
        """
        with self.lock:
            if self.env.step_num >= self.max_steps:
                return self.get_state()

            step_decisions = []
            step_transitions = []

            # 1. Dispatch Idle Vehicles
            for v in self.env.vehicles:
                if v["status"] == "idle":
                    v_id = v["id"]
                    val_acts = self.env.get_valid_actions(v_id)
                    s = self.env.encode_state(v_id)

                    if self.dispatcher_type == "rl":
                        # In training mode, finalize previous multi-step decision if applicable
                        if self.mode == "training":
                            tr = self.agent.finish_decision(v_id, next_state=s, next_valid_actions=val_acts, done=False)
                            if tr is not None:
                                step_transitions.append(tr)
                                self.recent_transitions.append(tr)
                                if len(self.recent_transitions) > 50:
                                    self.recent_transitions.pop(0)

                        # Retrieve Q-values for all 6 actions
                        q_vals = [float(self.agent.q_table[s, a]) for a in range(6)]
                        explore = (self.mode == "training")
                        chosen_action = self.agent.select_action(s, val_acts, explore=explore)

                        # Check if action was selected greedily
                        valid_q = [q_vals[a] for a in val_acts]
                        max_valid_q = max(valid_q) if valid_q else max(q_vals)
                        is_greedy = (q_vals[chosen_action] == max_valid_q)

                        dispatched = self.env.dispatch(v_id, chosen_action)
                        if dispatched and (self.mode == "training"):
                            self.agent.start_decision(v_id, s, chosen_action)

                        action_info = ACTION_DEFINITIONS[chosen_action]
                        dec_record = {
                            "step": self.env.step_num,
                            "vehicle_id": v_id,
                            "policy": "Q-Learning (V2)",
                            "state_id": s,
                            "state_decoded": self.decode_state(s),
                            "valid_actions": val_acts,
                            "q_values": [round(q, 4) for q in q_vals],
                            "selected_action": chosen_action,
                            "action_name": action_info["name"],
                            "action_type": action_info["type"],
                            "action_desc": action_info["desc"],
                            "is_greedy": is_greedy,
                            "selection_method": "Greedy (Exploit)" if is_greedy else "Exploration (Epsilon)",
                            "dispatched": dispatched,
                            "new_status": v["status"],
                            "target": (v["target_x"], v["target_y"]) if v["target_x"] is not None else None,
                            "assigned_req_id": v["req_id"]
                        }

                    elif self.dispatcher_type == "baseline":
                        chosen_action = self.baseline.select_action(self.env, v_id)
                        dispatched = False
                        if chosen_action is not None:
                            dispatched = self.env.dispatch(v_id, chosen_action)

                        dec_record = {
                            "step": self.env.step_num,
                            "vehicle_id": v_id,
                            "policy": "Nearest-Vehicle Baseline",
                            "state_id": s,
                            "state_decoded": self.decode_state(s),
                            "valid_actions": val_acts,
                            "q_values": None,
                            "selected_action": chosen_action,
                            "action_name": ACTION_DEFINITIONS[chosen_action]["name"] if chosen_action is not None else "Idle Wait",
                            "action_type": "DISPATCH" if chosen_action == 0 else "IDLE",
                            "action_desc": "Serve Nearest Pending Request" if chosen_action == 0 else "No pending requests to serve",
                            "is_greedy": True,
                            "selection_method": "Heuristic (Nearest)",
                            "dispatched": dispatched,
                            "new_status": v["status"],
                            "target": (v["target_x"], v["target_y"]) if v["target_x"] is not None else None,
                            "assigned_req_id": v["req_id"]
                        }

                    else:  # "rule" - ForecastRuleDispatcher
                        chosen_action = self.forecast_rule.select_action(self.env, v_id)
                        dispatched = False
                        if chosen_action is not None:
                            dispatched = self.env.dispatch(v_id, chosen_action)

                        action_name = ACTION_DEFINITIONS[chosen_action]["name"] if chosen_action is not None else "Idle Wait"
                        dec_record = {
                            "step": self.env.step_num,
                            "vehicle_id": v_id,
                            "policy": "Forecast-Aware Rule",
                            "state_id": s,
                            "state_decoded": self.decode_state(s),
                            "valid_actions": val_acts,
                            "q_values": None,
                            "selected_action": chosen_action,
                            "action_name": action_name,
                            "action_type": "REPOSITION" if (chosen_action is not None and chosen_action >= 2) else ("DISPATCH" if chosen_action == 0 else "IDLE"),
                            "action_desc": "Proactive Hotspot Pre-positioning" if (chosen_action is not None and chosen_action >= 2) else "Nearest Request",
                            "is_greedy": True,
                            "selection_method": "Rule (Forecast-Guided)",
                            "dispatched": dispatched,
                            "new_status": v["status"],
                            "target": (v["target_x"], v["target_y"]) if v["target_x"] is not None else None,
                            "assigned_req_id": v["req_id"]
                        }

                    step_decisions.append(dec_record)
                    self.latest_decision_by_vehicle[v_id] = dec_record
                    self.recent_decisions.append(dec_record)
                    if len(self.recent_decisions) > 50:
                        self.recent_decisions.pop(0)

            self.latest_decisions = step_decisions

            # 2. Advance Environment by 1 Timestep
            step_reward, done, info = self.env.step()
            self.step_num = self.env.step_num
            self.cumulative_reward += step_reward
            self.step_rewards.append(round(step_reward, 2))

            # 3. Semi-Markov credit assignment (Mode B: Training Mode)
            if self.mode == "training":
                shared = (-0.15 * info["pending"] - 25.0 * info["drops"]) / self.num_vehicles
                for v in self.env.vehicles:
                    r = shared
                    if info["dropoffs_by_v"][v["id"]]:
                        r += 20.0
                    if info["moved_by_v"][v["id"]]:
                        r -= 0.05
                    self.agent.add_step_reward(v["id"], r)

                if done:
                    terminal_states = self.env.get_state_for_all()
                    for v in self.env.vehicles:
                        tr = self.agent.finish_decision(v["id"], next_state=terminal_states[v["id"]], done=True)
                        if tr is not None:
                            step_transitions.append(tr)
                            self.recent_transitions.append(tr)
                    self.agent.decay_eps()

            return self.get_state(step_info={
                "step_reward": round(step_reward, 2),
                "dropoffs": info["dropoffs"],
                "drops": info["drops"],
                "distance": info["distance"],
                "pending": info["pending"],
                "transitions": step_transitions
            })

    def get_state(self, step_info=None) -> dict:
        """Assemble complete live snapshot for frontend visualization and inspection."""
        with self.lock:
            metrics = self.env.get_metrics()
            frame = self.env.get_frame()

            total_resolved = metrics["served"] + metrics["dropped"]
            service_rate = round((metrics["served"] / total_resolved) * 100.0, 1) if total_resolved > 0 else 100.0

            # Active requests breakdown
            pending_requests = [
                {
                    "id": r["id"],
                    "px": r["px"],
                    "py": r["py"],
                    "dx": r["dx"],
                    "dy": r["dy"],
                    "spawn_step": r["spawn_step"],
                    "wait": r["wait"],
                    "status": r["status"],
                    "assigned_vehicle": r["assigned_vehicle"],
                    "quadrant": self.env.get_quadrant(r["px"], r["py"]),
                    "quadrant_name": QUADRANT_NAMES[self.env.get_quadrant(r["px"], r["py"])],
                    "manhattan_dist": abs(r["dx"] - r["px"]) + abs(r["dy"] - r["py"])
                }
                for r in self.env.requests
                if r["status"] in ("pending", "assigned", "in_transit")
            ]

            # Recent completed and dropped (last 5)
            recent_completed = [
                {"id": r["id"], "px": r["px"], "py": r["py"], "dx": r["dx"], "dy": r["dy"], "wait": r["wait"]}
                for r in self.env.requests if r["status"] == "completed"
            ][-5:]

            recent_dropped = [
                {"id": r["id"], "px": r["px"], "py": r["py"], "wait": r["wait"]}
                for r in self.env.requests if r["status"] == "dropped"
            ][-5:]

            # Inspector data for selected vehicle
            sel_v_id = self.selected_vehicle_id
            sel_v = self.env.vehicles[sel_v_id] if 0 <= sel_v_id < len(self.env.vehicles) else self.env.vehicles[0]
            sel_state = self.env.encode_state(sel_v["id"])
            sel_val_acts = self.env.get_valid_actions(sel_v["id"])
            inspector_data = self.get_q_values(sel_state, valid_actions=sel_val_acts)
            inspector_data["selected_vehicle"] = {
                "id": sel_v["id"],
                "x": sel_v["x"],
                "y": sel_v["y"],
                "status": sel_v["status"],
                "target_x": sel_v["target_x"],
                "target_y": sel_v["target_y"],
                "req_id": sel_v["req_id"],
                "total_dist": sel_v["total_dist"],
                "idle_dist": sel_v["idle_dist"],
                "trips": sel_v["trips"],
                "quadrant": self.env.get_quadrant(sel_v["x"], sel_v["y"]),
                "quadrant_name": QUADRANT_NAMES[self.env.get_quadrant(sel_v["x"], sel_v["y"])]
            }
            inspector_data["latest_decision"] = self.latest_decision_by_vehicle.get(sel_v["id"])

            return {
                "step": self.env.step_num,
                "max_steps": self.max_steps,
                "done": (self.env.step_num >= self.max_steps),
                "is_running": self.is_running,
                "speed_multiplier": self.speed_multiplier,
                "dispatcher_type": self.dispatcher_type,
                "mode": self.mode,
                "seed": self.seed,
                "surge_quad": self.surge_quad,
                "surge_quad_name": QUADRANT_NAMES[self.surge_quad],
                "phase": self.env.get_phase(),
                "phase_name": PHASE_NAMES[self.env.get_phase()],
                "is_surge": (100 <= self.env.step_num < 200),
                "grid_size": self.grid_size,
                "num_vehicles": self.num_vehicles,
                "vehicles": [
                    {
                        "id": v["id"],
                        "x": v["x"],
                        "y": v["y"],
                        "status": v["status"],
                        "target_x": v["target_x"],
                        "target_y": v["target_y"],
                        "req_id": v["req_id"],
                        "total_dist": v["total_dist"],
                        "idle_dist": v["idle_dist"],
                        "trips": v["trips"],
                        "quadrant": self.env.get_quadrant(v["x"], v["y"])
                    }
                    for v in self.env.vehicles
                ],
                "pending": pending_requests,
                "active_requests": pending_requests,
                "recent_completed": recent_completed,
                "recent_dropped": recent_dropped,
                "metrics": {
                    "served": metrics["served"],
                    "dropped": metrics["dropped"],
                    "service_rate": service_rate,
                    "avg_wait": metrics["avg_wait"],
                    "surge_wait": metrics["surge_wait"],
                    "normal_wait": metrics["normal_wait"],
                    "total_dist": metrics["total_dist"],
                    "idle_dist": metrics["idle_dist"],
                    "cumulative_reward": round(self.cumulative_reward, 2),
                    "pending_count": len([r for r in self.env.requests if r["status"] == "pending"]),
                    "in_transit_count": len([r for r in self.env.requests if r["status"] == "in_transit"]),
                    "total_spawned": len(self.env.requests)
                },
                "latest_decisions": self.latest_decisions,
                "recent_decisions": self.recent_decisions[-15:],
                "recent_transitions": self.recent_transitions[-10:],
                "inspector": inspector_data,
                "step_info": step_info
            }

    def run_paired_comparison(self, seed: int = None, surge_quad: int = None) -> dict:
        """
        Run an exact paired side-by-side benchmark comparison under identical initial conditions:
        Identical seed, identical passenger realizations, identical duration (300 steps).
        Evaluates Nearest-Vehicle Baseline vs Forecast-Aware Rule vs Tabular Q-Learning.
        """
        with self.lock:
            eval_seed = int(seed) if seed is not None else self.seed
            eval_surge = (int(surge_quad) % 4) if surge_quad is not None else self.surge_quad

            def run_single_dispatcher(dispatcher_code):
                test_env = CityEnv(
                    grid_size=self.grid_size,
                    num_vehicles=self.num_vehicles,
                    max_steps=self.max_steps,
                    max_wait=self.max_wait,
                    p_normal=self.p_normal,
                    p_surge=self.p_surge,
                    seed=eval_seed
                )
                test_env.reset(seed=eval_seed, surge_quad=eval_surge)
                ep_reward = 0.0

                frames = []
                for t in range(self.max_steps):
                    frames.append(test_env.get_frame())
                    for v in test_env.vehicles:
                        if v["status"] == "idle":
                            v_id = v["id"]
                            if dispatcher_code == "baseline":
                                a = self.baseline.select_action(test_env, v_id)
                                if a is not None:
                                    test_env.dispatch(v_id, a)
                            elif dispatcher_code == "rule":
                                a = self.forecast_rule.select_action(test_env, v_id)
                                if a is not None:
                                    test_env.dispatch(v_id, a)
                            elif dispatcher_code == "rl":
                                s = test_env.encode_state(v_id)
                                val_acts = test_env.get_valid_actions(v_id)
                                a = self.agent.select_action(s, val_acts, explore=False)
                                test_env.dispatch(v_id, a)

                    r, done, _ = test_env.step()
                    ep_reward += r

                m = test_env.get_metrics()
                total_res = m["served"] + m["dropped"]
                svc_rate = round((m["served"] / total_res) * 100.0, 1) if total_res > 0 else 100.0

                return {
                    "avg_wait": m["avg_wait"],
                    "surge_wait": m["surge_wait"],
                    "normal_wait": m["normal_wait"],
                    "served": m["served"],
                    "dropped": m["dropped"],
                    "service_rate": svc_rate,
                    "total_dist": m["total_dist"],
                    "idle_dist": m["idle_dist"],
                    "cumulative_reward": round(ep_reward, 2),
                    "frames": frames
                }

            base_res = run_single_dispatcher("baseline")
            rule_res = run_single_dispatcher("rule")
            rl_res = run_single_dispatcher("rl")

            # Comparative deltas (RL vs Baseline)
            delta_avg_wait = round(rl_res["avg_wait"] - base_res["avg_wait"], 2)
            delta_surge_wait = round(rl_res["surge_wait"] - base_res["surge_wait"], 2)
            delta_served = rl_res["served"] - base_res["served"]
            delta_dropped = rl_res["dropped"] - base_res["dropped"]
            delta_reward = round(rl_res["cumulative_reward"] - base_res["cumulative_reward"], 2)

            return {
                "seed": eval_seed,
                "surge_quad": eval_surge,
                "surge_quad_name": QUADRANT_NAMES[eval_surge],
                "baseline": {
                    "name": "Nearest-Vehicle Baseline",
                    "metrics": {k: v for k, v in base_res.items() if k != "frames"}
                },
                "forecast_rule": {
                    "name": "Forecast-Aware Rule",
                    "metrics": {k: v for k, v in rule_res.items() if k != "frames"}
                },
                "qlearning": {
                    "name": "Tabular Q-Learning (V2)",
                    "metrics": {k: v for k, v in rl_res.items() if k != "frames"}
                },
                "deltas_rl_vs_baseline": {
                    "avg_wait_diff": delta_avg_wait,
                    "avg_wait_winner": "RL" if delta_avg_wait < 0 else ("Baseline" if delta_avg_wait > 0 else "Tie"),
                    "surge_wait_diff": delta_surge_wait,
                    "surge_wait_winner": "RL" if delta_surge_wait < 0 else ("Baseline" if delta_surge_wait > 0 else "Tie"),
                    "served_diff": delta_served,
                    "dropped_diff": delta_dropped,
                    "reward_diff": delta_reward
                },
                "baseline_frames": base_res["frames"],
                "rl_frames": rl_res["frames"]
            }
