"""Adapt the root simulator's outputs to the dashboard's REPLAY_DATA contract."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import config


def _phase(step: int) -> int:
    if step < config.PRE_SURGE_START_STEP:
        return 0
    if step < config.SURGE_START_STEP:
        return 1
    if step < config.SURGE_END_STEP:
        return 2
    return 3


def _frame(source: dict[str, Any]) -> dict[str, Any]:
    vehicles = []
    for vehicle in source["vehicles"]:
        pos, target = vehicle["pos"], vehicle["target_pos"]
        status = {"IDLE": "idle", "DISPATCHED_PICKUP": "pickup", "CARRYING_TRIP": "trip", "REPOSITIONING": "repo"}[vehicle["status"]]
        vehicles.append({"id": vehicle["id"], "x": pos[0], "y": pos[1], "status": status,
                         "target_x": target[0] if target else None, "target_y": target[1] if target else None,
                         "req_id": vehicle["current_request_id"]})
    pending = []
    for request in source["requests"]:
        if request["status"] not in ("PENDING", "ASSIGNED"):
            continue
        pending.append({"id": request["id"], "px": request["pickup"][0], "py": request["pickup"][1],
                        "dx": request["dropoff"][0], "dy": request["dropoff"][1], "wait": request["wait_time"],
                        "status": request["status"].lower(), "assigned_vehicle": request["assigned_vehicle_id"]})
    step = source["step"]
    return {"step": step, "phase": _phase(step), "surge_quad": source["surge_quadrant"],
            "is_surge": source["is_surge"], "grid_size": config.GRID_SIZE, "num_vehicles": config.NUM_VEHICLES,
            "total_requests": source["total_requests"], "served": source["served"],
        "dropped": source["dropped"], "avg_wait": source["avg_wait_time"],
            "vehicles": vehicles, "pending": pending}


def build_replay_data(simulation: dict[str, Any], training: dict[str, Any], q_table: np.ndarray) -> dict[str, Any]:
    cfg, summary, episodes = simulation["config"], simulation["summary"], simulation["episodes"]
    def policy_stats(policy: str) -> dict[str, Any]:
        values = summary[policy]
        normal_wait = [episode["normal_avg_wait_time"] for episode in episodes[policy]]
        return {"avg_wait": values["avg_wait_time"], "surge_wait": values["surge_avg_wait"],
                "normal_wait": {"mean": round(float(np.mean(normal_wait)), 2), "std": round(float(np.std(normal_wait)), 2)},
                "served": values["requests_served"], "dropped": values["requests_dropped"],
                "total_dist": values["total_distance"], "service_rate": values["service_rate"]}
    return {"project": {"name": "Autonomous Fleet Dispatch", "version": "root-v1",
                        "grid_size": cfg["grid_size"], "num_vehicles": cfg["num_vehicles"],
                        "episode_steps": cfg["episode_steps"], "num_episodes": cfg["num_eval_episodes"],
                        "evaluation_seed": cfg.get("evaluation_seed"),
                        "normal_demand_probability": config.NORMAL_DEMAND_PROB,
                        "surge_demand_probability": config.SURGE_DEMAND_PROB},
            "results": {"version": "root-v1", "grid_size": cfg["grid_size"], "num_vehicles": cfg["num_vehicles"],
                        "p_normal": config.NORMAL_DEMAND_PROB, "p_surge": config.SURGE_DEMAND_PROB,
                        "num_episodes": cfg["num_eval_episodes"], "baseline": policy_stats("baseline"),
                        "qlearning": policy_stats("qlearning")},
            "training": training, "q_table": np.asarray(q_table).round(6).tolist(),
            "baseline_frames": [_frame(frame) for frame in simulation["replay"]["baseline"]],
            "rl_frames": [_frame(frame) for frame in simulation["replay"]["qlearning"]]}


def write_replay_data(simulation_path="simulation_results.json", training_path="training_history.json",
                      q_table_path="q_table.npy", output_path="replay_data.js") -> None:
    with open(simulation_path, "r", encoding="utf-8") as source:
        simulation = json.load(source)
    with open(training_path, "r", encoding="utf-8") as source:
        training = json.load(source)
    data = build_replay_data(simulation, training, np.load(q_table_path))
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    Path(output_path).write_text(
        "var REPLAY_DATA = " + payload + ";\n"
        "if (typeof window !== 'undefined') { window.REPLAY_DATA = REPLAY_DATA; }\n",
        encoding="utf-8",
    )