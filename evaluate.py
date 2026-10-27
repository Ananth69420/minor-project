"""
Evaluation and Benchmarking Pipeline.
Compares Tabular Q-Learning Dispatcher against Nearest-Vehicle Baseline Dispatcher
over identical seeded simulation episodes, recording real metrics and replay trajectories.
"""

import json
import time
from typing import Dict, List, Any
import numpy as np
import config
from env.city import CityFleetEnv
from agents.baseline import NearestVehicleDispatcher
from agents.qlearning import QLearningAgent


def evaluate_policy(
    env: CityFleetEnv,
    agent_type: str,
    q_agent: QLearningAgent = None,
    baseline_agent: NearestVehicleDispatcher = None,
    episode_seed: int = 42,
    surge_quadrant: int = 2,
    record_frames: bool = False,
) -> Dict[str, Any]:
    """
    Runs a single evaluation episode with a fixed seed and surge quadrant.
    Returns metrics and optional step-by-step replay frames.
    """
    env.reset(seed=episode_seed, surge_quadrant=surge_quadrant)

    step_queue_lengths: List[int] = []
    frames: List[Dict[str, Any]] = []

    for step in range(config.EPISODE_STEPS):
        # 1. Idle vehicles make decisions
        for v in env.vehicles:
            if v.status == "IDLE":
                if agent_type == "qlearning":
                    s = env.encode_state(v.id)
                    a = q_agent.select_action(s, evaluate=True)
                    env.dispatch_vehicle(v.id, a)
                elif agent_type == "baseline":
                    a = baseline_agent.select_action(env, v.id)
                    if a is not None:
                        env.dispatch_vehicle(v.id, a)

        # Record visualizer frame if requested
        if record_frames:
            frames.append(env.get_state_summary())

        # 2. Advance simulation
        step_reward, done, info = env.step()
        step_queue_lengths.append(info["pending_count"])

        if done:
            break

    metrics = env.get_metrics()
    metrics["step_queue_lengths"] = step_queue_lengths

    result = {"metrics": metrics}
    if record_frames:
        result["frames"] = frames
    return result


def run_benchmark(
    num_episodes: int = config.NUM_EVAL_EPISODES,
    q_table_path: str = "q_table.npy",
    save_results_path: str = "simulation_results.json",
    seed: int = 12345,
) -> Dict[str, Any]:
    """
    Runs full benchmark across N paired episodes with identical seeds.
    """
    print("=" * 70)
    print("STARTING BENCHMARK: Q-LEARNING vs. NEAREST-VEHICLE BASELINE")
    print(f"Evaluation Episodes: {num_episodes} (Paired Seeds)")
    print(f"Loading Q-table from: '{q_table_path}'")
    print("=" * 70)

    env = CityFleetEnv()
    baseline = NearestVehicleDispatcher()
    rl_agent = QLearningAgent()
    rl_agent.load(q_table_path)

    # Deterministic evaluation seeds & surge hotspots
    rng = np.random.default_rng(seed)
    eval_seeds = [int(rng.integers(0, 1000000)) for _ in range(num_episodes)]
    eval_surges = [int(rng.integers(0, 4)) for _ in range(num_episodes)]

    baseline_metrics_list: List[Dict[str, Any]] = []
    rl_metrics_list: List[Dict[str, Any]] = []

    # Store step queue length averages across all episodes
    baseline_avg_queues = np.zeros(config.EPISODE_STEPS)
    rl_avg_queues = np.zeros(config.EPISODE_STEPS)

    # Trajectory recordings for dashboard replay (first episode)
    sample_baseline_frames = None
    sample_rl_frames = None

    t0 = time.time()
    for ep in range(num_episodes):
        ep_seed = eval_seeds[ep]
        sq = eval_surges[ep]
        record_traj = (ep == 0)

        # Run Baseline
        base_res = evaluate_policy(
            env=env,
            agent_type="baseline",
            baseline_agent=baseline,
            episode_seed=ep_seed,
            surge_quadrant=sq,
            record_frames=record_traj,
        )
        base_m = base_res["metrics"]
        baseline_metrics_list.append(base_m)
        baseline_avg_queues += np.array(base_m["step_queue_lengths"])
        if record_traj:
            sample_baseline_frames = base_res["frames"]

        # Run Q-Learning
        rl_res = evaluate_policy(
            env=env,
            agent_type="qlearning",
            q_agent=rl_agent,
            episode_seed=ep_seed,
            surge_quadrant=sq,
            record_frames=record_traj,
        )
        rl_m = rl_res["metrics"]
        rl_metrics_list.append(rl_m)
        rl_avg_queues += np.array(rl_m["step_queue_lengths"])
        if record_traj:
            sample_rl_frames = rl_res["frames"]

    elapsed = time.time() - t0
    baseline_avg_queues /= num_episodes
    rl_avg_queues /= num_episodes

    # Helper function to compute mean and std
    def stats(key: str, data: List[Dict[str, Any]]) -> Dict[str, float]:
        vals = [d[key] for d in data]
        return {
            "mean": round(float(np.mean(vals)), 2),
            "std": round(float(np.std(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2),
        }

    keys = [
        "avg_wait_time", "avg_wait_all", "requests_served", "requests_dropped",
        "service_rate", "total_distance", "idle_distance", "empty_ratio",
        "total_reward", "surge_service_rate", "surge_avg_wait"
    ]

    summary_baseline = {k: stats(k, baseline_metrics_list) for k in keys}
    summary_rl = {k: stats(k, rl_metrics_list) for k in keys}

    # Calculate percentage improvements
    wait_reduction = (
        (summary_baseline["avg_wait_time"]["mean"] - summary_rl["avg_wait_time"]["mean"]) /
        summary_baseline["avg_wait_time"]["mean"] * 100.0
    ) if summary_baseline["avg_wait_time"]["mean"] > 0 else 0.0

    drop_reduction = (
        (summary_baseline["requests_dropped"]["mean"] - summary_rl["requests_dropped"]["mean"]) /
        summary_baseline["requests_dropped"]["mean"] * 100.0
    ) if summary_baseline["requests_dropped"]["mean"] > 0 else 0.0

    service_gain = summary_rl["service_rate"]["mean"] - summary_baseline["service_rate"]["mean"]
    surge_service_gain = summary_rl["surge_service_rate"]["mean"] - summary_baseline["surge_service_rate"]["mean"]

    comparison = {
        "wait_time_reduction_pct": round(wait_reduction, 2),
        "cancellation_reduction_pct": round(drop_reduction, 2),
        "service_rate_gain_pct": round(service_gain, 2),
        "surge_service_gain_pct": round(surge_service_gain, 2),
    }

    # Print Formatted Results Table for Examination
    print("\n" + "=" * 78)
    print(f"BENCHMARK RESULTS SUMMARY ({num_episodes} EVALUATION EPISODES)")
    print("=" * 78)
    print(f"{'METRIC':<30} | {'BASELINE (NEAREST)':<18} | {'RL (Q-LEARNING)':<18} | {'IMPROVEMENT':<10}")
    print("-" * 78)
    print(f"{'Average Wait Time (s)':<30} | {summary_baseline['avg_wait_time']['mean']:>6.2f} +/- {summary_baseline['avg_wait_time']['std']:<6.2f} | {summary_rl['avg_wait_time']['mean']:>6.2f} +/- {summary_rl['avg_wait_time']['std']:<6.2f} | {wait_reduction:>+7.1f}%")
    print(f"{'Requests Served (count)':<30} | {summary_baseline['requests_served']['mean']:>6.2f} +/- {summary_baseline['requests_served']['std']:<6.2f} | {summary_rl['requests_served']['mean']:>6.2f} +/- {summary_rl['requests_served']['std']:<6.2f} | {summary_rl['requests_served']['mean'] - summary_baseline['requests_served']['mean']:>+7.1f}")
    print(f"{'Requests Dropped (cancellations)':<30} | {summary_baseline['requests_dropped']['mean']:>6.2f} +/- {summary_baseline['requests_dropped']['std']:<6.2f} | {summary_rl['requests_dropped']['mean']:>6.2f} +/- {summary_rl['requests_dropped']['std']:<6.2f} | {drop_reduction:>+7.1f}%")
    print(f"{'Overall Service Rate (%)':<30} | {summary_baseline['service_rate']['mean']:>6.2f}% +/- {summary_baseline['service_rate']['std']:<5.2f} | {summary_rl['service_rate']['mean']:>6.2f}% +/- {summary_rl['service_rate']['std']:<5.2f} | {service_gain:>+7.1f}%")
    print(f"{'Surge Phase Service Rate (%)':<30} | {summary_baseline['surge_service_rate']['mean']:>6.2f}% +/- {summary_baseline['surge_service_rate']['std']:<5.2f} | {summary_rl['surge_service_rate']['mean']:>6.2f}% +/- {summary_rl['surge_service_rate']['std']:<5.2f} | {surge_service_gain:>+7.1f}%")
    print(f"{'Surge Phase Avg Wait (s)':<30} | {summary_baseline['surge_avg_wait']['mean']:>6.2f} +/- {summary_baseline['surge_avg_wait']['std']:<6.2f} | {summary_rl['surge_avg_wait']['mean']:>6.2f} +/- {summary_rl['surge_avg_wait']['std']:<6.2f} | {(summary_baseline['surge_avg_wait']['mean'] - summary_rl['surge_avg_wait']['mean'])/summary_baseline['surge_avg_wait']['mean']*100:>+7.1f}%")
    print(f"{'Total Distance (fuel proxy)':<30} | {summary_baseline['total_distance']['mean']:>6.2f} +/- {summary_baseline['total_distance']['std']:<6.2f} | {summary_rl['total_distance']['mean']:>6.2f} +/- {summary_rl['total_distance']['std']:<6.2f} | -")
    print(f"{'Idle Distance (empty moves)':<30} | {summary_baseline['idle_distance']['mean']:>6.2f} +/- {summary_baseline['idle_distance']['std']:<6.2f} | {summary_rl['idle_distance']['mean']:>6.2f} +/- {summary_rl['idle_distance']['std']:<6.2f} | -")
    print(f"{'Cumulative Fleet Reward':<30} | {summary_baseline['total_reward']['mean']:>6.2f} +/- {summary_baseline['total_reward']['std']:<6.2f} | {summary_rl['total_reward']['mean']:>6.2f} +/- {summary_rl['total_reward']['std']:<6.2f} | {summary_rl['total_reward']['mean'] - summary_baseline['total_reward']['mean']:>+7.1f}")
    print("=" * 78)
    print(f"Benchmark finished in {elapsed:.2f} seconds ({elapsed/num_episodes*1000:.1f}ms/paired-episode)")

    full_export = {
        "config": {
            "grid_size": config.GRID_SIZE,
            "num_vehicles": config.NUM_VEHICLES,
            "episode_steps": config.EPISODE_STEPS,
            "max_wait_time": config.MAX_WAIT_TIME,
            "surge_start": config.SURGE_START_STEP,
            "surge_end": config.SURGE_END_STEP,
            "num_eval_episodes": num_episodes,
        },
        "summary": {
            "baseline": summary_baseline,
            "qlearning": summary_rl,
            "comparison": comparison,
        },
        "time_series": {
            "steps": list(range(config.EPISODE_STEPS)),
            "baseline_avg_queue": [round(x, 2) for x in baseline_avg_queues.tolist()],
            "rl_avg_queue": [round(x, 2) for x in rl_avg_queues.tolist()],
        },
        "episodes": {
            "baseline": baseline_metrics_list,
            "qlearning": rl_metrics_list,
        },
        "replay": {
            "baseline": sample_baseline_frames,
            "qlearning": sample_rl_frames,
        }
    }

    with open(save_results_path, "w", encoding="utf-8") as f:
        json.dump(full_export, f, indent=2)
    print(f"Complete simulation results exported to '{save_results_path}'.")

    return full_export


if __name__ == "__main__":
    run_benchmark()
