import os
import sys
import json
import numpy as np
from env.city import CityEnv
from agents.baseline import BaselineDispatcher, ForecastRuleDispatcher
from agents.qlearning import QLearningAgent
from config import ENV_V1, ENV_V2

def evaluate(version="v2", num_eval_episodes=50, save_files=True):
    cfg = ENV_V2 if version == "v2" else ENV_V1
    print(f"\n==========================================================================================")
    print(f"RUNNING FINAL THREE-POLICY EVALUATION ({cfg['version'].upper()}: Grid={cfg['grid_size']}x{cfg['grid_size']}, Fleet={cfg['num_vehicles']})")
    print(f"Test Seeds: 5000 to {5000 + num_eval_episodes - 1} (Strictly Reserved Test Seeds, N={num_eval_episodes})")
    print(f"==========================================================================================")

    env = CityEnv(
        grid_size=cfg["grid_size"],
        num_vehicles=cfg["num_vehicles"],
        max_steps=cfg["max_steps"],
        max_wait=cfg["max_wait"],
        p_normal=cfg["p_normal"],
        p_surge=cfg["p_surge"],
        seed=42
    )

    baseline = BaselineDispatcher()
    forecast_rule = ForecastRuleDispatcher()
    agent = QLearningAgent(num_states=576, num_actions=6)
    agent.load("q_table.npy")

    base_metrics = []
    rule_metrics = []
    rl_metrics = []

    # Store full episode frames for replay on episode 0
    base_replay_frames = []
    rl_replay_frames = []

    # Final test seeds: strictly 5000 to 5000 + num_eval_episodes - 1
    for ep in range(num_eval_episodes):
        ep_seed = 5000 + ep
        surge_quad = ep % 4

        # 1. Run Baseline (Nearest Vehicle)
        env.reset(seed=ep_seed, surge_quad=surge_quad)
        for step in range(300):
            if ep == 0:
                base_replay_frames.append(env.get_frame())
            for v in env.vehicles:
                if v["status"] == "idle":
                    a = baseline.select_action(env, v["id"])
                    if a is not None:
                        env.dispatch(v["id"], a)
            env.step()
        base_metrics.append(env.get_metrics())

        # 2. Run Forecast-Aware Rule
        env.reset(seed=ep_seed, surge_quad=surge_quad)
        for step in range(300):
            for v in env.vehicles:
                if v["status"] == "idle":
                    a = forecast_rule.select_action(env, v["id"])
                    if a is not None:
                        env.dispatch(v["id"], a)
            env.step()
        rule_metrics.append(env.get_metrics())

        # 3. Run RL Agent (Q-Learning with Action Masking)
        env.reset(seed=ep_seed, surge_quad=surge_quad)
        for step in range(300):
            if ep == 0:
                rl_replay_frames.append(env.get_frame())
            for v in env.vehicles:
                if v["status"] == "idle":
                    s = env.encode_state(v["id"])
                    val_acts = env.get_valid_actions(v["id"])
                    a = agent.select_action(s, val_acts, explore=False)
                    env.dispatch(v["id"], a)
            env.step()
        rl_metrics.append(env.get_metrics())

    def calc_stats(metric_list, key):
        vals = [m[key] for m in metric_list]
        return {
            "mean": round(float(np.mean(vals)), 2),
            "std": round(float(np.std(vals)), 2)
        }

    keys = ["avg_wait", "surge_wait", "normal_wait", "served", "dropped", "total_dist"]
    base_summary = {k: calc_stats(base_metrics, k) for k in keys}
    rule_summary = {k: calc_stats(rule_metrics, k) for k in keys}
    rl_summary = {k: calc_stats(rl_metrics, k) for k in keys}

    results = {
        "version": version,
        "grid_size": cfg["grid_size"],
        "num_vehicles": cfg["num_vehicles"],
        "p_normal": cfg["p_normal"],
        "p_surge": cfg["p_surge"],
        "num_episodes": num_eval_episodes,
        "baseline": base_summary,
        "forecast_rule": rule_summary,
        "qlearning": rl_summary
    }

    # Print three-policy results table
    print("\n" + "=" * 90)
    print(f"{'Metric':<26} | {'Baseline (Nearest)':<18} | {'Forecast Rule':<18} | {'RL (Q-Learning)':<18}")
    print("-" * 90)
    print(f"{'Average Wait (steps)':<26} | {base_summary['avg_wait']['mean']:>6.2f} +/- {base_summary['avg_wait']['std']:<6.2f} | {rule_summary['avg_wait']['mean']:>6.2f} +/- {rule_summary['avg_wait']['std']:<6.2f} | {rl_summary['avg_wait']['mean']:>6.2f} +/- {rl_summary['avg_wait']['std']:<6.2f}")
    print(f"{'Surge Wait (steps)':<26} | {base_summary['surge_wait']['mean']:>6.2f} +/- {base_summary['surge_wait']['std']:<6.2f} | {rule_summary['surge_wait']['mean']:>6.2f} +/- {rule_summary['surge_wait']['std']:<6.2f} | {rl_summary['surge_wait']['mean']:>6.2f} +/- {rl_summary['surge_wait']['std']:<6.2f}")
    print(f"{'Normal Wait (steps)':<26} | {base_summary['normal_wait']['mean']:>6.2f} +/- {base_summary['normal_wait']['std']:<6.2f} | {rule_summary['normal_wait']['mean']:>6.2f} +/- {rule_summary['normal_wait']['std']:<6.2f} | {rl_summary['normal_wait']['mean']:>6.2f} +/- {rl_summary['normal_wait']['std']:<6.2f}")
    print(f"{'Requests Served':<26} | {base_summary['served']['mean']:>6.2f} +/- {base_summary['served']['std']:<6.2f} | {rule_summary['served']['mean']:>6.2f} +/- {rule_summary['served']['std']:<6.2f} | {rl_summary['served']['mean']:>6.2f} +/- {rl_summary['served']['std']:<6.2f}")
    print(f"{'Requests Dropped':<26} | {base_summary['dropped']['mean']:>6.2f} +/- {base_summary['dropped']['std']:<6.2f} | {rule_summary['dropped']['mean']:>6.2f} +/- {rule_summary['dropped']['std']:<6.2f} | {rl_summary['dropped']['mean']:>6.2f} +/- {rl_summary['dropped']['std']:<6.2f}")
    print(f"{'Total Distance':<26} | {base_summary['total_dist']['mean']:>6.2f} +/- {base_summary['total_dist']['std']:<6.2f} | {rule_summary['total_dist']['mean']:>6.2f} +/- {rule_summary['total_dist']['std']:<6.2f} | {rl_summary['total_dist']['mean']:>6.2f} +/- {rl_summary['total_dist']['std']:<6.2f}")
    print("=" * 90 + "\n")

    if save_files:
        # Save results.json
        out_name = "results.json" if version == "v2" else "results_v1.json"
        with open(out_name, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {out_name}")

        # Read training curve if exists
        training_data = []
        if os.path.exists("training_curve.json"):
            with open("training_curve.json", "r") as f:
                training_data = json.load(f)

        # Also load v1 results if exists to include in replay_data for dashboard comparison
        v1_results = None
        if os.path.exists("results_v1.json"):
            with open("results_v1.json", "r") as f:
                v1_results = json.load(f)

        os.makedirs("dashboard", exist_ok=True)
        replay_payload = {
            "results": results,
            "results_v1": v1_results,
            "training": training_data,
            "baseline_frames": base_replay_frames,
            "rl_frames": rl_replay_frames
        }

        replay_js_content = f"const REPLAY_DATA = {json.dumps(replay_payload)};\n"
        with open(os.path.join("dashboard", "replay_data.js"), "w") as f:
            f.write(replay_js_content)
        print("Saved dashboard/replay_data.js")

    return results

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v2"
    evaluate(version=version)
