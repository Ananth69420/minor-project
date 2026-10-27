import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.city import CityEnv
from agents.baseline import BaselineDispatcher, ForecastRuleDispatcher

def run_validation():
    val_seeds = list(range(1000, 1200)) # 200 validation seeds
    print(f"Evaluating Baseline and Forecast-Aware Rule on {len(val_seeds)} validation seeds (1000-1199)...")

    base_dispatcher = BaselineDispatcher()
    rule_dispatcher = ForecastRuleDispatcher()
    env = CityEnv(seed=42)

    base_metrics = []
    rule_metrics = []

    for ep, seed in enumerate(val_seeds):
        surge_quad = ep % 4

        # 1. Baseline
        env.reset(seed=seed, surge_quad=surge_quad)
        for step in range(300):
            for v in env.vehicles:
                if v["status"] == "idle":
                    a = base_dispatcher.select_action(env, v["id"])
                    if a is not None:
                        env.dispatch(v["id"], a)
            env.step()
        base_metrics.append(env.get_metrics())

        # 2. Forecast-Aware Rule
        env.reset(seed=seed, surge_quad=surge_quad)
        for step in range(300):
            for v in env.vehicles:
                if v["status"] == "idle":
                    a = rule_dispatcher.select_action(env, v["id"])
                    if a is not None:
                        env.dispatch(v["id"], a)
            env.step()
        rule_metrics.append(env.get_metrics())

    def summarize(metrics, name):
        avg_wait = np.mean([m["avg_wait"] for m in metrics])
        surge_wait = np.mean([m["surge_wait"] for m in metrics])
        normal_wait = np.mean([m["normal_wait"] for m in metrics])
        served = np.mean([m["served"] for m in metrics])
        dropped = np.mean([m["dropped"] for m in metrics])
        dist = np.mean([m["total_dist"] for m in metrics])
        print(f"\n--- {name} (Validation 1000-1199) ---")
        print(f"Average Wait Time (steps) : {avg_wait:.2f}")
        print(f"Surge Phase Wait (steps)  : {surge_wait:.2f}")
        print(f"Normal Phase Wait (steps) : {normal_wait:.2f}")
        print(f"Requests Served           : {served:.2f}")
        print(f"Requests Dropped          : {dropped:.2f}")
        print(f"Total Distance            : {dist:.2f}")
        return avg_wait, surge_wait, served, dropped

    summarize(base_metrics, "Baseline (Nearest)")
    summarize(rule_metrics, "Forecast-Aware Rule")

if __name__ == "__main__":
    run_validation()
