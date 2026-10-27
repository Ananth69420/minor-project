import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.city import CityEnv
from agents.baseline import BaselineDispatcher, ForecastRuleDispatcher

def test_candidate(name, grid_size, num_vehicles, p_normal, p_surge, val_seeds):
    baseline = BaselineDispatcher()
    rule = ForecastRuleDispatcher()

    base_metrics = []
    rule_metrics = []

    for ep_idx, seed in enumerate(val_seeds):
        surge_quad = ep_idx % 4

        # Baseline
        env_base = CityEnv(grid_size=grid_size, num_vehicles=num_vehicles, p_normal=p_normal, p_surge=p_surge, seed=seed)
        env_base.reset(seed=seed, surge_quad=surge_quad)
        for _ in range(300):
            for v in env_base.vehicles:
                if v["status"] == "idle":
                    a = baseline.select_action(env_base, v["id"])
                    if a is not None:
                        env_base.dispatch(v["id"], a)
            env_base.step()
        base_metrics.append(env_base.get_metrics())

        # Forecast-Aware Rule
        env_rule = CityEnv(grid_size=grid_size, num_vehicles=num_vehicles, p_normal=p_normal, p_surge=p_surge, seed=seed)
        env_rule.reset(seed=seed, surge_quad=surge_quad)
        for _ in range(300):
            for v in env_rule.vehicles:
                if v["status"] == "idle":
                    a = rule.select_action(env_rule, v["id"])
                    if a is not None:
                        env_rule.dispatch(v["id"], a)
            env_rule.step()
        rule_metrics.append(env_rule.get_metrics())

    b_wait = np.mean([m["avg_wait"] for m in base_metrics])
    b_surge = np.mean([m["surge_wait"] for m in base_metrics])
    b_serv = np.mean([m["served"] for m in base_metrics])
    b_drop = np.mean([m["dropped"] for m in base_metrics])
    b_dist = np.mean([m["total_dist"] for m in base_metrics])

    r_wait = np.mean([m["avg_wait"] for m in rule_metrics])
    r_surge = np.mean([m["surge_wait"] for m in rule_metrics])
    r_serv = np.mean([m["served"] for m in rule_metrics])
    r_drop = np.mean([m["dropped"] for m in rule_metrics])
    r_dist = np.mean([m["total_dist"] for m in rule_metrics])

    diff_surge = r_surge - b_surge
    diff_wait = r_wait - b_wait

    print(f"\nConfiguration: {name} (Grid={grid_size}x{grid_size}, Fleet={num_vehicles}, P_norm={p_normal}, P_surge={p_surge})")
    print(f"  Baseline     : Avg Wait={b_wait:5.2f} | Surge Wait={b_surge:5.2f} | Served={b_serv:5.1f} | Dropped={b_drop:4.1f} | Dist={b_dist:6.1f}")
    print(f"  Forecast Rule: Avg Wait={r_wait:5.2f} | Surge Wait={r_surge:5.2f} | Served={r_serv:5.1f} | Dropped={r_drop:4.1f} | Dist={r_dist:6.1f}")
    print(f"  Delta (Rule - Base): Surge Wait={diff_surge:+5.2f} steps | Avg Wait={diff_wait:+5.2f} steps")

    return {
        "name": name,
        "grid_size": grid_size,
        "num_vehicles": num_vehicles,
        "p_normal": p_normal,
        "p_surge": p_surge,
        "baseline": {"avg_wait": b_wait, "surge_wait": b_surge, "served": b_serv, "dropped": b_drop, "dist": b_dist},
        "forecast_rule": {"avg_wait": r_wait, "surge_wait": r_surge, "served": r_serv, "dropped": r_drop, "dist": r_dist},
        "diff_surge_wait": diff_surge,
        "diff_avg_wait": diff_wait
    }

def run_all_candidates():
    val_seeds = list(range(1000, 1200)) # 200 validation seeds strictly
    print("=" * 80)
    print("V2 ENVIRONMENT CANDIDATE SCREENING (Seeds 1000-1199, N=200)")
    print("=" * 80)

    candidates = [
        ("Candidate A (16x16, 6 veh, p=0.20/0.45)", 16, 6, 0.20, 0.45),
        ("Candidate B (16x16, 8 veh, p=0.25/0.55)", 16, 8, 0.25, 0.55),
        ("Candidate C (20x20, 8 veh, p=0.25/0.55)", 20, 8, 0.25, 0.55),
    ]

    results = []
    for name, g, v, pn, ps in candidates:
        res = test_candidate(name, g, v, pn, ps, val_seeds)
        results.append(res)

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp06_candidates_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved candidates screening results to {out_file}")

if __name__ == "__main__":
    run_all_candidates()
