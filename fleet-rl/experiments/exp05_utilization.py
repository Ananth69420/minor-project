import sys
import os
import json
import numpy as np

# Ensure fleet-rl root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.city import CityEnv
from agents.baseline import BaselineDispatcher

def diagnose_utilization():
    val_seeds = list(range(1000, 1200)) # 200 validation seeds strictly
    baseline = BaselineDispatcher()

    # Phases:
    # 0: Normal (0-79, 80 steps)
    # 1: Pre-surge (80-99, 20 steps)
    # 2: Surge (100-199, 100 steps)
    # 3: Post-surge (200-299, 100 steps)
    phase_lengths = {0: 80, 1: 20, 2: 100, 3: 100}
    phase_names = {0: "Normal (0-79)", 1: "Pre-surge (80-99)", 2: "Surge (100-199)", 3: "Post-surge (200-299)"}

    # Tracking accumulators per phase across episodes
    phase_data = {
        p: {
            "idle_vehicle_steps": [],
            "total_vehicle_steps": [],
            "pending_unassigned_per_step": [],
            "pending_total_per_step": [],
            "dispatch_pickup_dist": [],
            "pickup_travel_time": [],
            "arrivals": [],
            "completions": [],
            "cancellations": [],
            "idle_vehs_when_reqs_waiting": [],
            "all_busy_steps": []
        }
        for p in range(4)
    }

    for ep_idx, seed in enumerate(val_seeds):
        surge_quad = ep_idx % 4
        env = CityEnv(seed=seed)
        env.reset(seed=seed, surge_quad=surge_quad)

        # Episode level phase counters
        ep_phase_stats = {
            p: {
                "idle_veh_steps": 0,
                "unassigned_sum": 0,
                "total_pending_sum": 0,
                "dispatch_dists": [],
                "pickup_times": [],
                "arrivals": 0,
                "completions": 0,
                "cancellations": 0,
                "idle_when_waiting": [],
                "all_busy_count": 0
            }
            for p in range(4)
        }

        # Track active pickups: v_id -> (start_step, pickup_dist)
        active_pickups = {}

        for step in range(300):
            p = env.get_phase()
            st = ep_phase_stats[p]

            # 1. Check idle vehicles and dispatch baseline
            idle_count_before = sum(1 for v in env.vehicles if v["status"] == "idle")
            unassigned_before = [r for r in env.requests if r["status"] == "pending"]

            # Metric 6: Number of available idle vehicles when requests are waiting
            if len(unassigned_before) > 0:
                st["idle_when_waiting"].append(idle_count_before)

            # Perform dispatch
            for v in env.vehicles:
                if v["status"] == "idle":
                    a = baseline.select_action(env, v["id"])
                    if a is not None:
                        # Capture dispatch distance to nearest request
                        unassigned = [r for r in env.requests if r["status"] == "pending"]
                        if unassigned:
                            nearest = min(unassigned, key=lambda r: abs(v["x"] - r["px"]) + abs(v["y"] - r["py"]))
                            dist = abs(v["x"] - nearest["px"]) + abs(v["y"] - nearest["py"])
                            st["dispatch_dists"].append(dist)
                            active_pickups[v["id"]] = {"start_step": step, "initial_dist": dist}

                        env.dispatch(v["id"], a)

            # Record idle vehicle-steps (strictly status == 'idle')
            idle_now = sum(1 for v in env.vehicles if v["status"] == "idle")
            st["idle_veh_steps"] += idle_now

            # Metric 7: All vehicles busy (0 idle vehicles)
            if idle_now == 0:
                st["all_busy_count"] += 1

            # Count unassigned and total pending requests at this step
            unassigned_count = sum(1 for r in env.requests if r["status"] == "pending")
            total_pending_count = sum(1 for r in env.requests if r["status"] in ("pending", "assigned"))
            st["unassigned_sum"] += unassigned_count
            st["total_pending_sum"] += total_pending_count

            # Count request arrivals before env.step()
            reqs_before = len(env.requests)

            # Execute environment step
            step_reward, done, info = env.step()

            # Completions and cancellations from info
            st["completions"] += info["dropoffs"]
            st["cancellations"] += info["drops"]

            # Arrivals during step
            reqs_after = len(env.requests)
            st["arrivals"] += (reqs_after - reqs_before)

            # Check if any vehicle completed pickup (status changed from pickup to trip)
            for v in env.vehicles:
                if v["id"] in active_pickups and v["status"] == "trip":
                    pk_info = active_pickups.pop(v["id"])
                    time_spent = step - pk_info["start_step"]
                    st["pickup_times"].append(time_spent)
                elif v["id"] in active_pickups and v["status"] == "idle":
                    # Pickup cancelled or dropped
                    active_pickups.pop(v["id"], None)

        # Aggregate episode stats into phase_data
        for p in range(4):
            length = phase_lengths[p]
            st = ep_phase_stats[p]
            tot_veh_steps = env.num_vehicles * length

            phase_data[p]["idle_vehicle_steps"].append(st["idle_veh_steps"])
            phase_data[p]["total_vehicle_steps"].append(tot_veh_steps)
            phase_data[p]["pending_unassigned_per_step"].append(st["unassigned_sum"] / length)
            phase_data[p]["pending_total_per_step"].append(st["total_pending_sum"] / length)
            if st["dispatch_dists"]:
                phase_data[p]["dispatch_pickup_dist"].append(np.mean(st["dispatch_dists"]))
            if st["pickup_times"]:
                phase_data[p]["pickup_travel_time"].append(np.mean(st["pickup_times"]))
            phase_data[p]["arrivals"].append(st["arrivals"])
            phase_data[p]["completions"].append(st["completions"])
            phase_data[p]["cancellations"].append(st["cancellations"])
            if st["idle_when_waiting"]:
                phase_data[p]["idle_vehs_when_reqs_waiting"].append(np.mean(st["idle_when_waiting"]))
            else:
                phase_data[p]["idle_vehs_when_reqs_waiting"].append(0.0)
            phase_data[p]["all_busy_steps"].append(st["all_busy_count"] / length)

    # Compile phase-by-phase summary
    summary = {}
    print("\n" + "=" * 105)
    print("V1 FLEET UTILIZATION DIAGNOSIS (Baseline Dispatcher on Validation Seeds 1000-1199, N=200)")
    print("=" * 105)
    header = f"{'Phase':<22} | {'Idle Frac':<10} | {'Unassigned':<11} | {'All Busy %':<11} | {'Arr Rate':<9} | {'Serv Rate':<10} | {'Drops':<6} | {'Disp Dist':<10}"
    print(header)
    print("-" * 105)

    for p in range(4):
        pd = phase_data[p]
        length = phase_lengths[p]

        idle_frac = np.mean(pd["idle_vehicle_steps"]) / (4 * length)
        unassigned_avg = np.mean(pd["pending_unassigned_per_step"])
        tot_pending_avg = np.mean(pd["pending_total_per_step"])
        all_busy_frac = np.mean(pd["all_busy_steps"])
        arr_rate = np.mean(pd["arrivals"]) / length
        serv_rate = np.mean(pd["completions"]) / length
        drops_avg = np.mean(pd["cancellations"])
        disp_dist_avg = np.mean(pd["dispatch_pickup_dist"]) if pd["dispatch_pickup_dist"] else 0.0
        pk_time_avg = np.mean(pd["pickup_travel_time"]) if pd["pickup_travel_time"] else 0.0
        idle_when_waiting_avg = np.mean(pd["idle_vehs_when_reqs_waiting"])

        summary[phase_names[p]] = {
            "phase_index": p,
            "steps": length,
            "idle_fraction": round(float(idle_frac), 4),
            "idle_vehicle_steps_mean": round(float(np.mean(pd["idle_vehicle_steps"])), 2),
            "total_vehicle_steps": 4 * length,
            "pending_unassigned_per_step": round(float(unassigned_avg), 3),
            "pending_total_per_step": round(float(tot_pending_avg), 3),
            "all_vehicles_busy_proportion": round(float(all_busy_frac), 4),
            "arrival_rate_per_step": round(float(arr_rate), 3),
            "service_completion_rate_per_step": round(float(serv_rate), 3),
            "cancellations_mean": round(float(drops_avg), 2),
            "dispatch_pickup_distance_mean": round(float(disp_dist_avg), 2),
            "pickup_travel_time_mean": round(float(pk_time_avg), 2),
            "idle_vehicles_when_requests_waiting": round(float(idle_when_waiting_avg), 3)
        }

        row = (
            f"{phase_names[p]:<22} | "
            f"{idle_frac*100:>6.1f}%   | "
            f"{unassigned_avg:>6.2f} reqs | "
            f"{all_busy_frac*100:>6.1f}%   | "
            f"{arr_rate:>6.3f}/s  | "
            f"{serv_rate:>6.3f}/s   | "
            f"{drops_avg:>5.2f} | "
            f"{disp_dist_avg:>6.2f} cells"
        )
        print(row)

    print("=" * 105 + "\n")

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp05_utilization_results.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved machine-readable diagnosis to {out_file}")

if __name__ == "__main__":
    diagnose_utilization()
