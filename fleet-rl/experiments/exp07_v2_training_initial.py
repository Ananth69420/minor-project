import sys
import os
import time
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.city import CityEnv
from agents.qlearning import QLearningAgent

def run_exp(num_episodes=8000, lr=0.05, gamma=0.96, eps_decay=0.9996):
    print(f"\n--- Running Exp 07: V2 Q-Learning 576 states, {num_episodes} eps, lr={lr}, gamma={gamma}, eps_decay={eps_decay} ---")
    np.random.seed(42)
    env = CityEnv(grid_size=16, num_vehicles=8, p_normal=0.25, p_surge=0.55, seed=42)
    agent = QLearningAgent(num_states=576, num_actions=6, lr=lr, gamma=gamma, eps=1.0, eps_min=0.01, eps_decay=eps_decay)

    train_start = time.time()
    for ep in range(num_episodes):
        train_seed = 2000 + (ep % 3000)
        surge_quad = ep % 4
        env.reset(seed=train_seed, surge_quad=surge_quad)

        while True:
            for v in env.vehicles:
                if v["status"] == "idle":
                    s = env.encode_state(v["id"])
                    val_acts = env.get_valid_actions(v["id"])
                    agent.finish_decision(v["id"], next_state=s, next_valid_actions=val_acts, done=False)
                    a = agent.select_action(s, val_acts, explore=True)
                    dispatched = env.dispatch(v["id"], a)
                    if dispatched:
                        agent.start_decision(v["id"], s, a)

            step_reward, done, info = env.step()

            # Shared queue and cancellation penalty, plus vehicle dropoff and move costs
            shared = (-0.15 * info["pending"] - 25.0 * info["drops"]) / env.num_vehicles
            for v in env.vehicles:
                r = shared
                if info["dropoffs_by_v"][v["id"]]:
                    r += 20.0
                if info["moved_by_v"][v["id"]]:
                    r -= 0.05
                agent.add_step_reward(v["id"], r)

            if done:
                terminal_states = env.get_state_for_all()
                for v in env.vehicles:
                    agent.finish_decision(v["id"], next_state=terminal_states[v["id"]], done=True)
                agent.decay_eps()
                break

    train_time = time.time() - train_start
    print(f"Training finished in {train_time:.1f}s.")

    # Evaluate on validation seeds 1000-1199
    val_seeds = list(range(1000, 1200))
    rl_metrics = []

    for ep_idx, seed in enumerate(val_seeds):
        surge_quad = ep_idx % 4
        env.reset(seed=seed, surge_quad=surge_quad)
        for _ in range(300):
            for v in env.vehicles:
                if v["status"] == "idle":
                    s = env.encode_state(v["id"])
                    val_acts = env.get_valid_actions(v["id"])
                    a = agent.select_action(s, val_acts, explore=False)
                    env.dispatch(v["id"], a)
            env.step()
        rl_metrics.append(env.get_metrics())

    avg_wait = np.mean([m["avg_wait"] for m in rl_metrics])
    surge_wait = np.mean([m["surge_wait"] for m in rl_metrics])
    normal_wait = np.mean([m["normal_wait"] for m in rl_metrics])
    served = np.mean([m["served"] for m in rl_metrics])
    dropped = np.mean([m["dropped"] for m in rl_metrics])
    dist = np.mean([m["total_dist"] for m in rl_metrics])

    print(f"Validation Results on V2 (seeds 1000-1199, N=200):")
    print(f"Average Wait (steps) : {avg_wait:.2f} (Baseline: 14.38, Rule: 13.85)")
    print(f"Surge Wait (steps)   : {surge_wait:.2f} (Baseline: 17.29, Rule: 16.37)")
    print(f"Normal Wait (steps)  : {normal_wait:.2f}")
    print(f"Served               : {served:.2f}")
    print(f"Dropped              : {dropped:.2f}")
    print(f"Total Distance       : {dist:.2f}")

    # Inspect Q-table learned actions in Phase 1 warning states
    # Phase 1 is state where phase=1, hotspot != v_quad, counts=0
    # Quad 0, Phase 1, Hotspot 3, c_own=0, c_hot=0 -> state index:
    # (((0 * 4 + 1) * 4 + 3) * 3 + 0) * 3 + 0 = 63
    p1_actions = []
    for vq in range(4):
        for hq in range(4):
            if vq != hq:
                s = (((vq * 4 + 1) * 4 + hq) * 3 + 0) * 3 + 0
                q_row = agent.q_table[s]
                best_a = int(np.argmax(q_row))
                p1_actions.append((vq, hq, best_a, [round(float(x), 1) for x in q_row]))
    print("\nSample Learned Phase 1 Idle Actions (v_quad -> hotspot):")
    for vq, hq, ba, qvals in p1_actions[:4]:
        print(f"  V_Quad={vq} -> Hotspot={hq} | Best Action={ba} (Target Quad={ba-2 if ba>=2 else 'Serve'}) | Q={qvals}")

    return {
        "train_time": train_time,
        "avg_wait": avg_wait,
        "surge_wait": surge_wait,
        "normal_wait": normal_wait,
        "served": served,
        "dropped": dropped,
        "total_dist": dist
    }

if __name__ == "__main__":
    run_exp(num_episodes=8000, lr=0.05, gamma=0.96, eps_decay=0.9996)
