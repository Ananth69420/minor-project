import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.city import CityEnv
from agents.qlearning import QLearningAgent

def run_exp(num_episodes=8000, lr=0.06, gamma=0.95, eps_decay=0.9995):
    print(f"\n--- Running Exp 03: Q-Learning 576 states, {num_episodes} eps, lr={lr}, gamma={gamma}, eps_decay={eps_decay} ---")
    np.random.seed(42)
    env = CityEnv(seed=42)
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

            # Attributed reward
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

    for ep, seed in enumerate(val_seeds):
        surge_quad = ep % 4
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

    print(f"Validation Results (seeds 1000-1199):")
    print(f"Average Wait (steps) : {avg_wait:.2f}")
    print(f"Surge Wait (steps)   : {surge_wait:.2f}")
    print(f"Normal Wait (steps)  : {normal_wait:.2f}")
    print(f"Served               : {served:.2f}")
    print(f"Dropped              : {dropped:.2f}")
    print(f"Total Distance       : {dist:.2f}")

    return avg_wait, surge_wait, served, dropped

if __name__ == "__main__":
    run_exp(num_episodes=8000, lr=0.06, gamma=0.95, eps_decay=0.9995)
