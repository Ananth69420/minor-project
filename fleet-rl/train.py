import os
import json
import time
import numpy as np
from env.city import CityEnv
from agents.qlearning import QLearningAgent
from config import ENV_V2

# Configuration parameters for V2 environment
config = {
    "num_episodes": 10000,
    "learning_rate": 0.05,
    "discount_factor": 0.96,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9996,
    "seed": 42
}

def train():
    np.random.seed(config["seed"])
    env = CityEnv(
        grid_size=ENV_V2["grid_size"],
        num_vehicles=ENV_V2["num_vehicles"],
        max_steps=ENV_V2["max_steps"],
        max_wait=ENV_V2["max_wait"],
        p_normal=ENV_V2["p_normal"],
        p_surge=ENV_V2["p_surge"],
        seed=config["seed"]
    )
    agent = QLearningAgent(
        num_states=576,
        num_actions=6,
        lr=config["learning_rate"],
        gamma=config["discount_factor"],
        eps=config["epsilon_start"],
        eps_min=config["epsilon_min"],
        eps_decay=config["epsilon_decay"]
    )

    history = {
        "episode": [],
        "reward": [],
        "avg_wait": [],
        "served": [],
        "dropped": []
    }

    start_time = time.time()
    num_episodes = config["num_episodes"]

    for ep in range(num_episodes):
        surge_quad = ep % 4
        # Training seeds strictly in range [2000, 4999] (disjoint from val 1000-1199 and test 5000-5049)
        train_seed = 2000 + (ep % 3000)
        env.reset(seed=train_seed, surge_quad=surge_quad)
        ep_reward = 0.0

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
            ep_reward += step_reward

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

        m = env.get_metrics()
        history["episode"].append(ep + 1)
        history["reward"].append(round(ep_reward, 2))
        history["avg_wait"].append(m["avg_wait"])
        history["served"].append(m["served"])
        history["dropped"].append(m["dropped"])

        if (ep + 1) % 1000 == 0 or (ep + 1) == num_episodes:
            recent_reward = np.mean(history["reward"][-100:])
            recent_wait = np.mean(history["avg_wait"][-100:])
            recent_served = np.mean(history["served"][-100:])
            recent_dropped = np.mean(history["dropped"][-100:])
            print(
                f"Episode {ep + 1:5d}/{num_episodes} | "
                f"Reward: {recent_reward:7.1f} | "
                f"Wait: {recent_wait:5.1f} steps | "
                f"Served: {recent_served:4.1f} | "
                f"Dropped: {recent_dropped:4.1f} | "
                f"Eps: {agent.eps:.3f}"
            )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds.")

    # Save Q-table and training curve
    agent.save("q_table.npy")
    with open("training_curve.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Saved q_table.npy and training_curve.json")

if __name__ == "__main__":
    train()
