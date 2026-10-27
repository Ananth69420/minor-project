"""
Training Pipeline for Tabular Q-Learning Autonomous Fleet Dispatcher.
Fully reproducible with fixed random seeds.
Runs in seconds on a standard laptop CPU.
"""

import os
import time
import json
from typing import Dict, List, Any
import numpy as np
import config
from env.city import CityFleetEnv
from agents.qlearning import QLearningAgent


def run_training(
    num_episodes: int = config.NUM_TRAIN_EPISODES,
    save_q_table_path: str = "q_table.npy",
    save_history_path: str = "training_history.json",
    seed: int = config.RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Executes reproducible tabular Q-learning training for autonomous fleet management.
    """
    print("=" * 70)
    print("STARTING Q-LEARNING TRAINING PIPELINE")
    print(f"Total Episodes: {num_episodes} | Episode Length: {config.EPISODE_STEPS} steps")
    print(f"Fleet Size: {config.NUM_VEHICLES} vehicles | Grid: {config.GRID_SIZE}x{config.GRID_SIZE}")
    print(f"State Space: {config.NUM_STATES} | Action Space: {config.NUM_ACTIONS}")
    print(f"Initial Epsilon: {config.EPSILON_START} -> Min Epsilon: {config.EPSILON_MIN}")
    print(f"Random Seed: {seed}")
    print("=" * 70)

    start_time = time.time()
    env = CityFleetEnv(seed=seed)
    agent = QLearningAgent(seed=seed)

    history: Dict[str, List[float]] = {
        "episode": [],
        "reward": [],
        "avg_wait": [],
        "served": [],
        "dropped": [],
        "service_rate": [],
        "epsilon": [],
    }

    # Deterministic episode seeds
    rng = np.random.default_rng(seed)
    episode_seeds = [int(rng.integers(0, 1000000)) for _ in range(num_episodes)]

    for ep in range(num_episodes):
        ep_seed = episode_seeds[ep]
        env.reset(seed=ep_seed)

        while True:
            # 1. Idle vehicles make decisions
            for v in env.vehicles:
                if v.status == "IDLE":
                    # Update previous decision transition if any
                    s = env.encode_state(v.id)
                    agent.update_on_completion(v.id, next_state=s, done=False)

                    # Select action using epsilon-greedy
                    a = agent.select_action(s, evaluate=False)
                    env.dispatch_vehicle(v.id, a)
                    agent.record_decision(v.id, s, a)

            # 2. Advance environment by 1 step
            step_reward, done, info = env.step()

            # 3. Credit assignment: share queue wait & cancellation costs, credit vehicle-specific actions
            shared_penalty = (
                info["pending_count"] * config.PENALTY_WAIT_STEP +
                info["drops"] * config.PENALTY_DROPPED
            ) / env.num_vehicles

            for v in env.vehicles:
                ev = info["vehicle_events"][v.id]
                v_reward = shared_penalty
                if ev["dropoff"]:
                    v_reward += config.REWARD_DROPOFF
                if ev["moved"]:
                    v_reward += config.PENALTY_DISTANCE

                agent.accumulate_step_reward(v.id, v_reward)

            # 4. Check episode termination
            if done:
                terminal_states = {v.id: env.encode_state(v.id) for v in env.vehicles}
                agent.terminal_cleanup(terminal_states)
                agent.decay_epsilon()
                break

        # Log episode metrics
        metrics = env.get_metrics()
        history["episode"].append(ep + 1)
        history["reward"].append(metrics["total_reward"])
        history["avg_wait"].append(metrics["avg_wait_time"])
        history["served"].append(metrics["requests_served"])
        history["dropped"].append(metrics["requests_dropped"])
        history["service_rate"].append(metrics["service_rate"])
        history["epsilon"].append(round(agent.epsilon, 4))

        # Periodic logging
        if (ep + 1) % 250 == 0 or (ep + 1) == num_episodes:
            window = slice(max(0, ep - 99), ep + 1)
            mean_rew = np.mean(history["reward"][window])
            mean_wait = np.mean(history["avg_wait"][window])
            mean_served = np.mean(history["served"][window])
            mean_dropped = np.mean(history["dropped"][window])
            mean_rate = np.mean(history["service_rate"][window])
            elapsed = time.time() - start_time
            print(
                f"Episode {ep + 1:4d}/{num_episodes} | Time: {elapsed:5.1f}s | "
                f"Eps: {agent.epsilon:.3f} | Rew(100ep): {mean_rew:7.1f} | "
                f"Wait: {mean_wait:5.1f}s | Served: {mean_served:4.1f} | "
                f"Dropped: {mean_dropped:4.1f} | Rate: {mean_rate:5.1f}%"
            )

    total_training_time = time.time() - start_time
    print("=" * 70)
    print(f"TRAINING COMPLETED IN {total_training_time:.2f} SECONDS")
    non_zero = np.count_nonzero(agent.q_table)
    print(f"Trained Q-Table: {non_zero}/{config.NUM_STATES * config.NUM_ACTIONS} state-action pairs explored")
    print(f"Saving Q-table to '{save_q_table_path}'...")
    agent.save(save_q_table_path)

    # Save training history
    with open(save_history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to '{save_history_path}'.")
    print("=" * 70)

    return {
        "training_time": round(total_training_time, 2),
        "total_episodes": num_episodes,
        "final_epsilon": round(agent.epsilon, 4),
        "q_table_non_zero": int(non_zero),
        "history": history,
    }


if __name__ == "__main__":
    run_training()
