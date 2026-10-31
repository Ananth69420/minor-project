"""
Master Execution Pipeline for Reinforcement Learning Autonomous Fleet Dispatch.
Executes end-to-end:
1. Q-Learning Training (10,000 episodes with reproducible seeds)
2. Evaluation & Statistical Benchmark (50 paired episodes vs Nearest Baseline)
3. Matplotlib Figure Generation (4 publication-quality PNG charts in figures/)
4. Convert simulator results, replay frames, and Q-table to the dashboard data format
"""

import time
import config
from train import run_training
from evaluate import run_benchmark
from generate_plots import generate_all_plots
from dashboard_adapter import write_replay_data


def main():
    total_start = time.time()
    print("=" * 78)
    print("   AUTONOMOUS FLEET MANAGEMENT RL DISPATCH - MASTER RUNNER")
    print("   Minor Project Submission - Pure Python / NumPy / Matplotlib")
    print("=" * 78)

    # 1. Train Q-Learning Agent
    print("\n[STEP 1/4] Running Tabular Q-Learning Training...")
    train_res = run_training(
        num_episodes=config.NUM_TRAIN_EPISODES,
        save_q_table_path="q_table.npy",
        save_history_path="training_history.json",
        seed=config.RANDOM_SEED,
    )

    # 2. Benchmark against Nearest Baseline
    print("\n[STEP 2/4] Running Statistical Benchmark (50 Paired Episodes)...")
    eval_res = run_benchmark(
        num_episodes=config.NUM_EVAL_EPISODES,
        q_table_path="q_table.npy",
        save_results_path="simulation_results.json",
        seed=12345,
    )

    # 3. Generate Publication Plots
    print("\n[STEP 3/4] Generating Publication Figures in 'figures/'...")
    generate_all_plots()

    # 4. Adapt fresh evaluation data for the dashboard
    print("\n[STEP 4/4] Updating Dashboard Data...")
    write_replay_data()

    total_time = time.time() - total_start
    print("\n" + "=" * 78)
    print(f"ALL STEPS COMPLETED IN {total_time:.2f} SECONDS")
    print("Generated outputs:")
    print("  - Trained Model:        q_table.npy")
    print("  - Training History:     training_history.json")
    print("  - Evaluation Results:   simulation_results.json")
    print("  - Publication Figures:  figures/ (*.png)")
    print("  - Dashboard:             index.html + replay_data.js")
    print("=" * 78)


if __name__ == "__main__":
    main()
