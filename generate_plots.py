"""
Matplotlib Plot Generation for Reinforcement Learning Fleet Dispatch Project.
Produces publication-quality figures for the minor project submission and report.
Only depends on matplotlib and numpy (standard library + config).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import config


# Apply modern styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
})

COLORS = {
    "baseline": "#ef4444",    # Vibrant red/coral
    "qlearning": "#10b981",   # Emerald green
    "primary": "#3b82f6",     # Royal blue
    "amber": "#f59e0b",       # Amber
    "purple": "#8b5cf6",      # Purple
    "surge_bg": "#fef3c7",    # Pale yellow surge background
}


def ensure_fig_dir(dir_path: str = "figures") -> None:
    os.makedirs(dir_path, exist_ok=True)


def plot_training_curves(
    history_path: str = "training_history.json",
    save_path: str = "figures/training_performance.png",
) -> None:
    """Figure 1: Training convergence curves (Reward, Wait Time, Served vs Dropped, Epsilon)."""
    if not os.path.exists(history_path):
        print(f"Warning: '{history_path}' not found, skipping training plot.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        h = json.load(f)

    episodes = np.array(h["episode"])
    rewards = np.array(h["reward"])
    waits = np.array(h["avg_wait"])
    served = np.array(h["served"])
    dropped = np.array(h["dropped"])
    epsilons = np.array(h["epsilon"])

    # Rolling window smoothing helper
    def moving_average(a, n=50):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    smooth_ep = episodes[49:]
    smooth_rew = moving_average(rewards, 50)
    smooth_wait = moving_average(waits, 50)
    smooth_served = moving_average(served, 50)
    smooth_dropped = moving_average(dropped, 50)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Tabular Q-Learning Training Dynamics across 2,500 Episodes", fontweight="bold", fontsize=15)

    # Subplot 1: Cumulative Reward
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, color="#93c5fd", alpha=0.35, label="Raw Episode Reward")
    ax1.plot(smooth_ep, smooth_rew, color="#1d4ed8", label="50-Ep Moving Average")
    ax1.set_title("Cumulative Fleet Reward ($R$)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(loc="lower right")

    # Subplot 2: Average Wait Time
    ax2 = axes[0, 1]
    ax2.plot(episodes, waits, color="#fca5a5", alpha=0.35, label="Raw Avg Wait")
    ax2.plot(smooth_ep, smooth_wait, color="#b91c1c", label="50-Ep Moving Average")
    ax2.set_title("Average Passenger Wait Time (steps)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Wait Time (steps)")
    ax2.legend(loc="upper right")

    # Subplot 3: Requests Served vs Dropped
    ax3 = axes[1, 0]
    ax3.plot(smooth_ep, smooth_served, color="#059669", label="Served (50-MA)")
    ax3.plot(smooth_ep, smooth_dropped, color="#dc2626", label="Dropped / Cancellations (50-MA)")
    ax3.set_title("Trip Fulfillment vs. Cancellations")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Request Count per Episode")
    ax3.legend(loc="center right")

    # Subplot 4: Exploration Decay (Epsilon)
    ax4 = axes[1, 1]
    ax4.plot(episodes, epsilons, color="#7c3aed", label=r"Exploration $\epsilon$")
    ax4.set_title(r"$\epsilon$-Greedy Policy Exploration Decay")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel(r"$\epsilon$")
    ax4.set_ylim(-0.02, 1.05)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_benchmark_comparison(
    results_path: str = "simulation_results.json",
    save_path: str = "figures/benchmark_comparison.png",
) -> None:
    """Figure 2: Comprehensive benchmark metrics bar chart with error bars."""
    if not os.path.exists(results_path):
        print(f"Warning: '{results_path}' not found, skipping benchmark comparison plot.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    b_sum = data["summary"]["baseline"]
    q_sum = data["summary"]["qlearning"]

    metrics = [
        ("Avg Wait Time (s)", "avg_wait_time", "steps (lower is better)", False),
        ("Service Rate (%)", "service_rate", "% (higher is better)", True),
        ("Requests Dropped", "requests_dropped", "count (lower is better)", False),
        ("Fleet Reward", "total_reward", "reward units (higher is better)", True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    fig.suptitle(
        f"Empirical Benchmark Comparison: Nearest Baseline vs. Tabular Q-Learning ({data['config']['num_eval_episodes']} Paired Episodes)",
        fontweight="bold",
        fontsize=14,
    )

    for i, (title, key, ylabel, higher_better) in enumerate(metrics):
        ax = axes[i]
        b_mean = b_sum[key]["mean"]
        b_std = b_sum[key]["std"]
        q_mean = q_sum[key]["mean"]
        q_std = q_sum[key]["std"]

        x = np.arange(2)
        bars = ax.bar(
            x,
            [b_mean, q_mean],
            yerr=[b_std, q_std],
            capsize=6,
            color=[COLORS["baseline"], COLORS["qlearning"]],
            alpha=0.88,
            edgecolor=["#991b1b", "#065f46"],
            linewidth=1.2,
            width=0.55,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(["Nearest Baseline", "RL (Q-Learning)"], fontweight="medium")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=11.5)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=11,
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_surge_queue_dynamics(
    results_path: str = "simulation_results.json",
    save_path: str = "figures/surge_queue_dynamics.png",
) -> None:
    """Figure 3: Step-by-step queue length evolution showing surge absorption."""
    if not os.path.exists(results_path):
        print(f"Warning: '{results_path}' not found, skipping surge queue plot.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data["time_series"]["steps"]
    base_queue = data["time_series"]["baseline_avg_queue"]
    rl_queue = data["time_series"]["rl_avg_queue"]

    plt.figure(figsize=(12, 5.5))
    ax = plt.subplot(111)

    # Highlight surge phase [100, 200]
    ax.axvspan(100, 200, color="#fef08a", alpha=0.35, label="Surge Phase (3x Demand in Hotspot)")

    # Plot queue lines
    ax.plot(steps, base_queue, color=COLORS["baseline"], label="Nearest Baseline Pending Queue", linewidth=2.2)
    ax.plot(steps, rl_queue, color=COLORS["qlearning"], label="RL (Q-Learning) Pending Queue", linewidth=2.2)

    ax.set_title(
        "Step-by-Step Pending Request Queue Dynamics Across Episode Phases",
        fontweight="bold",
        fontsize=13,
    )
    ax.set_xlabel("Simulation Step (1 step = 1 grid move)")
    ax.set_ylabel("Average Pending Queue Length (passengers)")
    ax.set_xlim(0, 300)
    ax.legend(loc="upper left", frameon=True)

    # Add phase text annotations
    ax.text(50, ax.get_ylim()[1] * 0.9, "Normal Demand (p=0.2)", ha="center", fontsize=10, style="italic")
    ax.text(150, ax.get_ylim()[1] * 0.9, "SURGE HOTSPOT (p=0.6)", ha="center", fontsize=10, fontweight="bold", color="#854d0e")
    ax.text(250, ax.get_ylim()[1] * 0.9, "Normal Demand (p=0.2)", ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_policy_heatmap(
    q_table_path: str = "q_table.npy",
    save_path: str = "figures/policy_heatmap.png",
) -> None:
    """Figure 4: Learned Q-table heatmap for representative operational states."""
    if not os.path.exists(q_table_path):
        print(f"Warning: '{q_table_path}' not found, skipping policy heatmap.")
        return

    q_table = np.load(q_table_path)

    # Select representative states for presentation to college examiners:
    # We will pick 8 key states:
    # 1. State 0: Vehicle in Q0, all demand 0 (Quiet normal)
    # 2. State 1: Vehicle in Q0, demand in Q3 is 1..2
    # 3. State 2: Vehicle in Q0, demand in Q3 is 3+ (Surge opposite corner)
    # 4. State 6: Vehicle in Q0, demand in Q2 is 3+ (Surge south)
    # 5. State 18: Vehicle in Q0, demand in Q1 is 3+ (Surge east)
    # 6. State 54: Vehicle in Q0, demand in Q0 is 3+ (Surge local)
    # 7. State 162: Vehicle in Q2, demand in Q2 is 3+ (Vehicle inside surge hotspot)
    # 8. State 243: Vehicle in Q3, demand in Q2 is 3+ (Vehicle adjacent to surge)
    state_indices = [0, 2, 6, 18, 54, 87, 162, 243]
    state_labels = [
        "V@Q0 | All 0s (Quiet)",
        "V@Q0 | Surge in Q3 (3+)",
        "V@Q0 | Surge in Q2 (3+)",
        "V@Q0 | Surge in Q1 (3+)",
        "V@Q0 | Surge in Q0 (3+ Local)",
        "V@Q1 | Surge in Q2 (3+)",
        "V@Q2 | Surge in Q2 (3+ Inside)",
        "V@Q3 | Surge in Q2 (3+ Adjacent)",
    ]

    action_labels = [
        "Act 0\nNearest",
        "Act 1\nLongest-Wait",
        "Act 2\nRepo Q0 (NW)",
        "Act 3\nRepo Q1 (NE)",
        "Act 4\nRepo Q2 (SW)",
        "Act 5\nRepo Q3 (SE)",
    ]

    matrix = q_table[state_indices]

    plt.figure(figsize=(11, 7))
    plt.imshow(matrix, cmap="RdYlGn", aspect="auto")
    plt.colorbar(label="Learned Q-Value ($Q(s, a)$)")

    plt.xticks(np.arange(len(action_labels)), action_labels, fontsize=10)
    plt.yticks(np.arange(len(state_labels)), state_labels, fontsize=10)
    plt.title("Learned Q-Table Policy Matrix Across Surge & Normal Scenarios", fontweight="bold", fontsize=13)

    # Annotate numeric values inside each cell
    for r in range(len(state_indices)):
        for c in range(len(action_labels)):
            val = matrix[r, c]
            text_color = "black" if -5 < val < 15 else "white"
            plt.text(c, r, f"{val:.1f}", ha="center", va="center", color=text_color, fontweight="semibold", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots() -> None:
    ensure_fig_dir("figures")
    plot_training_curves()
    plot_benchmark_comparison()
    plot_surge_queue_dynamics()
    plot_policy_heatmap()


if __name__ == "__main__":
    generate_all_plots()
