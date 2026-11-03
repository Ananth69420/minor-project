import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 1. Create the environment
env = gym.make("Taxi-v4")

# 2. Initialize the Q-table
# 500 states, 6 actions per state
q_table = np.zeros((env.observation_space.n,
                    env.action_space.n))

# 3. Hyperparameters
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 1.0        # Initial exploration probability
epsilon_min = 0.05
epsilon_decay = 0.995

episodes = 10_000

episode_rewards = []
moving_avg_rewards = []

# 4. Training loop
for episode in range(episodes):

    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # 5. Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # 6. Take action and observe the result
        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        # 7. Compute the best future Q-value
        best_next_q = np.max(q_table[next_state])

        # 8. Q-learning update
        # Do not bootstrap beyond the end of an episode
        if terminated:
            target = reward
        else:
            target = reward + gamma * best_next_q

        q_table[state, action] += alpha * (
            target - q_table[state, action]
        )

        state = next_state
        total_reward += reward

    # 9. Record episode performance
    episode_rewards.append(total_reward)

    # 10. Reduce exploration over time
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        moving_avg_rewards.append(avg_reward)

        print(
            f"Episode {episode + 1:5d} | "
            f"Avg reward (last 100): {avg_reward:7.2f} | "
            f"Epsilon: {epsilon:.3f}"
        )

env.close()

# 11. Plot reward per episode
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.2, label="Episode reward")

# Smooth the noisy rewards for easier interpretation
window = 100
if len(episode_rewards) >= window:
    smoothed = np.convolve(
        episode_rewards,
        np.ones(window) / window,
        mode="valid"
    )
    plt.plot(
        np.arange(window - 1, len(episode_rewards)),
        smoothed,
        label="100-episode moving average"
    )

plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Tabular Q-learning on Taxi-v4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("taxi_training.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved graph to taxi_training.png")
# Save the learned table
np.save("taxi_q_table.npy", q_table)
print("Saved Q-table to taxi_q_table.npy")
