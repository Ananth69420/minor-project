import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.city import CityEnv
from agents.qlearning import QLearningAgent

def inspect_policy(num_episodes=4000, lr=0.08, gamma=0.92, eps_decay=0.999):
    np.random.seed(42)
    env = CityEnv(seed=42)
    agent = QLearningAgent(num_states=576, num_actions=6, lr=lr, gamma=gamma, eps=1.0, eps_min=0.01, eps_decay=eps_decay)

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

    # Let's inspect phase 1 decisions (pre-surge, 80-99):
    # What does the agent do when idle with 0 requests in own quadrant and hotspot is H?
    print("\n--- Inspecting Pre-Surge Phase (Phase 1) Policy for Empty Own Quadrant (b_own=0, b_hot=0) ---")
    for vq in range(4):
        for h in range(4):
            # phase=1, b_own=0, b_hot=0
            s = (((vq * 4 + 1) * 4 + h) * 3 + 0) * 3 + 0
            # If no requests, valid actions are 2, 3, 4, 5
            valid = [2, 3, 4, 5]
            q_vals = {a: round(agent.q_table[s, a], 2) for a in valid}
            best_a = valid[np.argmax([agent.q_table[s, a] for a in valid])]
            target_q = best_a - 2
            print(f"Vehicle in Q{vq}, Hotspot Q{h}: Q-vals={q_vals} -> Best Action={best_a} (Move to Q{target_q})")

if __name__ == "__main__":
    inspect_policy()
