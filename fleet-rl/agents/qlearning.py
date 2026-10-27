import numpy as np

class QLearningAgent:
    def __init__(self, num_states=576, num_actions=6, lr=0.08, gamma=0.92, eps=1.0, eps_min=0.01, eps_decay=0.999):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # Shared Q-table for the entire fleet
        self.q_table = np.zeros((num_states, num_actions))
        self.active_decisions = {}

    def select_action(self, state, valid_actions, explore=True):
        if not valid_actions:
            return 0

        if explore and (np.random.random() < self.eps):
            return int(np.random.choice(valid_actions))

        # Greedy choice among valid actions only
        q_vals = [self.q_table[state, a] for a in valid_actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(valid_actions, q_vals) if q == max_q]
        return int(np.random.choice(best_actions))

    def start_decision(self, v_id, state, action):
        self.active_decisions[v_id] = {
            "state": state,
            "action": action,
            "reward": 0.0,
            "duration": 0
        }

    def add_step_reward(self, v_id, r):
        if v_id in self.active_decisions:
            d = self.active_decisions[v_id]
            d["reward"] += (self.gamma ** d["duration"]) * r
            d["duration"] += 1

    def finish_decision(self, v_id, next_state, next_valid_actions=None, done=False):
        if v_id not in self.active_decisions:
            return

        d = self.active_decisions.pop(v_id)
        s = d["state"]
        a = d["action"]
        g = d["reward"]
        dt = d["duration"]

        if done:
            target = g
        else:
            if next_valid_actions and len(next_valid_actions) > 0:
                max_next = max(self.q_table[next_state, act] for act in next_valid_actions)
            else:
                max_next = np.max(self.q_table[next_state])
            target = g + (self.gamma ** dt) * max_next

        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)
