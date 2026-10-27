"""
Tabular Q-Learning Agent for Fleet Dispatch and Proactive Repositioning.
All autonomous vehicles share one common Q-table (Parameter Sharing).
Semi-Markov Decision Process (SMDP) Q-learning for variable-duration macro actions.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import config


class QLearningAgent:
    """
    Tabular Q-Learning Agent with Fleet Parameter Sharing.
    - State space (324): vehicle quadrant (4) x demand buckets (81)
    - Action space (6):
        0 = Serve Nearest Pending Request
        1 = Serve Longest-Waiting Pending Request
        2 = Reposition to Quadrant 0 (NW) Center (2, 2)
        3 = Reposition to Quadrant 1 (NE) Center (7, 2)
        4 = Reposition to Quadrant 2 (SW) Center (2, 7)
        5 = Reposition to Quadrant 3 (SE) Center (7, 7)
    """

    def __init__(
        self,
        num_states: int = config.NUM_STATES,
        num_actions: int = config.NUM_ACTIONS,
        learning_rate: float = config.LEARNING_RATE,
        discount_factor: float = config.DISCOUNT_FACTOR,
        epsilon_start: float = config.EPSILON_START,
        epsilon_min: float = config.EPSILON_MIN,
        epsilon_decay: float = config.EPSILON_DECAY,
        seed: Optional[int] = config.RANDOM_SEED,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Tabular Q-Table: shape (324, 6)
        # Small random initialization around 0 helps break initial symmetry
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        # Tracking active in-flight commitments per vehicle:
        # vehicle_id -> {"state": int, "action": int, "discounted_reward": float, "duration": int}
        self.active_transitions: Dict[int, Dict[str, Any]] = {}

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        If evaluate is True, acts purely greedily (argmax Q).
        """
        if not evaluate and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, self.num_actions))

        # Greedy choice with random tie-breaking
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))
        return int(self.rng.choice(best_actions))

    def record_decision(self, vehicle_id: int, state: int, action: int) -> None:
        """Records the start of a vehicle's dispatch decision."""
        self.active_transitions[vehicle_id] = {
            "state": state,
            "action": action,
            "discounted_reward": 0.0,
            "duration": 0,
        }

    def accumulate_step_reward(self, vehicle_id: int, reward: float) -> None:
        """Accumulates discounted step reward for an in-flight vehicle commitment."""
        if vehicle_id in self.active_transitions:
            trans = self.active_transitions[vehicle_id]
            discount = self.gamma ** trans["duration"]
            trans["discounted_reward"] += discount * reward
            trans["duration"] += 1

    def update_on_completion(self, vehicle_id: int, next_state: int, done: bool = False) -> Optional[float]:
        """
        Updates the shared Q-table when a vehicle completes its mission or becomes idle.
        Returns the TD error (for monitoring), or None if no active transition.
        """
        if vehicle_id not in self.active_transitions:
            return None

        trans = self.active_transitions.pop(vehicle_id)
        s = trans["state"]
        a = trans["action"]
        g = trans["discounted_reward"]
        dt = trans["duration"]

        current_q = self.q_table[s, a]

        if done:
            target = g
        else:
            best_next_q = np.max(self.q_table[next_state])
            target = g + (self.gamma ** dt) * best_next_q

        td_error = target - current_q
        self.q_table[s, a] += self.lr * td_error
        return float(td_error)

    def terminal_cleanup(self, next_states: Dict[int, int]) -> None:
        """Finalizes any remaining in-flight vehicle updates at the end of an episode."""
        for v_id in list(self.active_transitions.keys()):
            next_state = next_states.get(v_id, 0)
            self.update_on_completion(vehicle_id=v_id, next_state=next_state, done=True)

    def decay_epsilon(self) -> None:
        """Decays exploration rate after each training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Saves the Q-table to a .npy file."""
        np.save(filepath, self.q_table)

    def load(self, filepath: str) -> None:
        """Loads the Q-table from a .npy file."""
        self.q_table = np.load(filepath)
