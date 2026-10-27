
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CityFleetEnv(gym.Env):
    """
    A small city with one autonomous vehicle and at most
    one active pickup-and-delivery request.

    Actions:
        0 = North
        1 = South
        2 = East
        3 = West
        4 = Wait
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size=5,
        max_steps=200,
        request_prob=0.15,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.max_steps = max_steps
        self.request_prob = request_prob

        # Four movements plus wait.
        self.action_space = spaces.Discrete(5)

        # A pickup/destination value of n_cells means "none".
        self.none = self.n_cells

        # State = (vehicle, pickup, destination, status)
        # status: 0 = idle, 1 = going to pickup,
        #         2 = carrying passenger
        self.n_states = (
            self.n_cells
            * (self.n_cells + 1)
            * (self.n_cells + 1)
            * 3
        )

        self.observation_space = spaces.Discrete(self.n_states)

        self.vehicle = 0
        self.pickup = self.none
        self.destination = self.none
        self.status = 0

        self.steps = 0
        self.total_distance = 0
        self.total_wait_time = 0
        self.completed_trips = 0
        self.current_wait = 0

    def _encode_state(self):
        """Convert the current situation into one integer."""
        state = self.status
        state = state * (self.n_cells + 1) + self.pickup
        state = state * (self.n_cells + 1) + self.destination
        state = state * self.n_cells + self.vehicle

        return int(state)

    def _position(self, cell):
        """Convert a cell number into (x, y) coordinates."""
        return (
            cell % self.grid_size,
            cell // self.grid_size,
        )

    def _spawn_request(self):
        """Create a random request if the vehicle is idle."""
        self.pickup = int(
            self.np_random.integers(self.n_cells)
        )

        self.destination = int(
            self.np_random.integers(self.n_cells)
        )

        # Avoid a zero-distance passenger trip.
        while self.destination == self.pickup:
            self.destination = int(
                self.np_random.integers(self.n_cells)
            )

        self.status = 1
        self.current_wait = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.vehicle = int(
            self.np_random.integers(self.n_cells)
        )

        self.pickup = self.none
        self.destination = self.none
        self.status = 0

        self.steps = 0
        self.total_distance = 0
        self.total_wait_time = 0
        self.completed_trips = 0
        self.current_wait = 0

        # Start with a request so the first episode
        # has an actual pickup-and-delivery task.
        self._spawn_request()

        return self._encode_state(), self._get_info()

    def _get_info(self):
        return {
            "vehicle_position": self._position(self.vehicle),
            "pickup_position": (
                None if self.status == 0
                else self._position(self.pickup)
            ),
            "destination_position": (
                None if self.status == 0
                else self._position(self.destination)
            ),
            "status": self.status,
            "distance": self.total_distance,
            "wait_time": self.total_wait_time,
            "completed_trips": self.completed_trips,
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        reward = -1.0  # Every step costs time.

        x, y = self._position(self.vehicle)

        # Calculate the proposed movement.
        if action == 0:       # North
            y -= 1
        elif action == 1:     # South
            y += 1
        elif action == 2:     # East
            x += 1
        elif action == 3:     # West
            x -= 1
        # Action 4 means wait.

        # Prevent the vehicle from leaving the grid.
        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))

        new_vehicle = y * self.grid_size + x

        if new_vehicle != self.vehicle:
            self.total_distance += 1

        self.vehicle = new_vehicle

        if self.status == 1:
            # Passenger is waiting for pickup.
            self.current_wait += 1
            self.total_wait_time += 1

            if self.vehicle == self.pickup:
                # Passenger boards the vehicle.
                self.status = 2

        elif self.status == 2:
            # Passenger is onboard.
            if self.vehicle == self.destination:
                reward += 20.0
                self.completed_trips += 1

                # The completed request is removed.
                self.pickup = self.none
                self.destination = self.none
                self.status = 0
                self.current_wait = 0

        # If idle, a new request may arrive.
        if (
            self.status == 0
            and self.np_random.random() < self.request_prob
        ):
            self._spawn_request()

        terminated = False
        truncated = self.steps >= self.max_steps

        info = self._get_info()

        return (
            self._encode_state(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        """Return a simple text representation of the city."""
        grid = [
            ["."] * self.grid_size
            for _ in range(self.grid_size)
        ]

        if self.status == 1:
            px, py = self._position(self.pickup)
            grid[py][px] = "P"

        if self.status in (1, 2):
            dx, dy = self._position(self.destination)
            grid[dy][dx] = "D"

        vx, vy = self._position(self.vehicle)
        grid[vy][vx] = "V"

        return "\n".join(" ".join(row) for row in grid)
