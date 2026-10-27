"""
Urban Mobility Grid Environment for Autonomous Fleet Management.
Pure Python and NumPy implementation. No Gym/Gymnasium dependencies.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import config


class Request:
    """Represents a single ride request in the city."""
    def __init__(
        self,
        request_id: int,
        pickup: Tuple[int, int],
        dropoff: Tuple[int, int],
        spawn_step: int,
    ):
        self.id: int = request_id
        self.pickup: Tuple[int, int] = pickup
        self.dropoff: Tuple[int, int] = dropoff
        self.spawn_step: int = spawn_step
        self.pickup_step: Optional[int] = None
        self.dropoff_step: Optional[int] = None
        self.wait_time: int = 0
        # Statuses: PENDING, ASSIGNED, IN_TRANSIT, COMPLETED, DROPPED
        self.status: str = "PENDING"
        self.assigned_vehicle_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pickup": list(self.pickup),
            "dropoff": list(self.dropoff),
            "spawn_step": self.spawn_step,
            "pickup_step": self.pickup_step,
            "dropoff_step": self.dropoff_step,
            "wait_time": self.wait_time,
            "status": self.status,
            "assigned_vehicle_id": self.assigned_vehicle_id,
        }


class Vehicle:
    """Represents an autonomous vehicle in the fleet."""
    def __init__(self, vehicle_id: int, start_pos: Tuple[int, int]):
        self.id: int = vehicle_id
        self.pos: List[int] = list(start_pos)
        # Statuses: IDLE, DISPATCHED_PICKUP, CARRYING_TRIP, REPOSITIONING
        self.status: str = "IDLE"
        self.current_request: Optional[Request] = None
        self.target_pos: Optional[List[int]] = None
        self.total_distance: int = 0
        self.idle_distance: int = 0
        self.trips_served: int = 0

    @property
    def quadrant(self) -> int:
        """Quadrant 0: NW (0..4, 0..4), 1: NE (5..9, 0..4), 2: SW (0..4, 5..9), 3: SE (5..9, 5..9)"""
        qx = 0 if self.pos[0] < 5 else 1
        qy = 0 if self.pos[1] < 5 else 1
        return qy * 2 + qx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pos": list(self.pos),
            "status": self.status,
            "target_pos": list(self.target_pos) if self.target_pos is not None else None,
            "current_request_id": self.current_request.id if self.current_request else None,
            "total_distance": self.total_distance,
            "idle_distance": self.idle_distance,
            "trips_served": self.trips_served,
            "quadrant": self.quadrant,
        }


class CityFleetEnv:
    """
    2D Grid City Environment with a fleet of autonomous vehicles.
    Simulates demand surges, vehicle routing, pickups, dropoffs,
    cancellations, and tracks complete fleet performance metrics.
    """

    def __init__(
        self,
        grid_size: int = config.GRID_SIZE,
        num_vehicles: int = config.NUM_VEHICLES,
        max_steps: int = config.EPISODE_STEPS,
        max_wait_time: int = config.MAX_WAIT_TIME,
        seed: Optional[int] = config.RANDOM_SEED,
    ):
        self.grid_size = grid_size
        self.num_vehicles = num_vehicles
        self.max_steps = max_steps
        self.max_wait_time = max_wait_time
        self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.surge_quadrant = 0
        self.vehicles: List[Vehicle] = []
        self.all_requests: List[Request] = []
        self.next_request_id = 0

        # Step accounting
        self.step_dropoffs = 0
        self.step_drops = 0
        self.step_movement = 0
        self.cumulative_reward = 0.0

    @staticmethod
    def get_quadrant(x: int, y: int) -> int:
        """Determines which of the 4 quadrants a coordinate falls into."""
        qx = 0 if x < 5 else 1
        qy = 0 if y < 5 else 1
        return qy * 2 + qx

    def reset(
        self,
        seed: Optional[int] = None,
        surge_quadrant: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Resets the environment for a new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.all_requests = []
        self.next_request_id = 0
        self.step_dropoffs = 0
        self.step_drops = 0
        self.step_movement = 0
        self.cumulative_reward = 0.0

        # Select surge hotspot quadrant (0..3)
        if surge_quadrant is not None:
            self.surge_quadrant = surge_quadrant
        else:
            self.surge_quadrant = int(self.rng.integers(0, 4))

        # Initialize vehicles at quadrant centers
        self.vehicles = []
        for i in range(self.num_vehicles):
            start_pos = config.QUADRANT_CENTERS[i % len(config.QUADRANT_CENTERS)]
            self.vehicles.append(Vehicle(vehicle_id=i, start_pos=start_pos))

        return self.get_state_summary()

    def get_unassigned_pending_requests(self) -> List[Request]:
        """Returns all ride requests that are waiting and not yet assigned to any vehicle."""
        return [r for r in self.all_requests if r.status == "PENDING"]

    def encode_state(self, vehicle_id: int) -> int:
        """
        Encodes the state for an idle vehicle into an integer in [0, 323].
        State = vehicle's quadrant (4) x pending-request count bucket per quadrant (0, 1-2, 3+ each -> 81).
        """
        vehicle = self.vehicles[vehicle_id]
        v_quad = vehicle.quadrant

        # Count unassigned pending requests in each quadrant
        pending_counts = [0, 0, 0, 0]
        for req in self.get_unassigned_pending_requests():
            quad = self.get_quadrant(req.pickup[0], req.pickup[1])
            pending_counts[quad] += 1

        # Bucket counts: 0 -> 0, 1..2 -> 1, 3+ -> 2
        buckets = []
        for c in pending_counts:
            if c == 0:
                buckets.append(0)
            elif c <= 2:
                buckets.append(1)
            else:
                buckets.append(2)

        # 3^4 = 81 bucket states
        bucket_index = (
            buckets[0] * 27 +
            buckets[1] * 9 +
            buckets[2] * 3 +
            buckets[3] * 1
        )

        state_index = v_quad * 81 + bucket_index
        return state_index

    def dispatch_vehicle(self, vehicle_id: int, action: int) -> bool:
        """
        Executes a dispatch action for an IDLE vehicle.
        Actions:
          0 = Serve nearest pending request
          1 = Serve longest-waiting pending request
          2 = Reposition to Quadrant 0 Center (2, 2)
          3 = Reposition to Quadrant 1 Center (7, 2)
          4 = Reposition to Quadrant 2 Center (2, 7)
          5 = Reposition to Quadrant 3 Center (7, 7)
        Returns True if a dispatch/reposition mission was initiated, False otherwise.
        """
        vehicle = self.vehicles[vehicle_id]
        if vehicle.status != "IDLE":
            return False

        unassigned = self.get_unassigned_pending_requests()

        if action == 0:  # Nearest request
            if not unassigned:
                return False
            # Find nearest request by Manhattan distance
            best_req = min(
                unassigned,
                key=lambda r: (
                    abs(vehicle.pos[0] - r.pickup[0]) + abs(vehicle.pos[1] - r.pickup[1]),
                    -r.wait_time  # secondary tie-breaker: longer wait time
                )
            )
            best_req.status = "ASSIGNED"
            best_req.assigned_vehicle_id = vehicle.id
            vehicle.status = "DISPATCHED_PICKUP"
            vehicle.current_request = best_req
            vehicle.target_pos = list(best_req.pickup)
            return True

        elif action == 1:  # Longest waiting request
            if not unassigned:
                return False
            # Find request with longest wait time
            best_req = max(
                unassigned,
                key=lambda r: (
                    r.wait_time,
                    -(abs(vehicle.pos[0] - r.pickup[0]) + abs(vehicle.pos[1] - r.pickup[1]))  # secondary: closer
                )
            )
            best_req.status = "ASSIGNED"
            best_req.assigned_vehicle_id = vehicle.id
            vehicle.status = "DISPATCHED_PICKUP"
            vehicle.current_request = best_req
            vehicle.target_pos = list(best_req.pickup)
            return True

        elif 2 <= action <= 5:  # Reposition to quadrant center
            target_center = config.QUADRANT_CENTERS[action - 2]
            if vehicle.pos[0] == target_center[0] and vehicle.pos[1] == target_center[1]:
                # Already at target center, remain idle
                return False
            vehicle.status = "REPOSITIONING"
            vehicle.current_request = None
            vehicle.target_pos = list(target_center)
            return True

        return False

    def _spawn_requests(self) -> None:
        """Spawns ride requests based on current simulation phase and surge condition."""
        in_surge = (config.SURGE_START_STEP <= self.current_step < config.SURGE_END_STEP)
        demand_prob = config.SURGE_DEMAND_PROB if in_surge else config.NORMAL_DEMAND_PROB

        # Determine how many requests spawn this step (probabilistic)
        if self.rng.random() < demand_prob:
            # Generate pickup coordinate
            if in_surge and (self.rng.random() < config.SURGE_HOTSPOT_CONCENTRATION):
                # Spawn inside the surge hotspot quadrant
                sq = self.surge_quadrant
                qx_min = 0 if sq in (0, 2) else 5
                qy_min = 0 if sq in (0, 1) else 5
                px = int(self.rng.integers(qx_min, qx_min + 5))
                py = int(self.rng.integers(qy_min, qy_min + 5))
            else:
                # Uniform across city
                px = int(self.rng.integers(0, self.grid_size))
                py = int(self.rng.integers(0, self.grid_size))

            # Generate dropoff coordinate (must be different from pickup)
            dx = int(self.rng.integers(0, self.grid_size))
            dy = int(self.rng.integers(0, self.grid_size))
            while dx == px and dy == py:
                dx = int(self.rng.integers(0, self.grid_size))
                dy = int(self.rng.integers(0, self.grid_size))

            new_req = Request(
                request_id=self.next_request_id,
                pickup=(px, py),
                dropoff=(dx, dy),
                spawn_step=self.current_step,
            )
            self.next_request_id += 1
            self.all_requests.append(new_req)

    def _move_vehicles(self, vehicle_events: Dict[int, Dict[str, bool]]) -> None:
        """Moves each non-idle vehicle 1 grid cell along its Manhattan shortest path."""
        for vehicle in self.vehicles:
            if vehicle.target_pos is None or vehicle.status == "IDLE":
                continue

            tx, ty = vehicle.target_pos
            vx, vy = vehicle.pos

            # Manhattan move: advance in x first, then y
            moved = False
            if vx < tx:
                vehicle.pos[0] += 1
                moved = True
            elif vx > tx:
                vehicle.pos[0] -= 1
                moved = True
            elif vy < ty:
                vehicle.pos[1] += 1
                moved = True
            elif vy > ty:
                vehicle.pos[1] -= 1
                moved = True

            if moved:
                vehicle.total_distance += 1
                self.step_movement += 1
                vehicle_events[vehicle.id]["moved"] = True
                if vehicle.status in ("DISPATCHED_PICKUP", "REPOSITIONING"):
                    vehicle.idle_distance += 1

    def _process_vehicle_arrivals(self, vehicle_events: Dict[int, Dict[str, bool]]) -> None:
        """Processes auto-pickup, auto-dropoff, and reposition arrival upon reaching target cell."""
        for vehicle in self.vehicles:
            if vehicle.target_pos is None:
                continue

            # Check if vehicle reached its target
            if vehicle.pos[0] == vehicle.target_pos[0] and vehicle.pos[1] == vehicle.target_pos[1]:
                if vehicle.status == "DISPATCHED_PICKUP":
                    # Auto-pickup
                    req = vehicle.current_request
                    if req is not None and req.status == "ASSIGNED":
                        req.status = "IN_TRANSIT"
                        req.pickup_step = self.current_step
                        vehicle.status = "CARRYING_TRIP"
                        vehicle.target_pos = list(req.dropoff)

                elif vehicle.status == "CARRYING_TRIP":
                    # Auto-dropoff
                    req = vehicle.current_request
                    if req is not None:
                        req.status = "COMPLETED"
                        req.dropoff_step = self.current_step
                        vehicle.trips_served += 1
                        self.step_dropoffs += 1
                        vehicle_events[vehicle.id]["dropoff"] = True

                    vehicle.status = "IDLE"
                    vehicle.current_request = None
                    vehicle.target_pos = None

                elif vehicle.status == "REPOSITIONING":
                    # Arrived at quadrant center
                    vehicle.status = "IDLE"
                    vehicle.target_pos = None

    def _update_requests_and_timeouts(self) -> None:
        """Updates wait times for waiting passengers and drops requests that exceeded max wait time."""
        for req in self.all_requests:
            if req.status in ("PENDING", "ASSIGNED"):
                req.wait_time += 1

                # Check cancellation timeout
                if req.wait_time >= self.max_wait_time:
                    req.status = "DROPPED"
                    self.step_drops += 1

                    # If a vehicle was on the way to pick up this passenger, release it
                    if req.assigned_vehicle_id is not None:
                        v = self.vehicles[req.assigned_vehicle_id]
                        if v.current_request and v.current_request.id == req.id:
                            v.status = "IDLE"
                            v.current_request = None
                            v.target_pos = None

    def step(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Advances the simulation by 1 time step.
        Returns: (step_reward, done, info)
        """
        self.step_dropoffs = 0
        self.step_drops = 0
        self.step_movement = 0

        vehicle_events = {
            v.id: {"dropoff": False, "moved": False} for v in self.vehicles
        }

        # 1. Spawn new requests
        self._spawn_requests()

        # 2. Move vehicles
        self._move_vehicles(vehicle_events)

        # 3. Process arrivals (auto-pickup, auto-dropoff, repositioning finish)
        self._process_vehicle_arrivals(vehicle_events)

        # 4. Update wait times and handle drop timeouts
        self._update_requests_and_timeouts()

        # 5. Compute reward
        # Pending request count (both unassigned and assigned still waiting at curb)
        pending_count = sum(1 for r in self.all_requests if r.status in ("PENDING", "ASSIGNED"))

        step_reward = (
            self.step_dropoffs * config.REWARD_DROPOFF +
            pending_count * config.PENALTY_WAIT_STEP +
            self.step_movement * config.PENALTY_DISTANCE +
            self.step_drops * config.PENALTY_DROPPED
        )

        self.cumulative_reward += step_reward
        self.current_step += 1
        done = self.current_step >= self.max_steps

        info = {
            "step": self.current_step,
            "dropoffs": self.step_dropoffs,
            "drops": self.step_drops,
            "movement": self.step_movement,
            "pending_count": pending_count,
            "step_reward": step_reward,
            "cumulative_reward": self.cumulative_reward,
            "vehicle_events": vehicle_events,
        }

        return step_reward, done, info

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates final comprehensive episode metrics."""
        completed = [r for r in self.all_requests if r.status == "COMPLETED"]
        dropped = [r for r in self.all_requests if r.status == "DROPPED"]
        total_spawned = len(self.all_requests)

        avg_wait_completed = (
            float(np.mean([r.wait_time for r in completed])) if completed else 0.0
        )
        avg_wait_all = (
            float(np.mean([r.wait_time for r in self.all_requests])) if self.all_requests else 0.0
        )

        total_distance = sum(v.total_distance for v in self.vehicles)
        idle_distance = sum(v.idle_distance for v in self.vehicles)
        empty_ratio = (idle_distance / total_distance) if total_distance > 0 else 0.0
        service_rate = (len(completed) / total_spawned * 100.0) if total_spawned > 0 else 0.0

        # Surge phase specific metrics
        surge_spawned = [r for r in self.all_requests if config.SURGE_START_STEP <= r.spawn_step < config.SURGE_END_STEP]
        surge_completed = [r for r in surge_spawned if r.status == "COMPLETED"]
        surge_dropped = [r for r in surge_spawned if r.status == "DROPPED"]
        surge_service_rate = (len(surge_completed) / len(surge_spawned) * 100.0) if surge_spawned else 0.0
        surge_avg_wait = float(np.mean([r.wait_time for r in surge_completed])) if surge_completed else 0.0

        return {
            "total_requests": total_spawned,
            "requests_served": len(completed),
            "requests_dropped": len(dropped),
            "service_rate": round(service_rate, 2),
            "avg_wait_time": round(avg_wait_completed, 2),
            "avg_wait_all": round(avg_wait_all, 2),
            "total_distance": total_distance,
            "idle_distance": idle_distance,
            "empty_ratio": round(empty_ratio, 3),
            "total_reward": round(self.cumulative_reward, 2),
            "surge_quadrant": self.surge_quadrant,
            "surge_requests": len(surge_spawned),
            "surge_served": len(surge_completed),
            "surge_dropped": len(surge_dropped),
            "surge_service_rate": round(surge_service_rate, 2),
            "surge_avg_wait": round(surge_avg_wait, 2),
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Returns a snapshot of the current environment state (used for visualization and debugging)."""
        return {
            "step": self.current_step,
            "surge_quadrant": self.surge_quadrant,
            "is_surge": config.SURGE_START_STEP <= self.current_step < config.SURGE_END_STEP,
            "vehicles": [v.to_dict() for v in self.vehicles],
            "requests": [r.to_dict() for r in self.all_requests if r.status in ("PENDING", "ASSIGNED", "IN_TRANSIT")],
            "cumulative_reward": round(self.cumulative_reward, 2),
        }
