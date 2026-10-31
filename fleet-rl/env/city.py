import numpy as np

class CityEnv:
    def __init__(self, grid_size=10, num_vehicles=4, max_steps=300, max_wait=60, seed=42, p_normal=0.20, p_surge=0.60):
        self.grid_size = grid_size
        self.num_vehicles = num_vehicles
        self.max_steps = max_steps
        self.max_wait = max_wait
        self.seed = seed
        self.p_normal = p_normal
        self.p_surge = p_surge
        self.rng = np.random.default_rng(seed)

        # Dynamic quadrant centers
        q1 = grid_size // 4
        q2 = (3 * grid_size) // 4
        self.quad_centers = [(q1, q1), (q2, q1), (q1, q2), (q2, q2)]

        self.step_num = 0
        self.surge_quad = 0
        self.vehicles = []
        self.requests = []
        self.req_id_counter = 0

    def get_quadrant(self, x, y):
        mid = self.grid_size // 2
        qx = 0 if x < mid else 1
        qy = 0 if y < mid else 1
        return qy * 2 + qx

    def get_phase(self):
        # 0 = normal (0-79), 1 = pre-surge warning (80-99), 2 = surge (100-199), 3 = post-surge (200-299)
        if self.step_num < 80:
            return 0
        elif self.step_num < 100:
            return 1
        elif self.step_num < 200:
            return 2
        else:
            return 3

    def reset(self, seed=None, surge_quad=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_num = 0
        self.req_id_counter = 0
        self.requests = []
        self.surge_quad = surge_quad if surge_quad is not None else int(self.rng.integers(0, 4))

        self.vehicles = []
        for i in range(self.num_vehicles):
            cx, cy = self.quad_centers[i % 4]
            self.vehicles.append({
                "id": i,
                "x": cx,
                "y": cy,
                "status": "idle",       # idle, pickup, trip, repo
                "target_x": None,
                "target_y": None,
                "req_id": None,
                "total_dist": 0,
                "idle_dist": 0,
                "trips": 0
            })

        return self.get_state_for_all()

    def get_pending_counts(self):
        counts = [0, 0, 0, 0]
        for r in self.requests:
            if r["status"] == "pending":
                q = self.get_quadrant(r["px"], r["py"])
                counts[q] += 1
        return counts

    def encode_state(self, v_id):
        v = self.vehicles[v_id]
        v_quad = self.get_quadrant(v["x"], v["y"])
        phase = self.get_phase()

        # Forecast hotspot available in phases 1 and 2; placeholder v_quad in phases 0 and 3
        if phase in (1, 2):
            hotspot = self.surge_quad
        else:
            hotspot = v_quad

        counts = self.get_pending_counts()

        c_own = counts[v_quad]
        b_own = 0 if c_own == 0 else (1 if c_own <= 2 else 2)

        c_hot = counts[hotspot]
        b_hot = 0 if c_hot == 0 else (1 if c_hot <= 2 else 2)

        # 4 (v_quad) x 4 (phase) x 4 (hotspot) x 3 (b_own) x 3 (b_hot) = 576 states
        return (((v_quad * 4 + phase) * 4 + hotspot) * 3 + b_own) * 3 + b_hot

    def get_state_for_all(self):
        return {v["id"]: self.encode_state(v["id"]) for v in self.vehicles}

    def get_valid_actions(self, v_id):
        v = self.vehicles[v_id]
        if v["status"] != "idle":
            return []

        unassigned = [r for r in self.requests if r["status"] == "pending"]
        has_unassigned = len(unassigned) > 0
        v_quad = self.get_quadrant(v["x"], v["y"])

        valid = []
        if has_unassigned:
            valid.extend([0, 1])

        for a in range(2, 6):
            target_quad = a - 2
            # Available when no request is available, OR when chosen quad differs from current quad
            if (not has_unassigned) or (target_quad != v_quad):
                valid.append(a)

        return valid

    def dispatch(self, v_id, action):
        v = self.vehicles[v_id]
        if v["status"] != "idle":
            return False

        unassigned = [r for r in self.requests if r["status"] == "pending"]

        if action == 0:
            # Serve nearest pending request
            if not unassigned:
                return False
            best_req = min(unassigned, key=lambda r: abs(v["x"] - r["px"]) + abs(v["y"] - r["py"]))
            best_req["status"] = "assigned"
            best_req["assigned_vehicle"] = v_id
            v["status"] = "pickup"
            v["target_x"] = best_req["px"]
            v["target_y"] = best_req["py"]
            v["req_id"] = best_req["id"]
            return True

        elif action == 1:
            # Serve longest-waiting pending request (local quadrant first, else nearest)
            if not unassigned:
                return False
            v_quad = self.get_quadrant(v["x"], v["y"])
            local = [r for r in unassigned if self.get_quadrant(r["px"], r["py"]) == v_quad]
            if local:
                best_req = max(local, key=lambda r: (r["wait"], -(abs(v["x"] - r["px"]) + abs(v["y"] - r["py"]))))
            else:
                best_req = min(unassigned, key=lambda r: abs(v["x"] - r["px"]) + abs(v["y"] - r["py"]))
            best_req["status"] = "assigned"
            best_req["assigned_vehicle"] = v_id
            v["status"] = "pickup"
            v["target_x"] = best_req["px"]
            v["target_y"] = best_req["py"]
            v["req_id"] = best_req["id"]
            return True

        elif 2 <= action <= 5:
            # Reposition to quadrant center
            target_quad = action - 2
            tx, ty = self.quad_centers[target_quad]
            if v["x"] == tx and v["y"] == ty:
                # Already at target quadrant center
                v["status"] = "idle"
                v["target_x"] = None
                v["target_y"] = None
                v["req_id"] = None
                return True

            v["status"] = "repo"
            v["target_x"] = tx
            v["target_y"] = ty
            v["req_id"] = None
            return True

        return False

    def step(self):
        # 1. Spawn requests based on schedule:
        # Steps 0..99: normal (p = 0.20)
        # Steps 100..199: surge (p = 0.60, 3x rate, 75% in surge quadrant)
        # Steps 200..299: normal (p = 0.20)
        is_surge = (100 <= self.step_num < 200)
        prob = self.p_surge if is_surge else self.p_normal

        if self.rng.random() < prob:
            mid = self.grid_size // 2
            if is_surge and (self.rng.random() < 0.75):
                sq = self.surge_quad
                min_x = 0 if sq in (0, 2) else mid
                min_y = 0 if sq in (0, 1) else mid
                px = int(self.rng.integers(min_x, min_x + mid))
                py = int(self.rng.integers(min_y, min_y + mid))
            else:
                px = int(self.rng.integers(0, self.grid_size))
                py = int(self.rng.integers(0, self.grid_size))

            dx = int(self.rng.integers(0, self.grid_size))
            dy = int(self.rng.integers(0, self.grid_size))
            while dx == px and dy == py:
                dx = int(self.rng.integers(0, self.grid_size))
                dy = int(self.rng.integers(0, self.grid_size))

            self.requests.append({
                "id": self.req_id_counter,
                "px": px,
                "py": py,
                "dx": dx,
                "dy": dy,
                "spawn_step": self.step_num,
                "wait": 0,
                "status": "pending",
                "assigned_vehicle": None
            })
            self.req_id_counter += 1

        # 2. Move vehicles along Manhattan shortest paths
        step_dist = 0
        moved_by_v = {v["id"]: False for v in self.vehicles}
        for v in self.vehicles:
            if v["status"] == "idle" or v["target_x"] is None:
                continue

            tx, ty = v["target_x"], v["target_y"]
            moved = False
            if v["x"] < tx:
                v["x"] += 1
                moved = True
            elif v["x"] > tx:
                v["x"] -= 1
                moved = True
            elif v["y"] < ty:
                v["y"] += 1
                moved = True
            elif v["y"] > ty:
                v["y"] -= 1
                moved = True

            if moved:
                v["total_dist"] += 1
                step_dist += 1
                moved_by_v[v["id"]] = True
                if v["status"] in ("pickup", "repo"):
                    v["idle_dist"] += 1

        # 3. Check vehicle arrivals (auto-pickup, auto-dropoff, repo finish)
        dropoffs = 0
        dropoffs_by_v = {v["id"]: False for v in self.vehicles}
        for v in self.vehicles:
            if v["target_x"] is None:
                continue

            if v["x"] == v["target_x"] and v["y"] == v["target_y"]:
                if v["status"] == "pickup":
                    # Auto-pickup on arrival
                    req = next((r for r in self.requests if r["id"] == v["req_id"]), None)
                    if req and req["status"] == "assigned":
                        req["status"] = "in_transit"
                        v["status"] = "trip"
                        v["target_x"] = req["dx"]
                        v["target_y"] = req["dy"]

                elif v["status"] == "trip":
                    # Auto-dropoff on arrival
                    req = next((r for r in self.requests if r["id"] == v["req_id"]), None)
                    if req:
                        req["status"] = "completed"
                        v["trips"] += 1
                        dropoffs += 1
                        dropoffs_by_v[v["id"]] = True
                    v["status"] = "idle"
                    v["target_x"] = None
                    v["target_y"] = None
                    v["req_id"] = None

                elif v["status"] == "repo":
                    # Reached quadrant center
                    v["status"] = "idle"
                    v["target_x"] = None
                    v["target_y"] = None

        # 4. Update wait times and drop requests waiting > 60 steps
        drops = 0
        for r in self.requests:
            if r["status"] in ("pending", "assigned"):
                r["wait"] += 1
                if r["wait"] >= self.max_wait:
                    r["status"] = "dropped"
                    drops += 1
                    # Free assigned vehicle if still traveling to pickup
                    if r["assigned_vehicle"] is not None:
                        v = self.vehicles[r["assigned_vehicle"]]
                        if v["req_id"] == r["id"]:
                            v["status"] = "idle"
                            v["target_x"] = None
                            v["target_y"] = None
                            v["req_id"] = None

        # 5. Reward function:
        # - negative per step for every pending request (-0.15)
        # + positive per completed dropoff (+20.0)
        # - small penalty per unit distance (-0.05)
        # - larger penalty per dropped request (-25.0)
        pending_count = sum(1 for r in self.requests if r["status"] in ("pending", "assigned"))
        reward = (
            dropoffs * 20.0
            - pending_count * 0.15
            - step_dist * 0.05
            - drops * 25.0
        )

        self.step_num += 1
        done = self.step_num >= self.max_steps

        info = {
            "dropoffs": dropoffs,
            "drops": drops,
            "pending": pending_count,
            "distance": step_dist,
            "dropoffs_by_v": dropoffs_by_v,
            "moved_by_v": moved_by_v
        }

        return reward, done, info

    def get_metrics(self):
        served = [r for r in self.requests if r["status"] == "completed"]
        dropped = [r for r in self.requests if r["status"] == "dropped"]

        avg_wait = float(np.mean([r["wait"] for r in served])) if served else 0.0
        tot_dist = sum(v["total_dist"] for v in self.vehicles)
        idle_dist = sum(v["idle_dist"] for v in self.vehicles)

        surge_reqs = [r for r in self.requests if 100 <= r["spawn_step"] < 200]
        surge_served = [r for r in surge_reqs if r["status"] == "completed"]
        surge_wait = float(np.mean([r["wait"] for r in surge_served])) if surge_served else 0.0

        normal_reqs = [r for r in self.requests if r["spawn_step"] < 100 or r["spawn_step"] >= 200]
        normal_served = [r for r in normal_reqs if r["status"] == "completed"]
        normal_wait = float(np.mean([r["wait"] for r in normal_served])) if normal_served else 0.0

        return {
            "avg_wait": round(avg_wait, 2),
            "served": len(served),
            "dropped": len(dropped),
            "total_dist": tot_dist,
            "idle_dist": idle_dist,
            "surge_wait": round(surge_wait, 2),
            "normal_wait": round(normal_wait, 2),
            "surge_served": len(surge_served),
            "surge_dropped": sum(1 for r in surge_reqs if r["status"] == "dropped")
        }

    def get_frame(self):
        served_count = sum(1 for r in self.requests if r["status"] == "completed")
        dropped_count = sum(1 for r in self.requests if r["status"] == "dropped")
        completed_waits = [r["wait"] for r in self.requests if r["status"] == "completed"]
        cur_avg_wait = round(float(np.mean(completed_waits)), 2) if completed_waits else 0.0

        return {
            "step": self.step_num,
            "phase": self.get_phase(),
            "surge_quad": self.surge_quad,
            "is_surge": (100 <= self.step_num < 200),
            "grid_size": self.grid_size,
            "num_vehicles": self.num_vehicles,
            "served": served_count,
            "dropped": dropped_count,
            "avg_wait": cur_avg_wait,
            "vehicles": [
                {
                    "id": v["id"],
                    "x": v["x"],
                    "y": v["y"],
                    "status": v["status"],
                    "target_x": v["target_x"],
                    "target_y": v["target_y"],
                    "req_id": v["req_id"]
                }
                for v in self.vehicles
            ],
            "pending": [
                {
                    "id": r["id"],
                    "px": r["px"],
                    "py": r["py"],
                    "dx": r["dx"],
                    "dy": r["dy"],
                    "wait": r["wait"],
                    "status": r["status"],
                    "assigned_vehicle": r["assigned_vehicle"]
                }
                for r in self.requests
                if r["status"] in ("pending", "assigned")
            ]
        }
