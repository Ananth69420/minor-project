class BaselineDispatcher:
    def __init__(self):
        self.name = "Nearest Vehicle"

    def select_action(self, env, v_id):
        # Action 0 = nearest pending request
        unassigned = [r for r in env.requests if r["status"] == "pending"]
        if unassigned:
            return 0
        return None

class ForecastRuleDispatcher:
    def __init__(self):
        self.name = "Forecast-Aware Rule"

    def select_action(self, env, v_id):
        v = env.vehicles[v_id]
        if v["status"] != "idle":
            return None
        unassigned = [r for r in env.requests if r["status"] == "pending"]
        if unassigned:
            return 0
        # If idle with no requests during pre-surge or surge phases, reposition to hotspot
        phase = env.get_phase()
        if phase in (1, 2):
            hotspot = env.surge_quad
            tx, ty = env.quad_centers[hotspot]
            if v["x"] != tx or v["y"] != ty:
                return 2 + hotspot
        return None
