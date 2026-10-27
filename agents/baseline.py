"""
Traditional Nearest-Vehicle Dispatcher (Greedy Baseline).
Standard heuristic widely used in legacy taxi and ride-hailing fleets.
Dispatches the nearest available vehicle to an incoming request, with no proactive repositioning.
"""

from typing import Optional
from env.city import CityFleetEnv


class NearestVehicleDispatcher:
    """
    Greedy nearest-vehicle heuristic dispatcher.
    Whenever a vehicle is IDLE:
    - If there are unassigned pending requests, dispatches to the nearest one (Action 0).
    - Does NOT proactively reposition to anticipate demand surges (Actions 2-5 are never taken).
    """

    def __init__(self, name: str = "Nearest-Vehicle Baseline"):
        self.name = name

    def select_action(self, env: CityFleetEnv, vehicle_id: int) -> Optional[int]:
        """
        Determines the action for an idle vehicle.
        Returns:
            0: if unassigned pending requests exist (serve nearest).
            None: if no pending requests exist (remain in place / idle).
        """
        unassigned = env.get_unassigned_pending_requests()
        if unassigned:
            return 0  # Action 0 = Serve nearest pending request
        return None  # No action needed; vehicle stays idle where it is

    def update(self, *args, **kwargs) -> None:
        """Baseline does not learn; update is a no-op."""
        pass
