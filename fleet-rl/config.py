"""
Centralized configuration definitions for Fleet-RL environments and experiments.
Preserves both V1 and V2 configurations cleanly.
"""

from env.city import CityEnv

ENV_V1 = {
    "version": "v1",
    "grid_size": 10,
    "num_vehicles": 4,
    "max_steps": 300,
    "max_wait": 60,
    "p_normal": 0.20,
    "p_surge": 0.60,
    "surge_concentration": 0.75,
}

ENV_V2 = {
    "version": "v2",
    "grid_size": 16,
    "num_vehicles": 8,
    "max_steps": 300,
    "max_wait": 60,
    "p_normal": 0.25,
    "p_surge": 0.55,
    "surge_concentration": 0.75,
}

def make_env(version="v2", seed=42):
    """
    Factory creating a CityEnv with parameters matching the specified version.
    """
    cfg = ENV_V2 if version == "v2" else ENV_V1
    return CityEnv(
        grid_size=cfg["grid_size"],
        num_vehicles=cfg["num_vehicles"],
        max_steps=cfg["max_steps"],
        max_wait=cfg["max_wait"],
        p_normal=cfg["p_normal"],
        p_surge=cfg["p_surge"],
        seed=seed
    )
