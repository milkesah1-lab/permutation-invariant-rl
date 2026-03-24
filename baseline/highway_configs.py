from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_HIGHWAY_CONFIG = "realistic_light"

HIGHWAY_ENV_CONFIGS: dict[str, dict[str, Any]] = {
    # realistic_light: keeps multi-lane traffic and interactions, trimmed for faster rollouts.
    "realistic_light": {
        "action": {"type": "ContinuousAction"},
        "lanes_count": 3,  # fewer lanes reduces sim cost while preserving lane changes
        "vehicles_count": 30,  # lower density speeds up dynamics without losing traffic context
        "duration": 40,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 12,  # smaller observation vector for quicker forward passes
            "features": ["presence", "x", "y", "vx", "vy"],
        },
    },
    # fast_experiment: aggressive simplifications for rapid iteration, less realism.
    "fast_experiment": {
        "action": {"type": "ContinuousAction"},
        "lanes_count": 2,  # fewer lanes -> fewer neighbors to simulate
        "vehicles_count": 12,  # smaller traffic pool for faster steps
        "duration": 25,
        "simulation_frequency": 10,
        "policy_frequency": 2,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 6,  # fewer observed vehicles speeds up training
            "features": ["presence", "x", "y", "vx", "vy"],
        },
    },
    # debug_minimal: shortest, tiniest setup for quick sanity checks over realism.
    "debug_minimal": {
        "action": {"type": "ContinuousAction"},
        "lanes_count": 1,  # single lane removes most interactions
        "vehicles_count": 6,  # minimal traffic for fastest sim
        "duration": 12,
        "simulation_frequency": 6,
        "policy_frequency": 2,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 4,  # tiny observation for quick debug runs
            "features": ["presence", "x", "y", "vx", "vy"],
        },
    },
}


def get_highway_config(name: str) -> dict[str, Any]:
    if name not in HIGHWAY_ENV_CONFIGS:
        valid = ", ".join(HIGHWAY_ENV_CONFIGS.keys())
        raise ValueError(f"Unknown highway config '{name}'. Valid options: {valid}")
    return deepcopy(HIGHWAY_ENV_CONFIGS[name])
