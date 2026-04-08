from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_HIGHWAY_CONFIG = "realistic_light"

HIGHWAY_ENV_CONFIGS: dict[str, dict[str, Any]] = {
    # realistic_light:
    # Main training setup for the project.
    # Keeps realistic multi-lane interaction while avoiding unnecessary simulation cost.
    "realistic_light": {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,  # smaller observation for faster forward passes
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
        },
        "action": {
            "type": "ContinuousAction",
            "speed_range": [0.0, 30.0],
        },
        "lanes_count": 3,
        "vehicles_count": 24,          # enough traffic for interaction, lighter than stock 50
        "controlled_vehicles": 1,
        "duration": 60,                # more realistic than 40, but not excessively long
        "ego_spacing": 2,
        "vehicles_density": 1.0,
        "simulation_frequency": 15,
        "policy_frequency": 5,

        # Reward shaping
        "collision_reward": -1.0,
        "right_lane_reward": 0.05,     # mild encouragement, not dominant
        "high_speed_reward": 0.4,
        "lane_change_reward": -0.02,   # discourages unnecessary weaving
        "reward_speed_range": [20, 30],
        "normalize_reward": True,

        # Safety / termination
        "offroad_terminal": True,      # good for a safety-focused project

        # Vehicle model
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",

        # Rendering
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
    },

    # fast_experiment:
    # Faster setting for ablations and quick hyperparameter checks.
    "fast_experiment": {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 6,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
        },
        "action": {
            "type": "ContinuousAction",
            "speed_range": [0.0, 30.0],
        },
        "lanes_count": 2,
        "vehicles_count": 10,
        "controlled_vehicles": 1,
        "duration": 30,
        "ego_spacing": 2,
        "vehicles_density": 1.0,
        "simulation_frequency": 10,
        "policy_frequency": 2,

        "collision_reward": -1.0,
        "right_lane_reward": 0.03,
        "high_speed_reward": 0.35,
        "lane_change_reward": -0.02,
        "reward_speed_range": [18, 28],
        "normalize_reward": True,

        "offroad_terminal": True,

        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",

        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
    },

    # debug_minimal:
    # Tiny setup for quick code sanity checks, not for real evaluation.
    "debug_minimal": {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 4,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
        },
        "action": {
            "type": "ContinuousAction",
            "speed_range": [0.0, 30.0],
        },
        "lanes_count": 1,
        "vehicles_count": 6,
        "controlled_vehicles": 1,
        "duration": 12,
        "ego_spacing": 2,
        "vehicles_density": 1.0,
        "simulation_frequency": 6,
        "policy_frequency": 2,

        "collision_reward": -1.0,
        "right_lane_reward": 0.0,
        "high_speed_reward": 0.2,
        "lane_change_reward": 0.0,
        "reward_speed_range": [15, 25],
        "normalize_reward": True,

        "offroad_terminal": True,

        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",

        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
    },
}


def get_highway_config(name: str) -> dict[str, Any]:
    if name not in HIGHWAY_ENV_CONFIGS:
        valid = ", ".join(HIGHWAY_ENV_CONFIGS.keys())
        raise ValueError(f"Unknown highway config '{name}'. Valid options: {valid}")
    return deepcopy(HIGHWAY_ENV_CONFIGS[name])