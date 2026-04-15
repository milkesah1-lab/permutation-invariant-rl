from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_HIGHWAY_CONFIG = "curriculum_stage_2_easy_overtake"


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _make_curriculum_config(overrides: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(_BASE_CURRICULUM_CONFIG)
    return _deep_update(config, overrides)


_BASE_CURRICULUM_CONFIG: dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [18, 22, 26, 30],
    },
    "lanes_count": 3,
    "vehicles_count": 20,
    "controlled_vehicles": 1,
    "duration": 35,
    "ego_spacing": 2,
    "vehicles_density": 1.0,
    "simulation_frequency": 15,
    "policy_frequency": 2,
    "collision_reward": -1.0,
    "right_lane_reward": 0.05,
    "high_speed_reward": 0.4,
    "lane_change_reward": -0.02,
    "reward_speed_range": [20, 30],
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
    # Custom continuous-spawn curriculum settings
    "ego_start_x": 180.0,
    "ego_start_speed": 24.0,
    "ego_start_lane_policy": "center",
    "initial_vehicles_ahead": 2,
    "initial_vehicles_behind": 1,
    "spawn_probability": 0.10,
    "spawn_interval": 3,
    "spawn_min_gap": 20.0,
    "max_vehicles": 12,
    "spawn_ahead_probability": 0.6,
    "lead_vehicle_distance_range": [32.0, 48.0],
    "lead_vehicle_speed_range": [14.0, 18.0],
    "stay_best_lead_distance_range": [65.0, 90.0],
    "stay_best_lead_speed_range": [24.0, 27.0],
    "blocked_adjacent_front_distance_range": [18.0, 30.0],
    "blocked_adjacent_rear_distance_range": [15.0, 25.0],
    "blocked_adjacent_front_min_gap": 10.0,
    "blocked_adjacent_rear_min_gap": 10.0,
    "semi_open_adjacent_front_distance_range": [32.0, 48.0],
    "semi_open_adjacent_speed_range": [22.0, 26.0],
    "blocked_adjacent_speed_range": [22.0, 27.0],
    "scenario_probabilities": {
        "overtake_easy": 0.70,
        "overtake_blocked": 0.15,
        "stay_best": 0.15,
    },
    "same_lane_blocker_check_interval": 10,
    "same_lane_blocker_distance_range": [30.0, 44.0],
    "same_lane_blocker_speed_range": [15.0, 19.0],
    "same_lane_blocker_probability": 0.25,
    "same_lane_blocker_lookahead": 50.0,
    "close_follow_penalty_weight": 0.25,
    "close_follow_distance": 24.0,
    "close_follow_speed_diff": 1.0,
    "slow_leader_penalty_weight": 0.30,
    "slow_leader_distance": 55.0,
    "slow_leader_speed_diff": 1.5,
    "traffic_speed_range": [20.0, 27.0],
    "enforce_speed_each_step": False,
}


HIGHWAY_ENV_CONFIGS: dict[str, dict[str, Any]] = {
    # Curriculum stage 1:
    # Very easy overtakes with a clear adjacent lane and little ambient traffic.
    "curriculum_stage_1_open_lane": _make_curriculum_config(
        {
            "observation": {"vehicles_count": 8},
            "vehicles_count": 12,
            "duration": 30,
            "initial_vehicles_ahead": 1,
            "initial_vehicles_behind": 0,
            "spawn_probability": 0.05,
            "spawn_interval": 4,
            "max_vehicles": 8,
            "lead_vehicle_distance_range": [40.0, 55.0],
            "lead_vehicle_speed_range": [15.0, 18.0],
            "scenario_probabilities": {
                "overtake_easy": 0.90,
                "overtake_blocked": 0.00,
                "stay_best": 0.10,
            },
            "same_lane_blocker_probability": 0.10,
            "slow_leader_penalty_weight": 0.28,
            "close_follow_penalty_weight": 0.22,
            "traffic_speed_range": [20.0, 25.0],
        }
    ),

    # Curriculum stage 2:
    # Default training preset. Slow leaders are common but the overtake is usually available.
    "curriculum_stage_2_easy_overtake": _make_curriculum_config({}),

    # Curriculum stage 3:
    # Mixed traffic where blocked overtakes happen often enough that lane choice matters.
    "curriculum_stage_3_mixed_traffic": _make_curriculum_config(
        {
            "observation": {"vehicles_count": 10},
            "vehicles_count": 24,
            "duration": 40,
            "initial_vehicles_ahead": 3,
            "initial_vehicles_behind": 2,
            "spawn_probability": 0.18,
            "spawn_interval": 2,
            "max_vehicles": 16,
            "lead_vehicle_distance_range": [28.0, 40.0],
            "lead_vehicle_speed_range": [13.0, 17.0],
            "blocked_adjacent_front_distance_range": [12.0, 24.0],
            "blocked_adjacent_rear_distance_range": [10.0, 20.0],
            "semi_open_adjacent_front_distance_range": [26.0, 40.0],
            "scenario_probabilities": {
                "overtake_easy": 0.50,
                "overtake_blocked": 0.35,
                "stay_best": 0.15,
            },
            "same_lane_blocker_probability": 0.50,
            "slow_leader_penalty_weight": 0.36,
            "close_follow_penalty_weight": 0.30,
            "traffic_speed_range": [20.0, 28.0],
        }
    ),

    # Curriculum stage 4:
    # Dense traffic with tighter gaps and more blocked adjacent lanes.
    "curriculum_stage_4_dense_traffic": _make_curriculum_config(
        {
            "observation": {"vehicles_count": 12},
            "vehicles_count": 28,
            "duration": 45,
            "initial_vehicles_ahead": 4,
            "initial_vehicles_behind": 3,
            "spawn_probability": 0.24,
            "spawn_interval": 2,
            "max_vehicles": 20,
            "lead_vehicle_distance_range": [24.0, 36.0],
            "lead_vehicle_speed_range": [12.0, 16.0],
            "blocked_adjacent_front_distance_range": [10.0, 20.0],
            "blocked_adjacent_rear_distance_range": [8.0, 18.0],
            "semi_open_adjacent_front_distance_range": [22.0, 34.0],
            "scenario_probabilities": {
                "overtake_easy": 0.35,
                "overtake_blocked": 0.50,
                "stay_best": 0.15,
            },
            "same_lane_blocker_probability": 0.65,
            "slow_leader_penalty_weight": 0.42,
            "close_follow_penalty_weight": 0.34,
            "traffic_speed_range": [20.0, 29.0],
        }
    ),
}


def get_highway_config(name: str) -> dict[str, Any]:
    if name not in HIGHWAY_ENV_CONFIGS:
        valid = ", ".join(HIGHWAY_ENV_CONFIGS.keys())
        raise ValueError(f"Unknown highway config '{name}'. Valid options: {valid}")
    return deepcopy(HIGHWAY_ENV_CONFIGS[name])
