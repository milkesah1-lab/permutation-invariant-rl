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
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "order": "sorted",
        "see_behind": True,
    },
    "action": {
        "type": "ContinuousAction",
        "speed_range": [20.0, 34.0],
    },
    "lanes_count": 3,
    "vehicles_count": 20,
    "controlled_vehicles": 1,
    "duration": 35,
    "ego_spacing": 2,
    "vehicles_density": 1.0,
    "simulation_frequency": 15,
    "policy_frequency": 15,
    "collision_reward": -2.0,
    "right_lane_reward": 0.0,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0.0,
    "reward_speed_range": [22, 32],
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
    "ego_start_x": 180.0,
    "ego_start_speed": 27.0,
    "ego_start_lane_policy": "random",
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
    "same_lane_blocker_distance_range": [24.0, 34.0],
    "same_lane_blocker_speed_range": [12.0, 17.0],
    "same_lane_blocker_probability": 0.25,
    "same_lane_blocker_lookahead": 50.0,
    "close_follow_penalty_weight": 0.25,
    "close_follow_distance": 30.0,
    "close_follow_speed_diff": 1.0,
    "slow_leader_penalty_weight": 0.30,
    "slow_leader_distance": 70.0,
    "slow_leader_speed_diff": 1.5,
    "traffic_speed_range": [19.0, 25.0],
    "enforce_speed_each_step": False,
    "append_urgency_to_observation": True,
    "urgency_max_distance": 80.0,
    "urgency_speed_scale": 15.0,
    "urgency_ttc_horizon": 6.0,
}


HIGHWAY_ENV_CONFIGS: dict[str, dict[str, Any]] = {
    "curriculum_stage_1_open_lane": _make_curriculum_config(
        {
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
    "curriculum_stage_2_easy_overtake": _make_curriculum_config(
        {
            "lead_vehicle_distance_range": [24.0, 36.0],
            "lead_vehicle_speed_range": [12.0, 16.0],
            "same_lane_blocker_distance_range": [24.0, 34.0],
            "same_lane_blocker_speed_range": [12.0, 17.0],
        }
    ),
    "curriculum_stage_3_mixed_traffic": _make_curriculum_config(
        {
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
    "curriculum_stage_4_dense_traffic": _make_curriculum_config(
        {
            "vehicles_count": 32,
            "duration": 50,
            "initial_vehicles_ahead": 5,
            "initial_vehicles_behind": 4,
            "spawn_probability": 0.35,
            "spawn_interval": 1,
            "spawn_min_gap": 16.0,
            "max_vehicles": 24,
            "spawn_ahead_probability": 0.7,
            "lead_vehicle_distance_range": [20.0, 30.0],
            "lead_vehicle_speed_range": [11.0, 15.0],
            "blocked_adjacent_front_distance_range": [8.0, 16.0],
            "blocked_adjacent_rear_distance_range": [6.0, 14.0],
            "blocked_adjacent_front_min_gap": 7.0,
            "blocked_adjacent_rear_min_gap": 7.0,
            "semi_open_adjacent_front_distance_range": [18.0, 28.0],
            "semi_open_adjacent_speed_range": [23.0, 28.0],
            "blocked_adjacent_speed_range": [24.0, 30.0],
            "scenario_probabilities": {
                "overtake_easy": 0.20,
                "overtake_blocked": 0.70,
                "stay_best": 0.10,
            },
            "same_lane_blocker_check_interval": 8,
            "same_lane_blocker_distance_range": [18.0, 28.0],
            "same_lane_blocker_speed_range": [10.0, 15.0],
            "same_lane_blocker_probability": 0.85,
            "same_lane_blocker_lookahead": 60.0,
            "slow_leader_penalty_weight": 0.48,
            "close_follow_penalty_weight": 0.40,
            "traffic_speed_range": [22.0, 30.0],
        }
    ),
}


def get_highway_config(name: str) -> dict[str, Any]:
    if name not in HIGHWAY_ENV_CONFIGS:
        valid = ", ".join(HIGHWAY_ENV_CONFIGS.keys())
        raise ValueError(f"Unknown highway config '{name}'. Valid options: {valid}")
    return deepcopy(HIGHWAY_ENV_CONFIGS[name])
