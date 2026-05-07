from __future__ import annotations

import numpy as np
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.kinematics import Vehicle


class ContinuousSpawnHighwayEnv(HighwayEnv):
    """
    A highway-v0-style environment where:
    - the ego starts already on the highway
    - cars are placed both behind and ahead of the ego at reset
    - new cars can keep spawning during the episode, behind or ahead
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
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
                    "speed_range": [15.0, 30.0],
                },
                "lanes_count": 3,
                "vehicles_count": 20,
                "controlled_vehicles": 1,
                "duration": 35,
                "ego_spacing": 2,
                "vehicles_density": 1.0,
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "collision_reward": -2.0,
                "right_lane_reward": 0.0,
                "high_speed_reward": 0.3,
                "lane_change_reward": -0.01,
                "reward_speed_range": [18, 30],
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

                # custom traffic settings
                "ego_start_x": 180.0,
                "ego_start_speed": 24.0,
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

                # traffic speed controls
                "traffic_speed_range": [20.0, 27.0],
                "enforce_speed_each_step": False,
                "append_urgency_to_observation": True,
                "urgency_max_distance": 80.0,
                "urgency_speed_scale": 15.0,
                "urgency_ttc_horizon": 6.0,
            }
        )
        return config

    def define_spaces(self) -> None:
        super().define_spaces()
        self._base_observation_space = self.observation_type.space()

        if self.config.get("append_urgency_to_observation", False):
            base_dim = int(np.prod(self._base_observation_space.shape))
            total_dim = base_dim + self._urgency_feature_dim()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_dim,),
                dtype=np.float32,
            )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._spawn_step = 0
        self._other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        if self.config.get("enforce_speed_each_step", True):
            self._enforce_traffic_speeds()
        obs = self._augment_observation(obs)
        info["scenario"] = getattr(self, "_active_scenario", "unknown")
        info["urgency_features"] = self._urgency_feature_dict()
        return obs, info

    def _create_vehicles(self) -> None:
        """
        Create the ego first, then build one of several lane-decision scenarios.
        """
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        lanes_count = self.config["lanes_count"]
        ego_lane_id = self._sample_ego_lane(lanes_count)
        ego_lane = self.road.network.get_lane(("0", "1", int(ego_lane_id)))

        ego_x = float(self.config.get("ego_start_x", 180.0))
        ego_speed = float(self.config.get("ego_start_speed", 24.0))

        ego_base = Vehicle(
            self.road,
            ego_lane.position(ego_x, 0),
            ego_lane.heading_at(ego_x),
            ego_speed,
        )

        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_base.position,
            ego_base.heading,
            ego_base.speed,
        )

        self.vehicle = ego_vehicle
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)

        self._active_scenario = self._sample_scenario()
        self._build_scenario(other_vehicles_type, int(ego_lane_id))
        self._add_ambient_traffic(other_vehicles_type)

    def _reward(self, action) -> float:
        reward = super()._reward(action)
        close_follow_penalty = self._close_follow_penalty()
        reward -= float(self.config.get("close_follow_penalty_weight", 0.0)) * close_follow_penalty
        slow_leader_penalty = self._slow_leader_penalty()
        reward -= float(self.config.get("slow_leader_penalty_weight", 0.0)) * slow_leader_penalty
        return float(np.clip(reward, 0.0, 1.0))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        self._spawn_step = getattr(self, "_spawn_step", 0) + 1

        if not (terminated or truncated):
            spawned_vehicle = self._spawn_traffic()
            if self.config.get("enforce_speed_each_step", True):
                self._enforce_traffic_speeds()
            if spawned_vehicle:
                # Refresh the observation so newly spawned traffic is visible
                # to the policy on the same decision step.
                obs = self.observation_type.observe()

        obs = self._augment_observation(obs)
        info["scenario"] = getattr(self, "_active_scenario", "unknown")
        info["close_follow_penalty"] = self._close_follow_penalty()
        info["slow_leader_penalty"] = self._slow_leader_penalty()
        info["urgency_features"] = self._urgency_feature_dict()
        return obs, reward, terminated, truncated, info

    def _augment_observation(self, obs):
        obs_array = np.asarray(obs, dtype=np.float32)
        if not self.config.get("append_urgency_to_observation", False):
            return obs_array

        urgency_features = self._compute_urgency_features().astype(np.float32)
        return np.concatenate([obs_array.reshape(-1), urgency_features]).astype(np.float32)

    def _urgency_feature_dim(self) -> int:
        return 8 if self.config.get("append_urgency_to_observation", False) else 0

    def _compute_urgency_features(self) -> np.ndarray:
        max_gap = float(self.config.get("urgency_max_distance", 80.0))
        speed_scale = float(self.config.get("urgency_speed_scale", 15.0))
        ttc_horizon = float(self.config.get("urgency_ttc_horizon", 6.0))
        left_available = self._adjacent_lane_exists(-1)
        right_available = self._adjacent_lane_exists(1)

        same_front_gap, same_front_vehicle = self._nearest_vehicle_in_lane(direction="front")
        same_closing_speed = self._closing_speed_to_vehicle(same_front_vehicle)
        same_ttc_risk = self._ttc_risk(same_front_gap, same_closing_speed, ttc_horizon)

        left_front_gap, _ = self._nearest_vehicle_in_adjacent_lane(offset=-1, direction="front")
        left_rear_gap, _ = self._nearest_vehicle_in_adjacent_lane(offset=-1, direction="rear")
        right_front_gap, _ = self._nearest_vehicle_in_adjacent_lane(offset=1, direction="front")
        right_rear_gap, _ = self._nearest_vehicle_in_adjacent_lane(offset=1, direction="rear")

        left_escape_gap = min(left_front_gap, left_rear_gap) if left_available else 0.0
        right_escape_gap = min(right_front_gap, right_rear_gap) if right_available else 0.0
        best_escape_gap = max(left_escape_gap, right_escape_gap)

        features = np.array(
            [
                self._normalize_gap(same_front_gap, max_gap),
                np.clip(same_closing_speed / speed_scale, 0.0, 1.0),
                same_ttc_risk,
                self._normalize_gap(left_front_gap, max_gap) if left_available else 0.0,
                self._normalize_gap(left_rear_gap, max_gap) if left_available else 0.0,
                self._normalize_gap(right_front_gap, max_gap) if right_available else 0.0,
                self._normalize_gap(right_rear_gap, max_gap) if right_available else 0.0,
                self._normalize_gap(best_escape_gap, max_gap),
            ],
            dtype=np.float32,
        )
        return features

    def _urgency_feature_dict(self) -> dict[str, float]:
        values = self._compute_urgency_features()
        names = [
            "same_lane_front_gap",
            "same_lane_closing_speed",
            "same_lane_ttc_risk",
            "left_front_gap",
            "left_rear_gap",
            "right_front_gap",
            "right_rear_gap",
            "best_side_escape_gap",
        ]
        return {name: float(value) for name, value in zip(names, values)}

    def _nearest_vehicle_in_lane(self, direction: str):
        ego_lane = int(self.vehicle.lane_index[2])
        return self._nearest_vehicle_by_lane_id(ego_lane, direction)

    def _nearest_vehicle_in_adjacent_lane(self, offset: int, direction: str):
        if not self._adjacent_lane_exists(offset):
            return np.inf, None
        ego_lane = int(self.vehicle.lane_index[2])
        target_lane = ego_lane + offset
        return self._nearest_vehicle_by_lane_id(target_lane, direction)

    def _adjacent_lane_exists(self, offset: int) -> bool:
        if self.vehicle is None:
            return False
        ego_lane = int(self.vehicle.lane_index[2])
        target_lane = ego_lane + offset
        return 0 <= target_lane < self.config["lanes_count"]

    def _nearest_vehicle_by_lane_id(self, lane_id: int, direction: str):
        if self.road is None or self.vehicle is None:
            return np.inf, None

        ego_x = float(self.vehicle.position[0])
        best_gap = np.inf
        best_vehicle = None

        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue
            if int(vehicle.lane_index[2]) != lane_id:
                continue

            dx = float(vehicle.position[0]) - ego_x
            if direction == "front" and dx > 0.0 and dx < best_gap:
                best_gap = dx
                best_vehicle = vehicle
            if direction == "rear" and dx < 0.0 and -dx < best_gap:
                best_gap = -dx
                best_vehicle = vehicle

        return best_gap, best_vehicle

    def _closing_speed_to_vehicle(self, vehicle) -> float:
        if vehicle is None or self.vehicle is None:
            return 0.0
        return max(float(self.vehicle.speed - vehicle.speed), 0.0)

    def _ttc_risk(self, gap: float, closing_speed: float, ttc_horizon: float) -> float:
        if not np.isfinite(gap) or gap <= 0.0 or closing_speed <= 1e-6:
            return 0.0
        ttc = gap / closing_speed
        return float(np.clip(1.0 - (ttc / ttc_horizon), 0.0, 1.0))

    def _normalize_gap(self, gap: float, max_gap: float) -> float:
        if not np.isfinite(gap):
            return 1.0
        return float(np.clip(gap / max_gap, 0.0, 1.0))

    def _spawn_traffic(self) -> bool:
        if self.road is None:
            return False

        if len(self.road.vehicles) >= self.config["max_vehicles"]:
            return False

        spawn_interval = int(self.config.get("spawn_interval", 1))
        if spawn_interval > 1 and (self._spawn_step % spawn_interval != 0):
            return False

        if self.np_random.uniform() > self.config["spawn_probability"]:
            return False

        other_vehicles_type = getattr(self, "_other_vehicles_type", None)
        if other_vehicles_type is None:
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        if self._should_spawn_same_lane_blocker():
            blocker = self._spawn_same_lane_blocker(other_vehicles_type)
            if blocker is not None:
                self.road.vehicles.append(blocker)
                return True

        ahead_prob = float(self.config.get("spawn_ahead_probability", 0.5))
        spawn_ahead = self.np_random.uniform() < ahead_prob

        vehicle = self._make_vehicle_near_ego(other_vehicles_type, ahead=spawn_ahead)
        if vehicle is not None:
            self.road.vehicles.append(vehicle)
            return True

        return False

    def _sample_scenario(self) -> str:
        scenario_probs = self.config.get("scenario_probabilities", {})
        names = list(scenario_probs.keys())
        if not names:
            return "overtake_easy"

        weights = np.array([float(scenario_probs[name]) for name in names], dtype=float)
        if np.all(weights <= 0):
            return names[0]

        weights /= weights.sum()
        return str(self.np_random.choice(names, p=weights))

    def _build_scenario(self, vehicle_class, ego_lane_id: int) -> None:
        if self._active_scenario == "overtake_easy":
            lead_vehicle = self._spawn_same_lane_blocker(
                vehicle_class,
                lane_id=ego_lane_id,
                distance_range=tuple(self.config.get("lead_vehicle_distance_range", [38.0, 52.0])),
                speed_range=tuple(self.config.get("lead_vehicle_speed_range", [17.0, 21.0])),
            )
            if lead_vehicle is not None:
                self.road.vehicles.append(lead_vehicle)
            return

        if self._active_scenario == "overtake_blocked":
            lead_vehicle = self._spawn_same_lane_blocker(
                vehicle_class,
                lane_id=ego_lane_id,
                distance_range=tuple(self.config.get("lead_vehicle_distance_range", [32.0, 48.0])),
                speed_range=tuple(self.config.get("lead_vehicle_speed_range", [14.0, 18.0])),
            )
            if lead_vehicle is not None:
                self.road.vehicles.append(lead_vehicle)

            adjacent_lane_ids = self._adjacent_lane_ids(ego_lane_id)
            if not adjacent_lane_ids:
                return

            blocked_lane_id = int(self.np_random.choice(adjacent_lane_ids))
            semi_open_lane_ids = [lane_id for lane_id in adjacent_lane_ids if lane_id != blocked_lane_id]

            front_blocker = self._make_vehicle_near_ego(
                vehicle_class,
                ahead=True,
                forced_lane_id=blocked_lane_id,
                distance_range=tuple(self.config.get("blocked_adjacent_front_distance_range", [8.0, 18.0])),
                speed_range=tuple(self.config.get("blocked_adjacent_speed_range", [24.0, 28.0])),
                allow_lane_change=False,
                min_gap_override=float(self.config.get("blocked_adjacent_front_min_gap", 8.0)),
            )
            if front_blocker is not None:
                self.road.vehicles.append(front_blocker)

            rear_blocker = self._make_vehicle_near_ego(
                vehicle_class,
                ahead=False,
                forced_lane_id=blocked_lane_id,
                distance_range=tuple(self.config.get("blocked_adjacent_rear_distance_range", [6.0, 14.0])),
                speed_range=tuple(self.config.get("blocked_adjacent_speed_range", [24.0, 28.0])),
                allow_lane_change=False,
                min_gap_override=float(self.config.get("blocked_adjacent_rear_min_gap", 8.0)),
            )
            if rear_blocker is not None:
                self.road.vehicles.append(rear_blocker)

            for lane_id in semi_open_lane_ids:
                front_blocker = self._make_vehicle_near_ego(
                    vehicle_class,
                    ahead=True,
                    forced_lane_id=lane_id,
                    distance_range=tuple(self.config.get("semi_open_adjacent_front_distance_range", [28.0, 42.0])),
                    speed_range=tuple(self.config.get("semi_open_adjacent_speed_range", [22.0, 26.0])),
                    allow_lane_change=False,
                )
                if front_blocker is not None:
                    self.road.vehicles.append(front_blocker)
            return

        if self._active_scenario == "stay_best":
            lead_vehicle = self._make_vehicle_near_ego(
                vehicle_class,
                ahead=True,
                forced_lane_id=ego_lane_id,
                distance_range=tuple(self.config.get("stay_best_lead_distance_range", [60.0, 85.0])),
                speed_range=tuple(self.config.get("stay_best_lead_speed_range", [22.0, 25.0])),
                allow_lane_change=False,
            )
            if lead_vehicle is not None:
                self.road.vehicles.append(lead_vehicle)

    def _add_ambient_traffic(self, vehicle_class) -> None:
        for _ in range(int(self.config.get("initial_vehicles_behind", 1))):
            vehicle = self._make_vehicle_near_ego(vehicle_class, ahead=False)
            if vehicle is not None:
                self.road.vehicles.append(vehicle)

        for _ in range(int(self.config.get("initial_vehicles_ahead", 1))):
            vehicle = self._make_vehicle_near_ego(vehicle_class, ahead=True)
            if vehicle is not None:
                self.road.vehicles.append(vehicle)

    def _spawn_same_lane_blocker(
        self,
        vehicle_class,
        lane_id: int | None = None,
        distance_range: tuple[float, float] | None = None,
        speed_range: tuple[float, float] | None = None,
    ):
        if lane_id is None:
            lane_id = int(self.vehicle.lane_index[2])

        if distance_range is None:
            distance_range = tuple(self.config.get("same_lane_blocker_distance_range", [30.0, 45.0]))

        if speed_range is None:
            speed_range = tuple(self.config.get("same_lane_blocker_speed_range", [16.0, 20.0]))

        return self._make_vehicle_near_ego(
            vehicle_class,
            ahead=True,
            forced_lane_id=int(lane_id),
            distance_range=distance_range,
            speed_range=speed_range,
            allow_lane_change=False,
        )

    def _should_spawn_same_lane_blocker(self) -> bool:
        if self._active_scenario not in {"overtake_easy", "overtake_blocked"}:
            return False

        interval = int(self.config.get("same_lane_blocker_check_interval", 12))
        if interval <= 0 or self._spawn_step % interval != 0:
            return False

        lookahead = float(self.config.get("same_lane_blocker_lookahead", 55.0))
        if self._has_same_lane_leader(lookahead):
            return False

        probability = float(self.config.get("same_lane_blocker_probability", 0.6))
        return bool(self.np_random.uniform() < probability)

    def _has_same_lane_leader(self, max_distance: float) -> bool:
        ego_lane = int(self.vehicle.lane_index[2])
        ego_x = float(self.vehicle.position[0])
        min_speed_delta = float(self.config.get("close_follow_speed_diff", 1.5))

        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue

            if int(vehicle.lane_index[2]) != ego_lane:
                continue

            distance = float(vehicle.position[0]) - ego_x
            if 0.0 < distance <= max_distance and float(vehicle.speed) <= float(self.vehicle.speed) + min_speed_delta:
                return True

        return False

    def _close_follow_penalty(self) -> float:
        if self.road is None or self.vehicle is None:
            return 0.0

        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        if front_vehicle is None:
            return 0.0

        distance = float(self.vehicle.lane_distance_to(front_vehicle))
        speed_diff = float(self.vehicle.speed - front_vehicle.speed)
        safe_distance = float(self.config.get("close_follow_distance", 22.0))
        min_speed_diff = float(self.config.get("close_follow_speed_diff", 1.5))

        if distance <= 0 or distance >= safe_distance or speed_diff <= min_speed_diff:
            return 0.0

        closeness = 1.0 - np.clip(distance / safe_distance, 0.0, 1.0)
        closing_speed = np.clip((speed_diff - min_speed_diff) / 8.0, 0.0, 1.0)
        return float(closeness * closing_speed)

    def _slow_leader_penalty(self) -> float:
        if self.road is None or self.vehicle is None:
            return 0.0

        ego_lane = int(self.vehicle.lane_index[2])
        ego_x = float(self.vehicle.position[0])
        max_distance = float(self.config.get("slow_leader_distance", 40.0))
        min_speed_diff = float(self.config.get("slow_leader_speed_diff", 2.0))

        nearest_distance = None
        nearest_speed_diff = None
        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue

            if int(vehicle.lane_index[2]) != ego_lane:
                continue

            distance = float(vehicle.position[0]) - ego_x
            if distance <= 0.0 or distance > max_distance:
                continue

            speed_diff = float(self.vehicle.speed - vehicle.speed)
            if speed_diff <= min_speed_diff:
                continue

            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_speed_diff = speed_diff

        if nearest_distance is None or nearest_speed_diff is None:
            return 0.0

        closeness = 1.0 - np.clip(nearest_distance / max_distance, 0.0, 1.0)
        speed_factor = np.clip((nearest_speed_diff - min_speed_diff) / 8.0, 0.0, 1.0)
        return float(closeness * speed_factor)

    def _sample_ego_lane(self, lanes_count: int) -> int:
        policy = str(self.config.get("ego_start_lane_policy", "random")).lower()
        if policy == "center":
            return lanes_count // 2
        return int(self.np_random.integers(lanes_count))

    def _adjacent_lane_ids(self, ego_lane_id: int) -> list[int]:
        adjacent = []
        if ego_lane_id > 0:
            adjacent.append(ego_lane_id - 1)
        if ego_lane_id < self.config["lanes_count"] - 1:
            adjacent.append(ego_lane_id + 1)
        return adjacent

    def _sample_spawn_lane(self) -> int:
        """
        Bias traffic toward the ego lane and adjacent lanes so more vehicles are
        relevant to the ego's current decision.
        """
        lanes_count = self.config["lanes_count"]

        try:
            ego_lane_id = int(self.vehicle.lane_index[2])
        except Exception:
            ego_lane_id = int(self.np_random.integers(lanes_count))

        candidates = [ego_lane_id]
        weights = [0.65]

        if ego_lane_id > 0:
            candidates.append(ego_lane_id - 1)
            weights.append(0.175)

        if ego_lane_id < lanes_count - 1:
            candidates.append(ego_lane_id + 1)
            weights.append(0.175)

        for lane_id in range(lanes_count):
            if lane_id not in candidates:
                candidates.append(lane_id)
                weights.append(0.05)

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        return int(self.np_random.choice(candidates, p=weights))

    def _make_vehicle_near_ego(
        self,
        vehicle_class,
        ahead: bool,
        forced_lane_id: int | None = None,
        distance_range: tuple[float, float] | None = None,
        speed_range: tuple[float, float] | None = None,
        allow_lane_change: bool = True,
        min_gap_override: float | None = None,
    ):
        """
        Spawn one vehicle in a lane either ahead of or behind the ego.

        Optional arguments let us force a slower same-lane lead vehicle at reset.
        """
        ego_x = float(self.vehicle.position[0])
        lanes_count = self.config["lanes_count"]

        for _ in range(20):
            if forced_lane_id is None:
                lane_id = self._sample_spawn_lane()
            else:
                lane_id = int(forced_lane_id)
            lane = self.road.network.get_lane(("0", "1", lane_id))

            if distance_range is None:
                if ahead:
                    delta_x = self.np_random.uniform(25.0, 90.0)
                    x = ego_x + delta_x
                else:
                    delta_x = self.np_random.uniform(20.0, 70.0)
                    x = ego_x - delta_x
            else:
                delta_x = self.np_random.uniform(*distance_range)
                x = ego_x + delta_x if ahead else ego_x - delta_x

            if x < 0:
                continue

            if speed_range is None:
                min_speed, max_speed = self._speed_range()
                speed = self.np_random.uniform(min_speed, max_speed)
            else:
                speed = self.np_random.uniform(*speed_range)

            vehicle = vehicle_class(
                self.road,
                lane.position(x, 0),
                lane.heading_at(x),
                speed,
            )
            # vehicle.randomize_behavior() disabled for the time being
            if hasattr(vehicle, "enable_lane_change"):
                vehicle.enable_lane_change = allow_lane_change
            if not allow_lane_change and hasattr(vehicle, "target_lane_index"):
                vehicle.target_lane_index = vehicle.lane_index
            self._set_vehicle_speed(vehicle, target_speed=speed)

            if self._space_is_free(vehicle, min_gap_override=min_gap_override):
                return vehicle

        return None

    def _space_is_free(self, candidate, min_gap_override: float | None = None) -> bool:
        """
        Reject vehicles that would spawn too close to existing vehicles.
        """
        if min_gap_override is None:
            min_gap = float(self.config["spawn_min_gap"])
        else:
            min_gap = float(min_gap_override)
        min_gap_sq = min_gap * min_gap

        for vehicle in self.road.vehicles:
            dx = candidate.position[0] - vehicle.position[0]
            dy = candidate.position[1] - vehicle.position[1]
            if (dx * dx + dy * dy) < min_gap_sq:
                return False

        return True

    def _speed_range(self):
        speed_range = self.config.get("traffic_speed_range", [20.0, 30.0])
        if not isinstance(speed_range, (list, tuple)) or len(speed_range) != 2:
            speed_range = [20.0, 30.0]
        min_speed = float(min(speed_range))
        max_speed = float(max(speed_range))
        return min_speed, max_speed

    def _set_vehicle_speed(self, vehicle, target_speed: float | None = None) -> None:
        if target_speed is None:
            min_speed, max_speed = self._speed_range()
            target_speed = float(self.np_random.uniform(min_speed, max_speed))
        else:
            target_speed = float(target_speed)

        if hasattr(vehicle, "target_speed"):
            try:
                vehicle.target_speed = target_speed
            except Exception:
                pass

        if hasattr(vehicle, "speed"):
            try:
                vehicle.speed = target_speed
            except Exception:
                pass

        if hasattr(vehicle, "velocity"):
            try:
                vel = np.array(vehicle.velocity, dtype=float)
                norm = np.linalg.norm(vel)
                if norm > 1e-6:
                    vehicle.velocity = vel / norm * target_speed
                else:
                    vehicle.velocity = np.array([target_speed, 0.0], dtype=float)
            except Exception:
                pass

    def _enforce_traffic_speeds(self) -> None:
        if self.road is None:
            return

        min_speed, max_speed = self._speed_range()

        for vehicle in self.road.vehicles:
            if vehicle is getattr(self, "vehicle", None):
                continue

            current_speed = None
            if hasattr(vehicle, "speed"):
                try:
                    current_speed = float(vehicle.speed)
                except Exception:
                    current_speed = None
            elif hasattr(vehicle, "velocity"):
                try:
                    current_speed = float(np.linalg.norm(vehicle.velocity))
                except Exception:
                    current_speed = None

            if current_speed is None:
                continue

            if current_speed < min_speed or current_speed > max_speed:
                self._set_vehicle_speed(vehicle)
