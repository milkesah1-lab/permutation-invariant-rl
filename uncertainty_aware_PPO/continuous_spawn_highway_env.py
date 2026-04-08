from __future__ import annotations

import numpy as np

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
                },
                "action": {"type": "ContinuousAction",
                           "speed_range" : [0.0, 30.0],},
                "lanes_count": 4,
                "vehicles_count": 20,
                "controlled_vehicles": 1,
                "duration": 40,
                "ego_spacing": 2,
                "vehicles_density": 1.0,
                "simulation_frequency": 15,
                "policy_frequency": 5,

                # custom traffic settings
                "ego_start_x": 180.0,          # ego starts already well onto the highway
                "initial_vehicles_ahead": 6,
                "initial_vehicles_behind": 3,
                "spawn_probability": 0.05,
                "spawn_interval": 3,
                "spawn_min_gap": 20.0,
                "max_vehicles": 20,
                "spawn_ahead_probability": 0.5,

                # traffic speed controls
                "traffic_speed_range": [20.0, 30.0],  # desired speed range (m/s)
                "enforce_speed_each_step": False,     # keep non-ego traffic within range

                "normalize":True,
                "absolute" :False,
            }
        )
        return config

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._spawn_step = 0
        self._other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self._enforce_traffic_speeds()
        return obs, info

    def _create_vehicles(self) -> None:
        """
        Create ego first at a chosen position, then place traffic both behind and ahead.
        """
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        lanes_count = self.config["lanes_count"]
        ego_lane_id = self.np_random.integers(lanes_count)
        ego_lane = self.road.network.get_lane(("0", "1", int(ego_lane_id)))

        ego_x = float(self.config.get("ego_start_x", 180.0))
        ego_speed = 25.0

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

        # Initial traffic behind ego
        for _ in range(int(self.config.get("initial_vehicles_behind", 6))):
            vehicle = self._make_vehicle_near_ego(other_vehicles_type, ahead=False)
            if vehicle is not None:
                self.road.vehicles.append(vehicle)

        # Initial traffic ahead of ego
        for _ in range(int(self.config.get("initial_vehicles_ahead", 10))):
            vehicle = self._make_vehicle_near_ego(other_vehicles_type, ahead=True)
            if vehicle is not None:
                self.road.vehicles.append(vehicle)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        self._spawn_step = getattr(self, "_spawn_step", 0) + 1

        if not (terminated or truncated):
            self._spawn_traffic()
            if self.config.get("enforce_speed_each_step", True):
                self._enforce_traffic_speeds()

        return obs, reward, terminated, truncated, info

    def _spawn_traffic(self) -> None:
        if self.road is None:
            return

        if len(self.road.vehicles) >= self.config["max_vehicles"]:
            return

        spawn_interval = int(self.config.get("spawn_interval", 1))
        if spawn_interval > 1 and (self._spawn_step % spawn_interval != 0):
            return

        if self.np_random.uniform() > self.config["spawn_probability"]:
            return

        other_vehicles_type = getattr(self, "_other_vehicles_type", None)
        if other_vehicles_type is None:
            other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        ahead_prob = float(self.config.get("spawn_ahead_probability", 0.5))
        spawn_ahead = self.np_random.uniform() < ahead_prob

        vehicle = self._make_vehicle_near_ego(other_vehicles_type, ahead=spawn_ahead)
        if vehicle is not None:
            self.road.vehicles.append(vehicle)

    def _make_vehicle_near_ego(self, vehicle_class, ahead: bool):
        """
        Spawn one vehicle in a random lane either ahead of or behind the ego.
        """
        ego_x = float(self.vehicle.position[0])
        lanes_count = self.config["lanes_count"]

        for _ in range(20):
            lane_id = int(self.np_random.integers(lanes_count))
            lane = self.road.network.get_lane(("0", "1", lane_id))

            if ahead:
                delta_x = self.np_random.uniform(25.0, 90.0)
                x = ego_x + delta_x
            else:
                delta_x = self.np_random.uniform(20.0, 70.0)
                x = ego_x - delta_x

            if x < 0:
                continue

            speed = self.np_random.uniform(20.0, 28.0)

            vehicle = vehicle_class(
                self.road,
                lane.position(x, 0),
                lane.heading_at(x),
                speed,
            )
            vehicle.randomize_behavior()
            self._set_vehicle_speed(vehicle)

            if self._space_is_free(vehicle):
                return vehicle

        return None

    def _space_is_free(self, candidate) -> bool:
        """
        Reject vehicles that would spawn too close to existing vehicles.
        """
        min_gap = float(self.config["spawn_min_gap"])
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

    def _set_vehicle_speed(self, vehicle) -> None:
        min_speed, max_speed = self._speed_range()
        target_speed = float(self.np_random.uniform(min_speed, max_speed))

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
