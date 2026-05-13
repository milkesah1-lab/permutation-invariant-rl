from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_SUBDIR = os.environ.get("MODEL_SUBDIR", "mean+max")
MODEL_DIR = ROOT / MODEL_SUBDIR

DEFAULT_PYTHON = Path(r"C:\Users\milke\miniconda3\envs\highway\python.exe")
PYTHON = Path(os.environ.get("PYTHON_EXE", DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else sys.executable))

DEFAULT_RUN_ID = f"meanmax_highwayv0_stage2_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_ID = os.environ.get("RUN_ID", DEFAULT_RUN_ID)
OUT_DIR = ROOT / "experiment_runs" / RUN_ID

ENV_ID = "highway-v0"
MODEL_LABEL = os.environ.get("MODEL_LABEL", "deep_sets_mean_max")
CONFIG_NAME = os.environ.get("HIGHWAY_CONFIG", "curriculum_stage_2_easy_overtake")
TARGET_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "300000"))
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "12345"))
EVAL_SEEDS = json.loads(os.environ.get("EVAL_SEEDS", json.dumps(list(range(1000, 1050)))))
EVAL_MAX_STEPS_RAW = os.environ.get("EVAL_MAX_STEPS", "").strip().lower()
EVAL_MAX_STEPS = None if EVAL_MAX_STEPS_RAW in {"", "none", "full"} else int(EVAL_MAX_STEPS_RAW)
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
EVAL_ACTION_MODE = os.environ.get("EVAL_ACTION_MODE", "actor")
SAFEGUARD_THRESHOLD = float(os.environ.get("SAFEGUARD_THRESHOLD", "0.001"))
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "5"))

HYPERPARAMETERS = {
    "timesteps_per_batch": 4096,
    "max_timesteps_per_episode": 300,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 1e-4,
    "clip": 0.2,
    "fixed_cov_var": 0.08,
}


CHILD_CODE = r"""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation

from highway_configs import get_highway_config
from ppo import PPO


HIGHWAY_V0_STAGE_2_BENCHMARK = {
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
    "vehicles_count": 45,
    "vehicles_density": 1.6,
    "controlled_vehicles": 1,
    "duration": 40,
    "policy_frequency": 15,
    "simulation_frequency": 15,
    "initial_lane_id": None,
    "ego_spacing": 1.5,
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
    "urgency_max_distance": 80.0,
    "urgency_speed_scale": 15.0,
    "urgency_ttc_horizon": 6.0,
}


HIGHWAY_V0_STAGE_2_BENCHMARK_V2 = {
    **HIGHWAY_V0_STAGE_2_BENCHMARK,
    "vehicles_count": 35,
    "vehicles_density": 1.2,
}


HIGHWAY_V0_STAGE_2_BENCHMARK_V3 = {
    **HIGHWAY_V0_STAGE_2_BENCHMARK,
    "vehicles_count": 30,
    "vehicles_density": 1.0,
}


HIGHWAY_V0_STAGE_2_BENCHMARK_V4 = {
    **HIGHWAY_V0_STAGE_2_BENCHMARK,
    "vehicles_count": 27,
    "vehicles_density": 1.0,
}


HIGHWAY_V0_STAGE_2_BENCHMARK_V45 = {
    **HIGHWAY_V0_STAGE_2_BENCHMARK,
    "vehicles_count": 29,
    "vehicles_density": 1.0,
}


def resolve_config(name: str) -> dict:
    if name == "highway_v0_stage_2_benchmark":
        return json.loads(json.dumps(HIGHWAY_V0_STAGE_2_BENCHMARK))
    if name == "highway_v0_stage_2_benchmark_v2":
        return json.loads(json.dumps(HIGHWAY_V0_STAGE_2_BENCHMARK_V2))
    if name == "highway_v0_stage_2_benchmark_v3":
        return json.loads(json.dumps(HIGHWAY_V0_STAGE_2_BENCHMARK_V3))
    if name == "highway_v0_stage_2_benchmark_v4":
        return json.loads(json.dumps(HIGHWAY_V0_STAGE_2_BENCHMARK_V4))
    if name == "highway_v0_stage_2_benchmark_v45":
        return json.loads(json.dumps(HIGHWAY_V0_STAGE_2_BENCHMARK_V45))
    return get_highway_config(name)


train_seed = int(os.environ["TRAIN_SEED"])
out_dir = Path(os.environ["OUT_DIR"])
env_id = os.environ["ENV_ID"]
config_name = os.environ["CONFIG_NAME"]
model_label = os.environ["MODEL_LABEL"]
target_timesteps = int(os.environ["TARGET_TIMESTEPS"])
eval_max_steps_raw = os.environ["EVAL_MAX_STEPS"].strip().lower()
eval_max_steps = None if eval_max_steps_raw in {"", "none", "full"} else int(eval_max_steps_raw)
eval_seeds = json.loads(os.environ["EVAL_SEEDS"])
hyperparameters = json.loads(os.environ["HYPERPARAMETERS"])
torch_num_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
eval_action_mode = os.environ.get("EVAL_ACTION_MODE", "actor")
safeguard_threshold = float(os.environ.get("SAFEGUARD_THRESHOLD", "0.001"))
mc_samples = int(os.environ.get("MC_SAMPLES", "5"))

out_dir.mkdir(parents=True, exist_ok=True)

random.seed(train_seed)
np.random.seed(train_seed)
torch.manual_seed(train_seed)
torch.set_num_threads(torch_num_threads)
try:
    torch.set_num_interop_threads(torch_num_threads)
except RuntimeError:
    pass
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HighwayUrgencyObservation(gym.ObservationWrapper):
    # Append the 8 urgency features expected by the mean+max network.

    def __init__(self, env):
        super().__init__(env)
        base_shape = self.observation_space.shape
        if len(base_shape) != 1:
            raise ValueError(f"Expected flattened observation, got shape {base_shape}")
        self.urgency_dim = 8
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(base_shape[0]) + self.urgency_dim,),
            dtype=np.float32,
        )

    def observation(self, obs):
        obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
        return np.concatenate([obs_array, self._compute_urgency_features()]).astype(np.float32)

    def _compute_urgency_features(self) -> np.ndarray:
        env = self.unwrapped
        max_gap = float(env.config.get("urgency_max_distance", 80.0))
        speed_scale = float(env.config.get("urgency_speed_scale", 15.0))
        ttc_horizon = float(env.config.get("urgency_ttc_horizon", 6.0))
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

        return np.array(
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

    def _ego_lane_id(self):
        vehicle = getattr(self.unwrapped, "vehicle", None)
        if vehicle is None:
            return None
        try:
            return int(vehicle.lane_index[2])
        except Exception:
            return None

    def _adjacent_lane_exists(self, offset: int) -> bool:
        ego_lane = self._ego_lane_id()
        if ego_lane is None:
            return False
        return 0 <= ego_lane + offset < int(self.unwrapped.config["lanes_count"])

    def _nearest_vehicle_in_lane(self, direction: str):
        ego_lane = self._ego_lane_id()
        if ego_lane is None:
            return np.inf, None
        return self._nearest_vehicle_by_lane_id(ego_lane, direction)

    def _nearest_vehicle_in_adjacent_lane(self, offset: int, direction: str):
        ego_lane = self._ego_lane_id()
        if ego_lane is None or not self._adjacent_lane_exists(offset):
            return np.inf, None
        return self._nearest_vehicle_by_lane_id(ego_lane + offset, direction)

    def _nearest_vehicle_by_lane_id(self, lane_id: int, direction: str):
        env = self.unwrapped
        ego = getattr(env, "vehicle", None)
        road = getattr(env, "road", None)
        if ego is None or road is None:
            return np.inf, None

        ego_x = float(ego.position[0])
        best_gap = np.inf
        best_vehicle = None
        for vehicle in road.vehicles:
            if vehicle is ego:
                continue
            try:
                vehicle_lane = int(vehicle.lane_index[2])
            except Exception:
                continue
            if vehicle_lane != lane_id:
                continue
            dx = float(vehicle.position[0]) - ego_x
            if direction == "front" and 0.0 < dx < best_gap:
                best_gap = dx
                best_vehicle = vehicle
            if direction == "rear" and dx < 0.0 and -dx < best_gap:
                best_gap = -dx
                best_vehicle = vehicle
        return best_gap, best_vehicle

    def _closing_speed_to_vehicle(self, vehicle) -> float:
        ego = getattr(self.unwrapped, "vehicle", None)
        if ego is None or vehicle is None:
            return 0.0
        return max(float(ego.speed - vehicle.speed), 0.0)

    @staticmethod
    def _ttc_risk(gap: float, closing_speed: float, ttc_horizon: float) -> float:
        if not np.isfinite(gap) or gap <= 0.0 or closing_speed <= 1e-6:
            return 0.0
        ttc = gap / closing_speed
        return float(np.clip(1.0 - (ttc / ttc_horizon), 0.0, 1.0))

    @staticmethod
    def _normalize_gap(gap: float, max_gap: float) -> float:
        if not np.isfinite(gap):
            return 1.0
        return float(np.clip(gap / max_gap, 0.0, 1.0))


def make_env(seed: int | None = None):
    config = resolve_config(config_name)
    env = gym.make(env_id, config=config)
    env = FlattenObservation(env)
    env = HighwayUrgencyObservation(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def load_last_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No training rows found in {csv_path}")
    return rows[-1]


def start_lane(env) -> int | str:
    try:
        return int(env.unwrapped.vehicle.lane_index[2])
    except Exception:
        return "unknown"


run_config_path = out_dir / "run_config.json"
probe_env = make_env(seed=train_seed)
probe_obs_shape = tuple(probe_env.observation_space.shape)
probe_base_config = dict(probe_env.unwrapped.config)
probe_env.close()
run_config_path.write_text(
    json.dumps(
        {
            "model": model_label,
            "env_id": env_id,
            "config_name": config_name,
            "train_seed": train_seed,
            "target_timesteps": target_timesteps,
            "eval_episodes": len(eval_seeds),
            "eval_seeds": eval_seeds,
            "eval_max_steps": eval_max_steps,
            "eval_type": "full" if eval_max_steps is None else "capped",
            "hyperparameters": hyperparameters,
            "observation_space_shape": probe_obs_shape,
            "urgency_wrapper": "HighwayUrgencyObservation",
            "duration": probe_base_config.get("duration"),
            "policy_frequency": probe_base_config.get("policy_frequency"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch_num_threads,
        "eval_action_mode": eval_action_mode,
        "safeguard_threshold": safeguard_threshold if eval_action_mode == "safeguarded" else None,
        "mc_samples": mc_samples if eval_action_mode == "safeguarded" else None,
    },
    indent=2,
),
    encoding="utf-8",
)

print(f"Env: {env_id}", flush=True)
print(f"Config: {config_name}", flush=True)
print(f"Training seed: {train_seed}", flush=True)
print(f"Target timesteps: {target_timesteps}", flush=True)
print(f"Eval type: {'full' if eval_max_steps is None else f'capped at {eval_max_steps}'}", flush=True)
print(f"Eval action mode: {eval_action_mode}", flush=True)
if eval_action_mode == "safeguarded":
    print(f"Safeguard threshold: {safeguard_threshold}", flush=True)
    print(f"MC samples: {mc_samples}", flush=True)
print(f"Observation space: {probe_obs_shape}", flush=True)
print(f"Evaluation seeds: {eval_seeds[0]}..{eval_seeds[-1]} ({len(eval_seeds)} episodes)", flush=True)
print(f"Output directory: {out_dir}", flush=True)

env = make_env(seed=train_seed)
hyperparameters = dict(hyperparameters)
hyperparameters["seed"] = train_seed

model = PPO(env, **hyperparameters)
model.actor.to(model.device)
model.critic.to(model.device)

training_csv = out_dir / "training_log.csv"
model.set_csv_log_path(str(training_csv))
model.learn(total_timesteps=target_timesteps)

actor_path = out_dir / "checkpoint_actor.pth"
critic_path = out_dir / "checkpoint_critic.pth"
torch.save(model.actor.state_dict(), actor_path)
torch.save(model.critic.state_dict(), critic_path)
env.close()

last_training = load_last_row(training_csv)
training_summary_csv = out_dir / "training_summary.csv"
with training_summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
    fieldnames = [
        "model",
        "env_id",
        "train_seed",
        "config_name",
        "target_timesteps",
        "iteration",
        "timesteps_so_far",
        "avg_episodic_return",
        "avg_episodic_length",
        "avg_loss",
        "iteration_seconds",
        "training_csv",
        "actor_checkpoint",
        "critic_checkpoint",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(
        {
            "model": model_label,
            "env_id": env_id,
            "train_seed": train_seed,
            "config_name": config_name,
            "target_timesteps": target_timesteps,
            "iteration": last_training.get("iteration", ""),
            "timesteps_so_far": last_training.get("timesteps_so_far", ""),
            "avg_episodic_return": last_training.get("avg_episodic_return", ""),
            "avg_episodic_length": last_training.get("avg_episodic_length", ""),
            "avg_loss": last_training.get("avg_loss", ""),
            "iteration_seconds": last_training.get("iteration_seconds", ""),
            "training_csv": str(training_csv),
            "actor_checkpoint": str(actor_path),
            "critic_checkpoint": str(critic_path),
        }
    )

policy = model.actor
policy.eval()
device = next(policy.parameters()).device

per_episode_rows = []
eval_env = make_env()
eval_env.action_space.seed(0)
eval_env.observation_space.seed(0)

with torch.no_grad():
    for episode_index, seed in enumerate(eval_seeds, start=1):
        obs, _ = eval_env.reset(seed=int(seed))
        done = False
        collided = False
        episodic_return = 0.0
        episodic_length = 0
        capped = False
        terminated = False
        truncated = False
        lane = start_lane(eval_env)
        uncertainties = []
        activation_count = 0

        while not done:
            if eval_action_mode == "safeguarded":
                action, uncertainty, activated = model.get_safeguarded_action(
                    obs,
                    threshold=safeguard_threshold,
                    mc_samples=mc_samples,
                )
                uncertainties.append(float(uncertainty))
                if activated:
                    activation_count += 1
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action = policy(obs_t).detach().cpu().numpy()
                action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)

            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = bool(terminated or truncated)
            episodic_return += float(reward)
            episodic_length += 1

            if eval_env.unwrapped.vehicle.crashed:
                collided = True

            if eval_max_steps is not None and episodic_length >= eval_max_steps:
                capped = True
                done = True

        per_episode_rows.append(
            {
                "model": model_label,
                "env_id": env_id,
                "train_seed": train_seed,
                "episode_index": episode_index,
                "eval_seed": int(seed),
                "start_lane": lane,
                "episodic_return": episodic_return,
                "episodic_length": episodic_length,
                "collided": int(collided),
                "capped": int(capped),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "eval_action_mode": eval_action_mode,
                "safeguard_threshold": safeguard_threshold if eval_action_mode == "safeguarded" else "",
                "mc_samples": mc_samples if eval_action_mode == "safeguarded" else "",
                "avg_uncertainty": float(np.mean(uncertainties)) if uncertainties else "",
                "max_uncertainty": float(np.max(uncertainties)) if uncertainties else "",
                "activation_count": activation_count if eval_action_mode == "safeguarded" else "",
                "activation_rate": float(activation_count / episodic_length) if uncertainties and episodic_length else "",
            }
        )

eval_env.close()

per_episode_csv = out_dir / "evaluation_per_episode.csv"
with per_episode_csv.open("w", newline="", encoding="utf-8") as csv_file:
    fieldnames = [
        "model",
        "env_id",
        "train_seed",
        "episode_index",
        "eval_seed",
        "start_lane",
        "episodic_return",
        "episodic_length",
        "collided",
        "capped",
        "terminated",
        "truncated",
        "eval_action_mode",
        "safeguard_threshold",
        "mc_samples",
        "avg_uncertainty",
        "max_uncertainty",
        "activation_count",
        "activation_rate",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(per_episode_rows)

avg_return = float(np.mean([row["episodic_return"] for row in per_episode_rows]))
avg_length = float(np.mean([row["episodic_length"] for row in per_episode_rows]))
collision_rate = float(np.mean([row["collided"] for row in per_episode_rows]))
lanes = [row["start_lane"] for row in per_episode_rows]
lane_counts = {str(lane): lanes.count(lane) for lane in sorted(set(lanes), key=str)}
uncertainty_rows = [row for row in per_episode_rows if row["avg_uncertainty"] != ""]
avg_uncertainty = float(np.mean([row["avg_uncertainty"] for row in uncertainty_rows])) if uncertainty_rows else ""
avg_max_uncertainty = float(np.mean([row["max_uncertainty"] for row in uncertainty_rows])) if uncertainty_rows else ""
avg_activations = float(np.mean([row["activation_count"] for row in uncertainty_rows])) if uncertainty_rows else ""
avg_activation_rate = float(np.mean([row["activation_rate"] for row in uncertainty_rows])) if uncertainty_rows else ""
total_activations = int(np.sum([row["activation_count"] for row in uncertainty_rows])) if uncertainty_rows else ""

evaluation_summary_csv = out_dir / "evaluation_summary.csv"
with evaluation_summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
    fieldnames = [
        "model",
        "env_id",
        "train_seed",
        "config_name",
        "eval_episodes",
        "eval_type",
        "eval_max_steps",
        "avg_episodic_return",
        "avg_episodic_length",
        "collision_rate",
        "lane_counts",
        "eval_action_mode",
        "safeguard_threshold",
        "mc_samples",
        "avg_uncertainty",
        "avg_max_uncertainty",
        "avg_activations_per_episode",
        "avg_activation_rate",
        "total_activations",
        "per_episode_csv",
        "actor_checkpoint",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(
        {
            "model": model_label,
            "env_id": env_id,
            "train_seed": train_seed,
            "config_name": config_name,
            "eval_episodes": len(eval_seeds),
            "eval_type": "full" if eval_max_steps is None else "capped",
            "eval_max_steps": "" if eval_max_steps is None else eval_max_steps,
            "avg_episodic_return": round(avg_return, 6),
            "avg_episodic_length": round(avg_length, 6),
            "collision_rate": round(collision_rate, 6),
            "lane_counts": json.dumps(lane_counts),
            "eval_action_mode": eval_action_mode,
            "safeguard_threshold": safeguard_threshold if eval_action_mode == "safeguarded" else "",
            "mc_samples": mc_samples if eval_action_mode == "safeguarded" else "",
            "avg_uncertainty": round(avg_uncertainty, 8) if avg_uncertainty != "" else "",
            "avg_max_uncertainty": round(avg_max_uncertainty, 8) if avg_max_uncertainty != "" else "",
            "avg_activations_per_episode": round(avg_activations, 6) if avg_activations != "" else "",
            "avg_activation_rate": round(avg_activation_rate, 8) if avg_activation_rate != "" else "",
            "total_activations": total_activations,
            "per_episode_csv": str(per_episode_csv),
            "actor_checkpoint": str(actor_path),
        }
    )

summary_json = out_dir / "run_summary.json"
summary_json.write_text(
    json.dumps(
        {
            "training": {
                "summary_csv": str(training_summary_csv),
                "training_csv": str(training_csv),
                "final_training_return": float(last_training.get("avg_episodic_return", "nan")),
                "final_training_episode_length": float(last_training.get("avg_episodic_length", "nan")),
                "final_training_timesteps": int(float(last_training.get("timesteps_so_far", "0"))),
            },
            "evaluation": {
                "summary_csv": str(evaluation_summary_csv),
                "per_episode_csv": str(per_episode_csv),
                "avg_episodic_return": avg_return,
                "avg_episodic_length": avg_length,
                "collision_rate": collision_rate,
                "lane_counts": lane_counts,
                "eval_action_mode": eval_action_mode,
                "safeguard_threshold": safeguard_threshold if eval_action_mode == "safeguarded" else None,
                "mc_samples": mc_samples if eval_action_mode == "safeguarded" else None,
                "avg_uncertainty": avg_uncertainty,
                "avg_max_uncertainty": avg_max_uncertainty,
                "avg_activations_per_episode": avg_activations,
                "avg_activation_rate": avg_activation_rate,
                "total_activations": total_activations,
            },
            "checkpoints": {
                "actor": str(actor_path),
                "critic": str(critic_path),
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Training CSV: {training_csv}", flush=True)
print(f"Training summary CSV: {training_summary_csv}", flush=True)
print(f"Evaluation per-episode CSV: {per_episode_csv}", flush=True)
print(f"Evaluation summary CSV: {evaluation_summary_csv}", flush=True)
print(f"Actor checkpoint: {actor_path}", flush=True)
print(f"Critic checkpoint: {critic_path}", flush=True)
"""


def main() -> int:
    if len(EVAL_SEEDS) != 50:
        raise ValueError(f"Expected 50 evaluation seeds, got {len(EVAL_SEEDS)}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUT_DIR / "study_config.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "model": MODEL_LABEL,
                "model_dir": str(MODEL_DIR),
                "env_id": ENV_ID,
                "config_name": CONFIG_NAME,
                "target_timesteps": TARGET_TIMESTEPS,
                "train_seed": TRAIN_SEED,
                "eval_seeds": EVAL_SEEDS,
                "eval_max_steps": EVAL_MAX_STEPS,
                "eval_type": "full" if EVAL_MAX_STEPS is None else "capped",
                "hyperparameters": HYPERPARAMETERS,
                "urgency_wrapper": "HighwayUrgencyObservation",
                "torch_num_threads": TORCH_NUM_THREADS,
                "eval_action_mode": EVAL_ACTION_MODE,
                "safeguard_threshold": SAFEGUARD_THRESHOLD if EVAL_ACTION_MODE == "safeguarded" else None,
                "mc_samples": MC_SAMPLES if EVAL_ACTION_MODE == "safeguarded" else None,
                "python": str(PYTHON),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"Output directory: {OUT_DIR}", flush=True)
    print(f"Python: {PYTHON}", flush=True)
    print(f"Model directory: {MODEL_DIR}", flush=True)
    print(f"Env: {ENV_ID}", flush=True)
    print(f"Config: {CONFIG_NAME}", flush=True)
    print(f"Training seed: {TRAIN_SEED}", flush=True)
    print(f"Eval type: {'full' if EVAL_MAX_STEPS is None else f'capped at {EVAL_MAX_STEPS}'}", flush=True)
    print(f"Eval action mode: {EVAL_ACTION_MODE}", flush=True)
    if EVAL_ACTION_MODE == "safeguarded":
        print(f"Safeguard threshold: {SAFEGUARD_THRESHOLD}", flush=True)
        print(f"MC samples: {MC_SAMPLES}", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OUT_DIR": str(OUT_DIR),
            "ENV_ID": ENV_ID,
            "TRAIN_SEED": str(TRAIN_SEED),
            "CONFIG_NAME": CONFIG_NAME,
            "MODEL_LABEL": MODEL_LABEL,
            "TARGET_TIMESTEPS": str(TARGET_TIMESTEPS),
            "EVAL_MAX_STEPS": "" if EVAL_MAX_STEPS is None else str(EVAL_MAX_STEPS),
            "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
            "HYPERPARAMETERS": json.dumps(HYPERPARAMETERS),
            "TORCH_NUM_THREADS": str(TORCH_NUM_THREADS),
            "OMP_NUM_THREADS": str(TORCH_NUM_THREADS),
            "MKL_NUM_THREADS": str(TORCH_NUM_THREADS),
            "EVAL_ACTION_MODE": EVAL_ACTION_MODE,
            "SAFEGUARD_THRESHOLD": str(SAFEGUARD_THRESHOLD),
            "MC_SAMPLES": str(MC_SAMPLES),
        }
    )

    log_path = OUT_DIR / "training_stdout.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [str(PYTHON), "-c", CHILD_CODE],
            cwd=str(MODEL_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print("Completed highway-v0 mean+max study.", flush=True)
    print(f"Training/eval stdout: {log_path}", flush=True)
    print(f"Training summary: {OUT_DIR / 'training_summary.csv'}", flush=True)
    print(f"Evaluation summary: {OUT_DIR / 'evaluation_summary.csv'}", flush=True)
    print(f"Per-episode evaluation: {OUT_DIR / 'evaluation_per_episode.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
