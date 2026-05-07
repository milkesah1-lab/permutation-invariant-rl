from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\milke\miniconda3\envs\highway\python.exe")

DEFAULT_RUN_ID = f"meanmax_vs_uncertainty_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_ID = os.environ.get("RUN_ID", DEFAULT_RUN_ID)
CONFIG_NAME = os.environ.get("HIGHWAY_CONFIG", "curriculum_stage_3_mixed_traffic")
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "12345"))
EVAL_SEEDS = list(range(1000, 1050))

OUT_DIR = ROOT / "experiment_runs" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)


CHILD_CODE = r"""
import csv
import json
import os
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from gymnasium.envs.registration import register, registry
from gymnasium.wrappers import FlattenObservation

from highway_configs import get_highway_config
from ppo import PPO


out_dir = Path(os.environ["OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
config_name = os.environ["CONFIG_NAME"]
model_label = os.environ["MODEL_LABEL"]
train_seed = int(os.environ["TRAIN_SEED"])
eval_seeds = json.loads(os.environ["EVAL_SEEDS"])
hyperparameters = json.loads(os.environ["HYPERPARAMETERS"])
total_timesteps = int(os.environ["TOTAL_TIMESTEPS"])
use_flatten = os.environ["USE_FLATTEN"] == "1"

torch.manual_seed(train_seed)
np.random.seed(train_seed)

if "continuous-spawn-highway-v0" not in registry:
    register(
        id="continuous-spawn-highway-v0",
        entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
    )


def make_env():
    env = gym.make(
        "continuous-spawn-highway-v0",
        config=get_highway_config(config_name),
    )
    if use_flatten:
        env = FlattenObservation(env)
    return env


env = make_env()
env.reset(seed=train_seed)
env.action_space.seed(train_seed)
env.observation_space.seed(train_seed)

model = PPO(env, **hyperparameters)
training_csv = out_dir / f"{model_label}_training.csv"
model.set_csv_log_path(str(training_csv))
model.learn(total_timesteps=total_timesteps)

actor_path = out_dir / f"{model_label}_actor.pth"
critic_path = out_dir / f"{model_label}_critic.pth"
torch.save(model.actor.state_dict(), actor_path)
torch.save(model.critic.state_dict(), critic_path)
env.close()

policy = model.actor
policy.eval()

per_episode_rows = []
eval_env = make_env()

with torch.no_grad():
    for seed in eval_seeds:
        obs, _ = eval_env.reset(seed=seed)
        done = False
        collided = False
        episodic_return = 0.0
        episodic_length = 0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            mean = policy(obs_t)
            action = mean.detach().cpu().numpy()
            action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)

            obs, rew, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episodic_return += float(rew)
            episodic_length += 1

            if eval_env.unwrapped.vehicle.crashed:
                collided = True

        per_episode_rows.append(
            {
                "model": model_label,
                "seed": int(seed),
                "episodic_return": float(episodic_return),
                "episodic_length": int(episodic_length),
                "collided": int(collided),
            }
        )

eval_env.close()

per_episode_csv = out_dir / f"{model_label}_per_episode.csv"
with per_episode_csv.open("w", newline="") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["model", "seed", "episodic_return", "episodic_length", "collided"],
    )
    writer.writeheader()
    writer.writerows(per_episode_rows)

avg_return = float(np.mean([row["episodic_return"] for row in per_episode_rows]))
avg_length = float(np.mean([row["episodic_length"] for row in per_episode_rows]))
collision_rate = float(np.mean([row["collided"] for row in per_episode_rows]))

summary = {
    "model": model_label,
    "config_name": config_name,
    "total_timesteps": total_timesteps,
    "train_seed": train_seed,
    "eval_episodes": len(eval_seeds),
    "avg_episodic_return": avg_return,
    "avg_episodic_length": avg_length,
    "collision_rate": collision_rate,
    "training_csv": str(training_csv),
    "per_episode_csv": str(per_episode_csv),
    "actor_path": str(actor_path),
    "critic_path": str(critic_path),
}

summary_path = out_dir / f"{model_label}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
"""


MODELS = [
    {
        "label": "deep_sets_mean_max_baseline",
        "cwd": ROOT / "mean+max",
        "use_flatten": True,
        "hyperparameters": {
            "timesteps_per_batch": 4096,
            "max_timesteps_per_episode": 300,
            "gamma": 0.99,
            "n_updates_per_iteration": 5,
            "lr": 1e-4,
            "clip": 0.2,
            "fixed_cov_var": 0.08,
        },
    },
    {
        "label": "uncertainty_aware_lambda_0.01",
        "cwd": ROOT / "uncertainty_aware_PPO",
        "use_flatten": False,
        "hyperparameters": {
            "timesteps_per_batch": 4096,
            "max_timesteps_per_episode": 300,
            "gamma": 0.99,
            "n_updates_per_iteration": 5,
            "lr": 1e-4,
            "clip": 0.2,
            "dropout_p": 0.1,
            "mc_samples": 5,
            "lambda_u": 0.01,
            "fixed_cov_var": 0.08,
        },
    },
]


def load_last_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows[-1]


def run_model(model: dict) -> dict:
    model_out_dir = OUT_DIR / model["label"]
    model_out_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_out_dir / "run.log"

    env = os.environ.copy()
    env.update(
        {
            "OUT_DIR": str(model_out_dir),
            "CONFIG_NAME": CONFIG_NAME,
            "MODEL_LABEL": model["label"],
            "TRAIN_SEED": str(TRAIN_SEED),
            "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
            "HYPERPARAMETERS": json.dumps(model["hyperparameters"]),
            "TOTAL_TIMESTEPS": str(TOTAL_TIMESTEPS),
            "USE_FLATTEN": "1" if model["use_flatten"] else "0",
        }
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [str(PYTHON), "-c", CHILD_CODE],
            cwd=str(model["cwd"]),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    summary_path = model_out_dir / f"{model['label']}_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["log_path"] = str(log_path)

    last_training_row = load_last_row(Path(summary["training_csv"]))
    summary["final_training_iteration"] = {
        key: last_training_row[key] for key in last_training_row.keys()
    }
    return summary


def write_combined_outputs(results: list[dict]) -> None:
    summary_csv = OUT_DIR / "summary.csv"
    with summary_csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "model",
                "config_name",
                "total_timesteps",
                "train_seed",
                "eval_episodes",
                "avg_episodic_return",
                "avg_episodic_length",
                "collision_rate",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["model"],
                    result["config_name"],
                    result["total_timesteps"],
                    result["train_seed"],
                    result["eval_episodes"],
                    round(result["avg_episodic_return"], 6),
                    round(result["avg_episodic_length"], 6),
                    round(result["collision_rate"], 6),
                ]
            )

    training_summary_csv = OUT_DIR / "training_final_summary.csv"
    with training_summary_csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "model",
                "iteration",
                "timesteps_so_far",
                "avg_episodic_length",
                "avg_episodic_return",
                "avg_loss",
                "avg_raw_episodic_return",
                "avg_critic_uncertainty",
                "iteration_seconds",
            ]
        )
        for result in results:
            row = result["final_training_iteration"]
            writer.writerow(
                [
                    result["model"],
                    row.get("iteration", ""),
                    row.get("timesteps_so_far", ""),
                    row.get("avg_episodic_length", ""),
                    row.get("avg_episodic_return", ""),
                    row.get("avg_loss", ""),
                    row.get("avg_raw_episodic_return", ""),
                    row.get("avg_critic_uncertainty", ""),
                    row.get("iteration_seconds", ""),
                ]
            )

    per_episode_csv = OUT_DIR / "per_episode.csv"
    with per_episode_csv.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["model", "seed", "episodic_return", "episodic_length", "collided"])
        for result in results:
            model_per_episode = Path(result["per_episode_csv"])
            with model_per_episode.open(newline="") as model_csv_file:
                reader = csv.DictReader(model_csv_file)
                for row in reader:
                    writer.writerow(
                        [
                            row["model"],
                            row["seed"],
                            row["episodic_return"],
                            row["episodic_length"],
                            row["collided"],
                        ]
                    )

    config_json = OUT_DIR / "run_config.json"
    config_json.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "config_name": CONFIG_NAME,
                "total_timesteps": TOTAL_TIMESTEPS,
                "train_seed": TRAIN_SEED,
                "eval_seeds": EVAL_SEEDS,
                "models": [
                    {
                        "label": model["label"],
                        "cwd": str(model["cwd"]),
                        "use_flatten": model["use_flatten"],
                        "hyperparameters": model["hyperparameters"],
                    }
                    for model in MODELS
                ],
            },
            indent=2,
        )
    )


def main() -> int:
    print(f"Run ID: {RUN_ID}")
    print(f"Output directory: {OUT_DIR}")
    results = []
    for model in MODELS:
        print(f"Running {model['label']}...")
        results.append(run_model(model))

    write_combined_outputs(results)

    print("Completed experiment.")
    print(f"Summary CSV: {OUT_DIR / 'summary.csv'}")
    print(f"Training summary CSV: {OUT_DIR / 'training_final_summary.csv'}")
    print(f"Per-episode CSV: {OUT_DIR / 'per_episode.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
