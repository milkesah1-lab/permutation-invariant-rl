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

DEFAULT_RUN_ID = f"meanmax_single_stage2_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_ID = os.environ.get("RUN_ID", DEFAULT_RUN_ID)
OUT_DIR = ROOT / "experiment_runs" / RUN_ID
COMMAND_HEADER = os.environ.get("COMMAND_HEADER", "").strip()

MODEL_LABEL = os.environ.get("MODEL_LABEL", "deep_sets_mean_max")
CONFIG_NAME = os.environ.get("HIGHWAY_CONFIG", "curriculum_stage_2_easy_overtake")
TARGET_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "300000"))
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "12345"))
EVAL_SEEDS = json.loads(os.environ.get("EVAL_SEEDS", json.dumps(list(range(1000, 1050)))))
EVAL_MAX_STEPS_RAW = os.environ.get("EVAL_MAX_STEPS", "").strip().lower()
EVAL_MAX_STEPS = None if EVAL_MAX_STEPS_RAW in {"", "none", "full"} else int(EVAL_MAX_STEPS_RAW)
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
EGO_START_LANE_POLICY = os.environ.get("EGO_START_LANE_POLICY", "random")
CONFIG_OVERRIDES = json.loads(os.environ.get("CONFIG_OVERRIDES", "{}"))
EVAL_ACTION_MODE = os.environ.get("EVAL_ACTION_MODE", "actor").strip().lower()
SAFEGUARD_THRESHOLD = float(os.environ.get("SAFEGUARD_THRESHOLD", "0.0005"))
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

if "LAMBDA_U" in os.environ:
    HYPERPARAMETERS["lambda_u"] = float(os.environ["LAMBDA_U"])
if "DROPOUT_P" in os.environ:
    HYPERPARAMETERS["dropout_p"] = float(os.environ["DROPOUT_P"])
if "PPO_MC_SAMPLES" in os.environ:
    HYPERPARAMETERS["mc_samples"] = int(os.environ["PPO_MC_SAMPLES"])


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
from gymnasium.envs.registration import register, registry
from gymnasium.wrappers import FlattenObservation

from highway_configs import get_highway_config
from ppo import PPO


def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


train_seed = int(os.environ["TRAIN_SEED"])
out_dir = Path(os.environ["OUT_DIR"])
config_name = os.environ["CONFIG_NAME"]
model_label = os.environ["MODEL_LABEL"]
target_timesteps = int(os.environ["TARGET_TIMESTEPS"])
eval_max_steps_raw = os.environ["EVAL_MAX_STEPS"].strip().lower()
eval_max_steps = None if eval_max_steps_raw in {"", "none", "full"} else int(eval_max_steps_raw)
eval_seeds = json.loads(os.environ["EVAL_SEEDS"])
hyperparameters = json.loads(os.environ["HYPERPARAMETERS"])
torch_num_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
ego_start_lane_policy = os.environ.get("EGO_START_LANE_POLICY", "random")
config_overrides = json.loads(os.environ.get("CONFIG_OVERRIDES", "{}"))
eval_action_mode = os.environ.get("EVAL_ACTION_MODE", "actor").strip().lower()
safeguard_threshold = float(os.environ.get("SAFEGUARD_THRESHOLD", "0.0005"))
mc_samples = int(os.environ.get("MC_SAMPLES", "5"))

if eval_action_mode not in {"actor", "safeguarded"}:
    raise ValueError(f"Unsupported EVAL_ACTION_MODE={eval_action_mode!r}")
if eval_action_mode == "safeguarded" and not hasattr(PPO, "get_safeguarded_action"):
    raise AttributeError("Safeguarded evaluation requested, but PPO has no get_safeguarded_action method")

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

if "continuous-spawn-highway-v0" not in registry:
    register(
        id="continuous-spawn-highway-v0",
        entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
    )


def controlled_config() -> dict:
    config = get_highway_config(config_name)
    deep_update(config, config_overrides)
    config["ego_start_lane_policy"] = ego_start_lane_policy
    return config


def make_env(seed: int | None = None):
    env = gym.make("continuous-spawn-highway-v0", config=controlled_config())
    env = FlattenObservation(env)
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


run_config_path = out_dir / "run_config.json"
run_config_path.write_text(
    json.dumps(
        {
            "model": model_label,
            "train_seed": train_seed,
            "config_name": config_name,
            "target_timesteps": target_timesteps,
            "eval_episodes": len(eval_seeds),
            "eval_seeds": eval_seeds,
            "eval_max_steps": eval_max_steps,
            "eval_type": "full" if eval_max_steps is None else "capped",
            "eval_action_mode": eval_action_mode,
            "safeguard_threshold": safeguard_threshold if eval_action_mode == "safeguarded" else None,
            "mc_samples": mc_samples if eval_action_mode == "safeguarded" else None,
            "hyperparameters": hyperparameters,
            "ego_start_lane_policy": controlled_config().get("ego_start_lane_policy"),
            "config_overrides": config_overrides,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_num_threads": torch_num_threads,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Training seed: {train_seed}", flush=True)
print(f"Config: {config_name}", flush=True)
print(f"Target timesteps: {target_timesteps}", flush=True)
print(f"Eval type: {'full' if eval_max_steps is None else f'capped at {eval_max_steps}'}", flush=True)
print(f"Eval action mode: {eval_action_mode}", flush=True)
if eval_action_mode == "safeguarded":
    print(f"Safeguard threshold: {safeguard_threshold}", flush=True)
    print(f"MC samples: {mc_samples}", flush=True)
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
        "train_seed",
        "config_name",
        "target_timesteps",
        "iteration",
        "timesteps_so_far",
        "avg_episodic_return",
        "avg_episodic_length",
        "avg_raw_episodic_return",
        "avg_loss",
        "avg_critic_uncertainty",
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
            "train_seed": train_seed,
            "config_name": config_name,
            "target_timesteps": target_timesteps,
            "iteration": last_training.get("iteration", ""),
            "timesteps_so_far": last_training.get("timesteps_so_far", ""),
            "avg_episodic_return": last_training.get("avg_episodic_return", ""),
            "avg_episodic_length": last_training.get("avg_episodic_length", ""),
            "avg_raw_episodic_return": last_training.get("avg_raw_episodic_return", ""),
            "avg_loss": last_training.get("avg_loss", ""),
            "avg_critic_uncertainty": last_training.get("avg_critic_uncertainty", ""),
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
        obs, reset_info = eval_env.reset(seed=int(seed))
        done = False
        collided = False
        episodic_return = 0.0
        episodic_length = 0
        capped = False
        terminated = False
        truncated = False
        scenario = reset_info.get("scenario", "unknown")
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

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated or truncated)
            episodic_return += float(reward)
            episodic_length += 1
            scenario = info.get("scenario", scenario)

            if eval_env.unwrapped.vehicle.crashed:
                collided = True

            if eval_max_steps is not None and episodic_length >= eval_max_steps:
                capped = True
                done = True

        per_episode_rows.append(
            {
                "model": model_label,
                "train_seed": train_seed,
                "episode_index": episode_index,
                "eval_seed": int(seed),
                "episodic_return": episodic_return,
                "episodic_length": episodic_length,
                "collided": int(collided),
                "capped": int(capped),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "scenario": scenario,
                "eval_action_mode": eval_action_mode,
                "safeguard_threshold": safeguard_threshold if eval_action_mode == "safeguarded" else "",
                "mc_samples": mc_samples if eval_action_mode == "safeguarded" else "",
                "avg_uncertainty": float(np.mean(uncertainties)) if uncertainties else "",
                "max_uncertainty": float(np.max(uncertainties)) if uncertainties else "",
                "activation_count": activation_count if eval_action_mode == "safeguarded" else "",
                "activation_rate": (activation_count / episodic_length) if eval_action_mode == "safeguarded" and episodic_length else "",
            }
        )

eval_env.close()

per_episode_csv = out_dir / "evaluation_per_episode.csv"
with per_episode_csv.open("w", newline="", encoding="utf-8") as csv_file:
    fieldnames = [
        "model",
        "train_seed",
        "episode_index",
        "eval_seed",
        "episodic_return",
        "episodic_length",
        "collided",
        "capped",
        "terminated",
        "truncated",
        "scenario",
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
uncertainty_rows = [row for row in per_episode_rows if row["avg_uncertainty"] != ""]
avg_uncertainty = float(np.mean([row["avg_uncertainty"] for row in uncertainty_rows])) if uncertainty_rows else ""
avg_max_uncertainty = float(np.mean([row["max_uncertainty"] for row in uncertainty_rows])) if uncertainty_rows else ""
avg_activations_per_episode = (
    float(np.mean([row["activation_count"] for row in uncertainty_rows])) if uncertainty_rows else ""
)
avg_activation_rate = float(np.mean([row["activation_rate"] for row in uncertainty_rows])) if uncertainty_rows else ""
total_activations = int(sum(row["activation_count"] for row in uncertainty_rows)) if uncertainty_rows else ""

evaluation_summary_csv = out_dir / "evaluation_summary.csv"
with evaluation_summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
    fieldnames = [
        "model",
        "train_seed",
        "config_name",
        "eval_episodes",
        "eval_type",
        "eval_max_steps",
        "eval_action_mode",
        "safeguard_threshold",
        "mc_samples",
        "avg_episodic_return",
        "avg_episodic_length",
        "collision_rate",
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
            "train_seed": train_seed,
            "config_name": config_name,
            "eval_episodes": len(eval_seeds),
            "eval_type": "full" if eval_max_steps is None else "capped",
            "eval_max_steps": "" if eval_max_steps is None else eval_max_steps,
            "eval_action_mode": eval_action_mode,
            "safeguard_threshold": safeguard_threshold if eval_action_mode == "safeguarded" else "",
            "mc_samples": mc_samples if eval_action_mode == "safeguarded" else "",
            "avg_episodic_return": round(avg_return, 6),
            "avg_episodic_length": round(avg_length, 6),
            "collision_rate": round(collision_rate, 6),
            "avg_uncertainty": round(avg_uncertainty, 6) if avg_uncertainty != "" else "",
            "avg_max_uncertainty": round(avg_max_uncertainty, 6) if avg_max_uncertainty != "" else "",
            "avg_activations_per_episode": round(avg_activations_per_episode, 6) if avg_activations_per_episode != "" else "",
            "avg_activation_rate": round(avg_activation_rate, 6) if avg_activation_rate != "" else "",
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
                "eval_action_mode": eval_action_mode,
                "safeguard_threshold": safeguard_threshold if eval_action_mode == "safeguarded" else None,
                "mc_samples": mc_samples if eval_action_mode == "safeguarded" else None,
                "avg_uncertainty": avg_uncertainty if avg_uncertainty != "" else None,
                "avg_max_uncertainty": avg_max_uncertainty if avg_max_uncertainty != "" else None,
                "avg_activations_per_episode": (
                    avg_activations_per_episode if avg_activations_per_episode != "" else None
                ),
                "avg_activation_rate": avg_activation_rate if avg_activation_rate != "" else None,
                "total_activations": total_activations if total_activations != "" else None,
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
                "config_name": CONFIG_NAME,
                "target_timesteps": TARGET_TIMESTEPS,
                "train_seed": TRAIN_SEED,
                "eval_seeds": EVAL_SEEDS,
                "eval_max_steps": EVAL_MAX_STEPS,
                "eval_type": "full" if EVAL_MAX_STEPS is None else "capped",
                "eval_action_mode": EVAL_ACTION_MODE,
                "safeguard_threshold": SAFEGUARD_THRESHOLD if EVAL_ACTION_MODE == "safeguarded" else None,
                "mc_samples": MC_SAMPLES if EVAL_ACTION_MODE == "safeguarded" else None,
                "hyperparameters": HYPERPARAMETERS,
                "ego_start_lane_policy": EGO_START_LANE_POLICY,
                "config_overrides": CONFIG_OVERRIDES,
                "torch_num_threads": TORCH_NUM_THREADS,
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
            "TRAIN_SEED": str(TRAIN_SEED),
            "CONFIG_NAME": CONFIG_NAME,
            "MODEL_LABEL": MODEL_LABEL,
            "TARGET_TIMESTEPS": str(TARGET_TIMESTEPS),
            "EVAL_MAX_STEPS": "" if EVAL_MAX_STEPS is None else str(EVAL_MAX_STEPS),
            "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
            "HYPERPARAMETERS": json.dumps(HYPERPARAMETERS),
            "EGO_START_LANE_POLICY": EGO_START_LANE_POLICY,
            "CONFIG_OVERRIDES": json.dumps(CONFIG_OVERRIDES),
            "EVAL_ACTION_MODE": EVAL_ACTION_MODE,
            "SAFEGUARD_THRESHOLD": str(SAFEGUARD_THRESHOLD),
            "MC_SAMPLES": str(MC_SAMPLES),
            "TORCH_NUM_THREADS": str(TORCH_NUM_THREADS),
            "OMP_NUM_THREADS": str(TORCH_NUM_THREADS),
            "MKL_NUM_THREADS": str(TORCH_NUM_THREADS),
        }
    )

    log_path = OUT_DIR / "training_stdout.log"
    command_path = OUT_DIR / "command.txt"
    if COMMAND_HEADER:
        command_path.write_text(COMMAND_HEADER + "\n", encoding="utf-8")

    with log_path.open("w", encoding="utf-8") as log_file:
        if COMMAND_HEADER:
            log_file.write("COMMAND:\n")
            log_file.write(COMMAND_HEADER)
            log_file.write("\n\n")
            log_file.flush()

        subprocess.run(
            [str(PYTHON), "-c", CHILD_CODE],
            cwd=str(MODEL_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print("Completed single mean+max study.", flush=True)
    print(f"Training/eval stdout: {log_path}", flush=True)
    print(f"Training summary: {OUT_DIR / 'training_summary.csv'}", flush=True)
    print(f"Evaluation summary: {OUT_DIR / 'evaluation_summary.csv'}", flush=True)
    print(f"Per-episode evaluation: {OUT_DIR / 'evaluation_per_episode.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
