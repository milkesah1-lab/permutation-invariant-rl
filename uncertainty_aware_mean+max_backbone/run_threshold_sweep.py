from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from gymnasium.envs.registration import register, registry

from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config
from ppo import PPO


DEFAULT_ACTOR_CHECKPOINT = (
    r"C:\Users\milke\groupCoursework\FYP\experiment_runs"
    r"\meanmax_vs_uncertainty_3seeds_20260430_173848_seed34567"
    r"\uncertainty_aware_lambda_0.01\uncertainty_aware_lambda_0.01_actor.pth"
)
DEFAULT_CRITIC_CHECKPOINT = (
    r"C:\Users\milke\groupCoursework\FYP\experiment_runs"
    r"\meanmax_vs_uncertainty_3seeds_20260430_173848_seed34567"
    r"\uncertainty_aware_lambda_0.01\uncertainty_aware_lambda_0.01_critic.pth"
)


def resolve_output_dir() -> Path:
    explicit = os.environ.get("OUTPUT_DIR")
    if explicit:
        output_dir = Path(explicit)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path.cwd().parent
            / "experiment_runs"
            / f"uncertainty_aware2_threshold_sweep_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_thresholds() -> list[float]:
    raw = os.environ.get("THRESHOLDS", "0.0005,0.0010,0.0015,0.0020")
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_eval_seeds() -> list[int]:
    raw = os.environ.get("EVAL_SEEDS")
    if raw:
        return [int(value.strip()) for value in raw.split(",") if value.strip()]
    return list(range(1000, 1050))


def load_checkpoint(module, path: Path) -> dict[str, list[str] | int]:
    incompatible = module.load_state_dict(
        torch.load(path, map_location="cpu"),
        strict=False,
    )
    loaded_tensors = len(module.state_dict()) - len(incompatible.missing_keys)
    return {
        "loaded_tensors": loaded_tensors,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def main() -> None:
    config_name = os.environ.get("HIGHWAY_CONFIG", DEFAULT_HIGHWAY_CONFIG)
    config = get_highway_config(config_name)
    thresholds = parse_thresholds()
    eval_seeds = parse_eval_seeds()
    mc_samples = int(os.environ.get("MC_SAMPLES", "5"))

    actor_checkpoint = Path(os.environ.get("ACTOR_CHECKPOINT", DEFAULT_ACTOR_CHECKPOINT))
    critic_checkpoint = Path(os.environ.get("CRITIC_CHECKPOINT", DEFAULT_CRITIC_CHECKPOINT))
    output_dir = resolve_output_dir()

    if "continuous-spawn-highway-v0" not in registry:
        register(
            id="continuous-spawn-highway-v0",
            entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
        )

    hyperparameters = {
        "timesteps_per_batch": 4096,
        "max_timesteps_per_episode": 300,
        "gamma": 0.99,
        "n_updates_per_iteration": 5,
        "lr": 1e-4,
        "clip": 0.2,
        "dropout_p": 0.1,
        "mc_samples": mc_samples,
        "lambda_u": 0.01,
        "fixed_cov_var": 0.08,
    }

    env = gym.make("continuous-spawn-highway-v0", config=config)
    agent = PPO(env, **hyperparameters)
    actor_load = load_checkpoint(agent.actor, actor_checkpoint)
    critic_load = load_checkpoint(agent.critic, critic_checkpoint)
    agent.actor.eval()

    per_episode_rows: list[dict[str, float | int | bool]] = []
    summary_rows: list[dict[str, float | int]] = []

    for threshold in thresholds:
        returns: list[float] = []
        lengths: list[int] = []
        collisions: list[int] = []
        avg_uncertainties: list[float] = []
        max_uncertainties: list[float] = []
        activation_counts: list[int] = []
        activation_rates: list[float] = []

        for episode_index, seed in enumerate(eval_seeds, start=1):
            obs, _ = env.reset(seed=seed)
            done = False
            ep_ret = 0.0
            ep_len = 0
            ep_uncertainties: list[float] = []
            ep_activations = 0
            collided = False

            while not done:
                action, uncertainty, activated = agent.get_safeguarded_action(
                    obs,
                    threshold=threshold,
                    mc_samples=mc_samples,
                )
                obs, rew, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_ret += float(rew)
                ep_len += 1
                ep_uncertainties.append(float(uncertainty))
                if activated:
                    ep_activations += 1
                if env.unwrapped.vehicle.crashed:
                    collided = True

            avg_uncertainty = float(np.mean(ep_uncertainties))
            max_uncertainty = float(np.max(ep_uncertainties))
            activation_rate = float(ep_activations / ep_len) if ep_len > 0 else 0.0

            returns.append(ep_ret)
            lengths.append(ep_len)
            collisions.append(1 if collided else 0)
            avg_uncertainties.append(avg_uncertainty)
            max_uncertainties.append(max_uncertainty)
            activation_counts.append(ep_activations)
            activation_rates.append(activation_rate)

            per_episode_rows.append(
                {
                    "threshold": threshold,
                    "episode": episode_index,
                    "seed": seed,
                    "episodic_return": ep_ret,
                    "episodic_length": ep_len,
                    "collided": collided,
                    "avg_uncertainty": avg_uncertainty,
                    "max_uncertainty": max_uncertainty,
                    "activation_count": ep_activations,
                    "activation_rate": activation_rate,
                }
            )

        summary_rows.append(
            {
                "threshold": threshold,
                "episodes": len(eval_seeds),
                "avg_episodic_return": float(np.mean(returns)),
                "avg_episodic_length": float(np.mean(lengths)),
                "collision_rate": float(np.mean(collisions)),
                "avg_uncertainty_per_episode": float(np.mean(avg_uncertainties)),
                "avg_episode_max_uncertainty": float(np.mean(max_uncertainties)),
                "avg_activations_per_episode": float(np.mean(activation_counts)),
                "avg_activation_rate_per_episode": float(np.mean(activation_rates)),
                "total_activations": int(np.sum(activation_counts)),
            }
        )

    env.close()

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    per_episode_csv = output_dir / "per_episode.csv"
    with per_episode_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_episode_rows)

    run_config = {
        "config_name": config_name,
        "thresholds": thresholds,
        "eval_seeds": eval_seeds,
        "mc_samples": mc_samples,
        "actor_checkpoint": str(actor_checkpoint),
        "critic_checkpoint": str(critic_checkpoint),
        "actor_load": actor_load,
        "critic_load": critic_load,
    }
    config_json = output_dir / "run_config.json"
    config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("Threshold sweep complete.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-episode CSV: {per_episode_csv}")
    print(f"Run config: {config_json}")
    print()
    for row in summary_rows:
        print(
            "threshold={threshold:.4f} | return={avg_episodic_return:.2f} | "
            "length={avg_episodic_length:.2f} | collision={collision_rate:.3f} | "
            "avg_unc={avg_uncertainty_per_episode:.6f} | avg_act={avg_activations_per_episode:.2f}"
            .format(**row)
        )


if __name__ == "__main__":
    main()
