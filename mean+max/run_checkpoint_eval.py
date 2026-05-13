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

from PI_network import PIFeedForwardNN
from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config


DEFAULT_ACTOR_CHECKPOINT = (
    r"C:\Users\milke\groupCoursework\FYP\experiment_runs"
    r"\meanmax_vs_uncertainty_3seeds_20260430_173848_seed12345"
    r"\deep_sets_mean_max_baseline\deep_sets_mean_max_baseline_actor.pth"
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
            / f"meanmax_checkpoint_eval_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_eval_seeds() -> list[int]:
    raw = os.environ.get("EVAL_SEEDS")
    if raw:
        return [int(value.strip()) for value in raw.split(",") if value.strip()]
    return list(range(1000, 1050))


def main() -> None:
    config_name = os.environ.get("HIGHWAY_CONFIG", DEFAULT_HIGHWAY_CONFIG)
    config = get_highway_config(config_name)
    actor_checkpoint = Path(os.environ.get("ACTOR_CHECKPOINT", DEFAULT_ACTOR_CHECKPOINT))
    eval_seeds = parse_eval_seeds()
    output_dir = resolve_output_dir()

    if "continuous-spawn-highway-v0" not in registry:
        register(
            id="continuous-spawn-highway-v0",
            entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
        )

    env = gym.make("continuous-spawn-highway-v0", config=config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    num_of_vehicles = env.unwrapped.config["observation"]["vehicles_count"]
    num_of_features = len(env.unwrapped.config["observation"]["features"])

    actor = PIFeedForwardNN(
        out_dim=act_dim,
        num_of_features=num_of_features,
        num_of_vehicles=num_of_vehicles,
    )
    incompatible = actor.load_state_dict(torch.load(actor_checkpoint, map_location="cpu"), strict=False)
    actor.eval()

    per_episode_rows: list[dict[str, float | int | bool]] = []
    returns: list[float] = []
    lengths: list[int] = []
    collisions: list[int] = []

    with torch.no_grad():
        for episode_index, seed in enumerate(eval_seeds, start=1):
            obs, _ = env.reset(seed=seed)
            done = False
            ep_ret = 0.0
            ep_len = 0
            collided = False

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                action = actor(obs_t).detach().cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, rew, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_ret += float(rew)
                ep_len += 1
                if env.unwrapped.vehicle.crashed:
                    collided = True

            returns.append(ep_ret)
            lengths.append(ep_len)
            collisions.append(1 if collided else 0)
            per_episode_rows.append(
                {
                    "episode": episode_index,
                    "seed": seed,
                    "episodic_return": ep_ret,
                    "episodic_length": ep_len,
                    "collided": collided,
                }
            )

    env.close()

    summary = {
        "config_name": config_name,
        "episodes": len(eval_seeds),
        "avg_episodic_return": float(np.mean(returns)),
        "avg_episodic_length": float(np.mean(lengths)),
        "collision_rate": float(np.mean(collisions)),
        "actor_checkpoint": str(actor_checkpoint),
        "load_missing_keys": list(incompatible.missing_keys),
        "load_unexpected_keys": list(incompatible.unexpected_keys),
    }

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    per_episode_csv = output_dir / "per_episode.csv"
    with per_episode_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_episode_rows)

    run_config = {
        "config_name": config_name,
        "eval_seeds": eval_seeds,
        "actor_checkpoint": str(actor_checkpoint),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "num_of_vehicles": num_of_vehicles,
        "num_of_features": num_of_features,
        "load_missing_keys": list(incompatible.missing_keys),
        "load_unexpected_keys": list(incompatible.unexpected_keys),
    }
    config_json = output_dir / "run_config.json"
    config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("Mean+max checkpoint evaluation complete.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-episode CSV: {per_episode_csv}")
    print(f"Run config: {config_json}")
    print(
        "return={avg_episodic_return:.2f} | length={avg_episodic_length:.2f} | collision={collision_rate:.3f}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
