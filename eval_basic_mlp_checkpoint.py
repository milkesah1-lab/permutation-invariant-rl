from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from gymnasium.envs.registration import register, registry
from gymnasium.wrappers import FlattenObservation


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "baseline1"))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from highway_configs import get_highway_config  # noqa: E402
from network import FeedForwardNN  # noqa: E402


def parse_eval_seeds() -> list[int]:
    raw = os.environ.get("EVAL_SEEDS")
    if raw:
        return [int(seed) for seed in json.loads(raw)]
    return list(range(1000, 1050))


def main() -> int:
    output_dir = Path(os.environ["OUTPUT_DIR"])
    actor_checkpoint = Path(os.environ["ACTOR_CHECKPOINT"])
    config_name = os.environ.get("HIGHWAY_CONFIG", "curriculum_stage_3_mixed_traffic")
    eval_seeds = parse_eval_seeds()
    eval_max_steps_raw = os.environ.get("EVAL_MAX_STEPS", "full").strip().lower()
    eval_max_steps = None if eval_max_steps_raw in {"", "none", "full"} else int(eval_max_steps_raw)
    command_header = os.environ.get("COMMAND_HEADER", "").strip()
    source_run = os.environ.get("SOURCE_RUN", "")

    output_dir.mkdir(parents=True, exist_ok=True)
    if command_header:
        (output_dir / "command.txt").write_text(command_header + "\n", encoding="utf-8")

    if "continuous-spawn-highway-v0" not in registry:
        register(
            id="continuous-spawn-highway-v0",
            entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
        )

    config = get_highway_config(config_name)
    config["ego_start_lane_policy"] = "random"

    def make_env():
        env = gym.make("continuous-spawn-highway-v0", config=config)
        return FlattenObservation(env)

    env = make_env()
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    actor = FeedForwardNN(obs_dim, act_dim)
    actor.load_state_dict(torch.load(actor_checkpoint, map_location="cpu"))
    actor.eval()
    env.close()

    eval_env = make_env()
    eval_env.action_space.seed(0)
    eval_env.observation_space.seed(0)
    rows = []

    with torch.no_grad():
        for episode_index, seed in enumerate(eval_seeds, start=1):
            obs, info = eval_env.reset(seed=int(seed))
            done = False
            episodic_return = 0.0
            episodic_length = 0
            collided = False
            capped = False
            terminated = False
            truncated = False
            scenario = info.get("scenario", "unknown")

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                action = actor(obs_t).detach().cpu().numpy()
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

            rows.append(
                {
                    "model": "basic_mlp",
                    "train_seed": 12345,
                    "episode_index": episode_index,
                    "eval_seed": int(seed),
                    "episodic_return": episodic_return,
                    "episodic_length": episodic_length,
                    "collided": int(collided),
                    "capped": int(capped),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "scenario": scenario,
                }
            )

    eval_env.close()

    per_episode_csv = output_dir / "evaluation_per_episode.csv"
    with per_episode_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_return = float(np.mean([row["episodic_return"] for row in rows]))
    avg_length = float(np.mean([row["episodic_length"] for row in rows]))
    collision_rate = float(np.mean([row["collided"] for row in rows]))

    summary = {
        "model": "basic_mlp",
        "train_seed": 12345,
        "config_name": config_name,
        "eval_episodes": len(eval_seeds),
        "eval_type": "full" if eval_max_steps is None else "capped",
        "eval_max_steps": "" if eval_max_steps is None else eval_max_steps,
        "eval_action_mode": "actor",
        "avg_episodic_return": round(avg_return, 6),
        "avg_episodic_length": round(avg_length, 6),
        "collision_rate": round(collision_rate, 6),
        "per_episode_csv": str(per_episode_csv),
        "actor_checkpoint": str(actor_checkpoint),
    }
    summary_csv = output_dir / "evaluation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": "basic_mlp",
                "source_run": source_run,
                "actor_checkpoint": str(actor_checkpoint),
                "config_name": config_name,
                "eval_seeds": eval_seeds,
                "eval_max_steps": eval_max_steps,
                "eval_action_mode": "actor",
                "ego_start_lane_policy": config.get("ego_start_lane_policy"),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Basic MLP checkpoint re-evaluation complete.", flush=True)
    print(f"Summary CSV: {summary_csv}", flush=True)
    print(f"Per-episode CSV: {per_episode_csv}", flush=True)
    print(f"Average return: {avg_return:.6f}", flush=True)
    print(f"Average length: {avg_length:.6f}", flush=True)
    print(f"Collision rate: {collision_rate:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
