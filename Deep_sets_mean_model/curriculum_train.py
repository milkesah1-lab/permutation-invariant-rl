from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import torch
from gymnasium.envs.registration import register, registry
from gymnasium.wrappers import FlattenObservation

from eval_policy import eval_policy
from highway_configs import get_highway_config
from ppo import PPO


MODEL_LABEL = os.environ.get("MODEL_LABEL", Path(__file__).resolve().parent.name)
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "10"))
CAPPED_EVAL_STEPS = int(os.environ.get("CAPPED_EVAL_STEPS", "300"))
STAGE_LIMIT = int(os.environ.get("CURRICULUM_STAGE_LIMIT", "4"))
SINGLE_STAGE_CONFIG = os.environ.get("SINGLE_STAGE_CONFIG")
SINGLE_STAGE_TIMESTEPS = int(os.environ.get("SINGLE_STAGE_TIMESTEPS", "200000"))

HYPERPARAMETERS = {
    "timesteps_per_batch": 4096,
    "max_timesteps_per_episode": 300,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 1e-4,
    "clip": 0.2,
    "save_freq": 1000000,
    "fixed_cov_var": 0.08,
}

CURRICULUM_STAGES = [
    ("curriculum_stage_1_open_lane", 24576),
    ("curriculum_stage_2_easy_overtake", 49152),
    ("curriculum_stage_3_mixed_traffic", 98304),
    ("curriculum_stage_4_dense_traffic", 49152),
]


def get_training_stages():
    if SINGLE_STAGE_CONFIG:
        return [(SINGLE_STAGE_CONFIG, SINGLE_STAGE_TIMESTEPS)]
    return CURRICULUM_STAGES[:STAGE_LIMIT]


def make_env(config_name: str):
    env = gym.make(
        "continuous-spawn-highway-v0",
        config=get_highway_config(config_name),
    )
    return FlattenObservation(env)


def evaluate_final_model(model: PPO, stages, eval_csv_path: Path, final_actor_path: Path) -> None:
    eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "model",
                "config",
                "eval_type",
                "episodes",
                "max_steps",
                "avg_episodic_length",
                "avg_episodic_return",
                "collision_rate",
                "checkpoint",
            ]
        )

        for config_name, _ in stages:
            for eval_type, max_steps in (
                ("capped", CAPPED_EVAL_STEPS),
                ("full", None),
            ):
                print()
                print(
                    f"Evaluating {MODEL_LABEL} on {config_name} "
                    f"({eval_type}, episodes={EVAL_EPISODES})"
                )
                env = make_env(config_name)
                avg_len, avg_ret, collision_rate = eval_policy(
                    model.actor,
                    env,
                    num_episodes=EVAL_EPISODES,
                    render=False,
                    max_steps=max_steps,
                    label=f"{MODEL_LABEL} {config_name} {eval_type} eval",
                )
                env.close()

                writer.writerow(
                    [
                        MODEL_LABEL,
                        config_name,
                        eval_type,
                        EVAL_EPISODES,
                        "" if max_steps is None else max_steps,
                        round(avg_len, 6),
                        round(avg_ret, 6),
                        round(collision_rate, 6),
                        str(final_actor_path),
                    ]
                )


def main():
    if "continuous-spawn-highway-v0" not in registry:
        register(
            id="continuous-spawn-highway-v0",
            entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
        )

    stages = get_training_stages()
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "curriculum_artifacts" / RUN_ID
    csv_dir = artifacts_dir / "csv"
    checkpoints_dir = artifacts_dir / "checkpoints"
    eval_dir = artifacts_dir / "evaluation"
    csv_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL_LABEL}")
    print(f"Run ID: {RUN_ID}")
    print("Training from scratch: no checkpoints will be loaded.")
    print(f"Artifacts directory: {artifacts_dir}")

    model = None

    for stage_index, (config_name, stage_timesteps) in enumerate(stages, start=1):
        print()
        print(f"==================== Stage {stage_index}: {config_name} ====================")
        print(f"Target timesteps for this stage: {stage_timesteps}")

        env = make_env(config_name)

        if model is None:
            model = PPO(env, **HYPERPARAMETERS)
        else:
            model.set_env(env)

        csv_log_path = csv_dir / f"{stage_index:02d}_{config_name}_training.csv"
        model.set_csv_log_path(csv_log_path)
        model.learn(total_timesteps=stage_timesteps)

        actor_path = checkpoints_dir / f"{stage_index:02d}_{config_name}_actor.pth"
        critic_path = checkpoints_dir / f"{stage_index:02d}_{config_name}_critic.pth"
        torch.save(model.actor.state_dict(), actor_path)
        torch.save(model.critic.state_dict(), critic_path)

        env.close()

        print(f"Saved stage CSV to: {csv_log_path}")
        print(f"Saved stage actor checkpoint to: {actor_path}")
        print(f"Saved stage critic checkpoint to: {critic_path}")

    final_actor_path = checkpoints_dir / f"{MODEL_LABEL}_curriculum_final_actor.pth"
    final_critic_path = checkpoints_dir / f"{MODEL_LABEL}_curriculum_final_critic.pth"
    torch.save(model.actor.state_dict(), final_actor_path)
    torch.save(model.critic.state_dict(), final_critic_path)

    eval_csv_path = eval_dir / f"{MODEL_LABEL}_final_evaluation.csv"
    evaluate_final_model(model, stages, eval_csv_path, final_actor_path)

    print()
    print("Curriculum training and evaluation complete.")
    print(f"Final actor checkpoint: {final_actor_path}")
    print(f"Final critic checkpoint: {final_critic_path}")
    print(f"Evaluation CSV: {eval_csv_path}")


if __name__ == "__main__":
    main()
