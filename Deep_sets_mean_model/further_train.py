import os
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import torch
from gymnasium.envs.registration import register, registry
from gymnasium.wrappers import FlattenObservation, RecordVideo

from eval_policy import eval_policy
from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config
from ppo import PPO


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_checkpoint_pair(raw_value: str) -> tuple[Path, Path]:
    raw_path = Path(raw_value)
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path

    raw_str = str(raw_path)
    if raw_str.endswith("_actor.pth"):
        actor_path = raw_path
        critic_path = Path(raw_str[:-10] + "_critic.pth")
    elif raw_str.endswith("_critic.pth"):
        critic_path = raw_path
        actor_path = Path(raw_str[:-11] + "_actor.pth")
    else:
        actor_path = Path(raw_str + "_actor.pth")
        critic_path = Path(raw_str + "_critic.pth")

    return actor_path, critic_path


def resolve_output_prefix(raw_value: str) -> Path:
    raw_path = Path(raw_value)
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    return raw_path


hyperparameters = {
    "timesteps_per_batch": 4096,
    "max_timesteps_per_episode": 300,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 1e-4,
    "clip": 0.2,
    "fixed_cov_var": 0.08,
}

config_name = os.environ.get("HIGHWAY_CONFIG", DEFAULT_HIGHWAY_CONFIG)
config = get_highway_config(config_name)
total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))
eval_episodes = int(os.environ.get("EVAL_EPISODES", "5"))
capped_eval_steps = int(os.environ.get("CAPPED_EVAL_STEPS", "300"))
run_eval = env_flag("RUN_EVAL", False)
record_video = env_flag("RECORD_VIDEO", False)

if "continuous-spawn-highway-v0" not in registry:
    register(
        id="continuous-spawn-highway-v0",
        entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
    )

env = gym.make("continuous-spawn-highway-v0", config=config)
env = FlattenObservation(env)

model = PPO(env, **hyperparameters)
checkpoint_value = os.environ.get(
    "BASELINE_CHECKPOINT",
    "deep_sets_mean_curriculum_final",
)
actor_checkpoint, critic_checkpoint = resolve_checkpoint_pair(checkpoint_value)

output_value = os.environ.get(
    "OUTPUT_CHECKPOINT",
    f"deep_sets_mean_{config_name}_continued",
)
output_prefix = resolve_output_prefix(output_value)

csv_log_path = os.environ.get("CSV_LOG_PATH")
if csv_log_path:
    csv_path = Path(csv_log_path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    model.set_csv_log_path(csv_path)

print(f"Further training config: {config_name}")
print(f"Loading actor checkpoint: {actor_checkpoint}")
print(f"Loading critic checkpoint: {critic_checkpoint}")
print(f"Saving checkpoint prefix: {output_prefix}")
print(f"Further training timesteps: {total_timesteps}")
print(f"Run eval after training: {run_eval}")
print(f"Record video: {record_video}")


def load_checkpoint(module, path: Path):
    incompatible = module.load_state_dict(
        torch.load(path, map_location=model.device),
        strict=False,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"Loaded {path} with checkpoint compatibility adjustments.")
        if incompatible.missing_keys:
            print(f"  Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  Unexpected keys: {incompatible.unexpected_keys}")


load_checkpoint(model.actor, actor_checkpoint)
load_checkpoint(model.critic, critic_checkpoint)

model.learn(total_timesteps=total_timesteps)

torch.save(model.actor.state_dict(), str(output_prefix) + "_actor.pth")
torch.save(model.critic.state_dict(), str(output_prefix) + "_critic.pth")

env.close()

if run_eval:
    print(f"Running capped {capped_eval_steps}-step evaluation...")
    capped_eval_env = gym.make(
        "continuous-spawn-highway-v0",
        render_mode="rgb_array" if record_video else None,
        config=config,
    )
    capped_eval_env = FlattenObservation(capped_eval_env)
    capped_avg_len, capped_avg_ret, capped_collision_rate = eval_policy(
        model.actor,
        capped_eval_env,
        num_episodes=eval_episodes,
        render=False,
        max_steps=capped_eval_steps,
        label=f"Capped Evaluation Summary ({capped_eval_steps} steps)",
    )
    print(
        f"Capped Eval -> Avg Length: {capped_avg_len:.2f}, "
        f"Avg Return: {capped_avg_ret:.2f}, Collision Rate: {capped_collision_rate:.3f}"
    )
    capped_eval_env.close()

    print("Running full-length evaluation...")
    eval_env = gym.make(
        "continuous-spawn-highway-v0",
        render_mode="rgb_array" if record_video else None,
        config=config,
    )
    eval_env = FlattenObservation(eval_env)
    if record_video:
        eval_env = RecordVideo(
            eval_env,
            video_folder="videos/",
            episode_trigger=lambda episode_id: episode_id < min(eval_episodes, 5),
        )

    avg_len, avg_ret, collision_rate = eval_policy(
        model.actor,
        eval_env,
        num_episodes=eval_episodes,
        render=False,
        label="Full Evaluation Summary",
    )

    print(
        f"Final Eval -> Avg Length: {avg_len:.2f}, "
        f"Avg Return: {avg_ret:.2f}, Collision Rate: {collision_rate:.3f}"
    )

    eval_env.close()
