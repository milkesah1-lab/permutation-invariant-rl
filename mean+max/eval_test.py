import os
import gymnasium as gym
import highway_env
from gymnasium.envs.registration import registry, register
from gymnasium.wrappers import FlattenObservation
import torch
from ppo import PPO
from gymnasium.wrappers import RecordVideo

from eval_policy import eval_policy
from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config

hyperparameters = {
    "timesteps_per_batch": 2048,
    "max_timesteps_per_episode": 120,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 2e-4,
    "clip": 0.2,
    "fixed_cov_var":0.08
}

config_name = os.environ.get("HIGHWAY_CONFIG", DEFAULT_HIGHWAY_CONFIG)

if "continuous-spawn-highway-v0" not in registry:
    register(
        id="continuous-spawn-highway-v0",
        entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
    )

env = gym.make(
    "continuous-spawn-highway-v0",
    render_mode="rgb_array",
    config=get_highway_config(config_name)
)
env = FlattenObservation(env)

model = PPO(env, **hyperparameters)
model_name = f"baseline_ppo_{config_name}"


def load_checkpoint(module, path):
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


load_checkpoint(model.actor, f"./{model_name}_actor.pth")
load_checkpoint(model.critic, f"./{model_name}_critic.pth")

env.close()

# Evaluation environment
eval_env = gym.make(
    "continuous-spawn-highway-v0",
    render_mode="rgb_array",
    config=get_highway_config(config_name)
)

eval_env = FlattenObservation(eval_env)
eval_env = RecordVideo(
    eval_env,
    video_folder="videos/",
    episode_trigger=lambda episode_id: episode_id < 5
)

avg_len, avg_ret, collision_rate = eval_policy(
    model.actor,
    eval_env,
    num_episodes=5,
    render=False
)

print(f"Final Eval -> Avg Length: {avg_len:.2f}, Avg Return: {avg_ret:.2f}, Collision Rate: {collision_rate:.3f}")

eval_env.close()
