import os
import gymnasium as gym
import highway_env
from gymnasium.envs.registration import registry, register
from gymnasium.wrappers import FlattenObservation
import torch
from ppo import PPO
from gymnasium.wrappers import RecordVideo
from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config

from eval_policy import eval_policy

hyperparameters = {
    "timesteps_per_batch": 4096,
    "max_timesteps_per_episode": 225,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 1e-4,
    "clip": 0.2,
    "fixed_cov_var":0.08,
}

config_name = os.environ.get("HIGHWAY_CONFIG", DEFAULT_HIGHWAY_CONFIG)

# Register custom env once
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
model.learn(total_timesteps=50000)

model_name = f"baseline_ppo_{config_name}"

torch.save(model.actor.state_dict(), f'./{model_name}_actor.pth')
torch.save(model.critic.state_dict(), f'./{model_name}_critic.pth')

env.close()
env = gym.make(
    "continuous-spawn-highway-v0",
    render_mode="rgb_array",
    config=get_highway_config(config_name)
)
env = FlattenObservation(env)
env = RecordVideo(
    env,
    video_folder="videos/",
    episode_trigger=lambda episode_id: episode_id < 5
)

avg_len, avg_ret, collision_rate = eval_policy(model.actor, env, 5, render=False)
print(f"Final Eval -> Avg Length: {avg_len:.2f}, Avg Return: {avg_ret:.2f}, Collision Rate: {collision_rate:.3f}")
env.close()
