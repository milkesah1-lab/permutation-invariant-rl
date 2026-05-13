import os
import gymnasium as gym
import highway_env
from gymnasium.envs.registration import registry, register
from gymnasium.wrappers import RecordVideo
import torch

from ppo import PPO
from eval_policy import eval_policy
from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config

# Register custom env once
if "continuous-spawn-highway-v0" not in registry:
    register(
        id="continuous-spawn-highway-v0",
        entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
    )

hyperparameters = {
    "timesteps_per_batch": int(os.environ.get("TIMESTEPS_PER_BATCH", "4096")),
    "max_timesteps_per_episode": int(os.environ.get("MAX_TIMESTEPS_PER_EPISODE", "300")),
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 1e-4,
    "clip": 0.2,
    "dropout_p": 0.1,
    "mc_samples": 5,
    "lambda_u": float(os.environ.get("LAMBDA_U", "0.01")),
    "fixed_cov_var": 0.08,
}

config_name = os.environ.get("HIGHWAY_CONFIG", DEFAULT_HIGHWAY_CONFIG)
total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))
eval_episodes = int(os.environ.get("EVAL_EPISODES", "5"))

env = gym.make(
    "continuous-spawn-highway-v0",
    render_mode="rgb_array",
    config=get_highway_config(config_name)
)

model = PPO(env, **hyperparameters)
model.learn(total_timesteps=total_timesteps)

model_name = (
    f"uncertainty_aware_ppo_{config_name}_lambda_u_{hyperparameters['lambda_u']}"
)

torch.save(model.actor.state_dict(), f"./{model_name}_actor.pth")
torch.save(model.critic.state_dict(), f"./{model_name}_critic.pth")

env.close()

# Evaluation environment
eval_env = gym.make(
    "continuous-spawn-highway-v0",
    render_mode="rgb_array",
    config=get_highway_config(config_name)
)
eval_env = RecordVideo(
    eval_env,
    video_folder="videos/",
    episode_trigger=lambda episode_id: episode_id < 5
)

avg_len, avg_ret, collision_rate = eval_policy(
    model,
    eval_env,
    num_episodes=eval_episodes,
    render=False
)

print(f"Final Eval -> Avg Length: {avg_len:.2f}, Avg Return: {avg_ret:.2f}, Collision Rate: {collision_rate:.3f}")

eval_env.close()
