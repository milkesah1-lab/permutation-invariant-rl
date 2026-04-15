import gymnasium as gym
import highway_env
from gymnasium.envs.registration import registry, register
from gymnasium.wrappers import FlattenObservation, RecordVideo
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
    "timesteps_per_batch": 3072,
    "max_timesteps_per_episode": 120,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 1e-4,
    "clip": 0.2,
    "dropout_p": 0.1,
    "mc_samples": 5,
    "lambda_u": 0.01
}

env = gym.make(
    "continuous-spawn-highway-v0",
    render_mode="rgb_array",
    config=get_highway_config("realistic_light")
)


env = FlattenObservation(env)

model = PPO(env, **hyperparameters)
model.learn(total_timesteps=200000)

model_name = f"uncertainty_aware_ppo_model_lambda_u_{hyperparameters['lambda_u']}"

torch.save(model.actor.state_dict(), f"./{model_name}_actor.pth")
torch.save(model.critic.state_dict(), f"./{model_name}_critic.pth")

env.close()

# Evaluation environment
eval_env = gym.make(
    "continuous-spawn-highway-v0",
    render_mode="rgb_array",
    config=get_highway_config("realistic_light")
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
