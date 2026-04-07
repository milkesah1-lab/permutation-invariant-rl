import gymnasium as gym
import highway_env
from gymnasium.envs.registration import registry, register
from gymnasium.wrappers import FlattenObservation
import torch
from ppo import PPO
from network import FeedForwardNN
from gymnasium.wrappers import RecordVideo

from eval_policy import eval_policy

hyperparameters = {
    "timesteps_per_batch": 2048,
    "max_timesteps_per_episode": 40,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 2e-4,
    "clip": 0.2,
}

if "continuous-spawn-highway-v0" not in registry:
    register(
        id="continuous-spawn-highway-v0",
        entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
    )

env = gym.make("continuous-spawn-highway-v0")
env = FlattenObservation(env)

model = PPO(FeedForwardNN, env, **hyperparameters)
model_name = "uncertainty_aware_ppo_model_lambda_u_0.05"
model.actor.load_state_dict(model.actor.state_dict(), f"./{model_name}_actor.pth",)
model.critic.load_state_dict(model.critic.state_dict(), f"./{model_name}_critic.pth",)

env.close()

# Evaluation environment
eval_env = gym.make(
    "continuous-spawn-highway-v0",
    render_mode="rgb_array"
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

