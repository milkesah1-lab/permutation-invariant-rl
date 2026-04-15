import gymnasium as gym
import highway_env  # noqa: F401
import torch
from gymnasium.wrappers import FlattenObservation, RecordVideo

from eval_policy import eval_policy
from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config
from network import FeedForwardNN
from ppo import PPO


hyperparameters = {
    "timesteps_per_batch": 2048,
    "max_timesteps_per_episode": 80,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 2e-4,
    "clip": 0.2,
}

config_name = DEFAULT_HIGHWAY_CONFIG
config = get_highway_config(config_name)

env = gym.make("highway-v0", config=config)
env = FlattenObservation(env)

model = PPO(FeedForwardNN, env, **hyperparameters)
model_name = "baseline_discrete_meta_ppo_model"

model.actor.load_state_dict(
    torch.load(f"./{model_name}_actor.pth", map_location=model.device)
)
model.critic.load_state_dict(
    torch.load(f"./{model_name}_critic.pth", map_location=model.device)
)

model.learn(total_timesteps=200000)


env.close()

eval_env = gym.make(
    "highway-v0",
    render_mode="rgb_array",
    config=config,
)
eval_env = FlattenObservation(eval_env)
eval_env = RecordVideo(
    eval_env,
    video_folder="videos/",
    episode_trigger=lambda episode_id: episode_id < 5,
)

avg_len, avg_ret, collision_rate = eval_policy(
    model.actor,
    eval_env,
    num_episodes=5,
    render=False,
)

print(
    f"Final Eval -> Avg Length: {avg_len:.2f}, "
    f"Avg Return: {avg_ret:.2f}, Collision Rate: {collision_rate:.3f}"
)

eval_env.close()
