import os
import gymnasium as gym
import highway_env
from gymnasium.wrappers import FlattenObservation
from ppo import PPO
from network import FeedForwardNN

from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config

hyperparameters = {
    "timesteps_per_batch": 1024,
    "max_timesteps_per_episode": 40,
    "gamma": 0.99,
    "n_updates_per_iteration": 10,
    "lr": 3e-4,
    "clip": 0.2,
    "dropout_p": 0.1,
    "mc_samples": 5,
}

config_name = os.environ.get("HIGHWAY_CONFIG", DEFAULT_HIGHWAY_CONFIG)
env = gym.make("highway-v0", config=get_highway_config(config_name))
env = FlattenObservation(env)

model = PPO(FeedForwardNN, env, **hyperparameters)
model.learn(total_timesteps=10000)