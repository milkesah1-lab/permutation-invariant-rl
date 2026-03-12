import gymnasium as gym
import highway_env
from gymnasium.wrappers import FlattenObservation
from ppo import PPO
from network import FeedForwardNN

hyperparameters = {
    "timesteps_per_batch": 512,
    "max_timesteps_per_episode": 40,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 3e-4,
    "clip": 0.2,
}

env = gym.make(
    "highway-v0",
    config={
        "action": {"type": "ContinuousAction"}
    }
)
env = FlattenObservation(env)

model = PPO(FeedForwardNN, env, **hyperparameters)
model.learn(total_timesteps=2000)