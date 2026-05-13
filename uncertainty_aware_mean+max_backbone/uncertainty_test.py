import os
import gymnasium as gym
import highway_env
from gymnasium.wrappers import FlattenObservation
from ppo import PPO
import numpy as np

from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config

hyperparameters = {
    "timesteps_per_batch": 1024,
    "max_timesteps_per_episode": 100,
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

model = PPO(env, **hyperparameters)
model.learn(total_timesteps=10000)

# print(env)
# print(env.spec)
# print(env.unwrapped.config)

# obs, info = env.reset()
# states = [obs]

# done = False
# truncated = False
# steps = 0

# while not done and not truncated:
#     action, _ = model.get_action(obs)
#     obs, rew, done, truncated, _ = env.step(action)
#     states.append(obs)
#     steps +=1

# print("Here are the uncertainty approximation for each state")
# mean_var_dict = {}

# for i, state in enumerate(states):
#     mean, var = model.estimate_critic_uncertainty(state,mc_samples=hyperparameters["mc_samples"])
#     mean_var_dict[i] = (mean, var)

# for i, (mean, var) in mean_var_dict.items():
#     if i == 0:
#         print("Early states:")

#     elif i == len(states) - 10:
#         print("Late states:")
#     print(f"State {i}: mean={mean:.4f}, var={var:.4f}")

# print(f"average of late states variance: {np.mean([var for i, (mean, var) in mean_var_dict.items() if i >= len(states) - 50]):.4f}")
# print(f"average of early states variance: {np.mean([var for i, (mean, var) in mean_var_dict.items() if i < 500]):.4f}")
        

