import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from config import *

def make_env():
    env = gym.make(
        "highway-v0",
        config={
            "observation": {
                "type": "Kinematics"
            },
            "screen_width": 1,
            "screen_height": 1,
            "show_trajectories": False,
            "render": False
        }
    )
    env.reset(seed=SEED)
    return env



if __name__ == "__main__":
    set_random_seed(SEED)

    env = make_env()

    logger = configure("logs/baseline", ["stdout", "tensorboard"])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        verbose=1,
        seed=SEED,
    )

    model.set_logger(logger)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save("models/ppo_highway_baseline")
    env.close()
