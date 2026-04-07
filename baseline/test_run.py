import gymnasium as gym
import highway_env
from gymnasium.envs.registration import registry, register
from gymnasium.wrappers import FlattenObservation
import torch
from ppo import PPO
from network import FeedForwardNN
from gymnasium.wrappers import RecordVideo
from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config

from eval_policy import eval_policy

hyperparameters = {
    "timesteps_per_batch": 2048,
    "max_timesteps_per_episode": 40,
    "gamma": 0.99,
    "n_updates_per_iteration": 5,
    "lr": 2e-4,
    "clip": 0.2,
}

# Register custom env once
# if "continuous-spawn-highway-v0" not in registry:
#     register(
#         id="continuous-spawn-highway-v0",
#         entry_point="continuous_spawn_highway_env:ContinuousSpawnHighwayEnv",
#     )

env = gym.make(
    "highway-v0",
    config={
        "action": {"type": "ContinuousAction"},
        "lanes_count": 4,
        "vehicles_count": 30,
        "duration": 40,  # [s]
        "initial_spacing": 2,
        "collision_reward": -2,  # The reward received when colliding with a vehicle.
        "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 3,  # [Hz]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,  # [px]
        "screen_height": 150,  # [px]
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
 
    }
)
env = FlattenObservation(env)


model = PPO(FeedForwardNN, env, **hyperparameters)
model.learn(total_timesteps=200000)

model_name = "baseline_ppo_model"

torch.save(model.actor.state_dict(), f'./{model_name}_actor.pth')
torch.save(model.critic.state_dict(), f'./{model_name}_critic.pth')

env.close()
env = gym.make("highway-v0", config=get_highway_config(config_name))

env = gym.make(
    "continuous-spawn-highway-v0",
    render_mode="rgb_array"
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
