"""
Micro-benchmarks for PPO + highway-env timing.

Runs four measurements:
1) Raw highway-env step time (no FlattenObservation).
2) highway-env step time with FlattenObservation.
3) PPO get_action overhead (no env.step).
4) Full PPO rollout time per step.
"""

import os
import time
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401  # registers envs
from gymnasium.wrappers import FlattenObservation

from ppo import PPO
from network import FeedForwardNN
from highway_configs import DEFAULT_HIGHWAY_CONFIG, get_highway_config


def make_env(flatten: bool):
    config_name = os.environ.get("HIGHWAY_CONFIG", DEFAULT_HIGHWAY_CONFIG)
    env = gym.make(
        "highway-v0",
        config=get_highway_config(config_name),
    )
    if flatten:
        env = FlattenObservation(env)
    return env


def bench_env_step(env, num_steps: int, warmup: int) -> float:
    obs, _ = env.reset()
    action = env.action_space.sample()

    # Warm-up (not timed)
    for _ in range(warmup):
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # Timed steps only (resets excluded)
    total = 0.0
    for _ in range(num_steps):
        t0 = time.perf_counter()
        obs, _, terminated, truncated, _ = env.step(action)
        total += time.perf_counter() - t0
        if terminated or truncated:
            obs, _ = env.reset()

    return total / num_steps


def bench_get_action(num_steps: int, warmup: int) -> float:
    env = make_env(flatten=True)
    obs, _ = env.reset()

    hyperparameters = {
        "timesteps_per_batch": 1000,
        "max_timesteps_per_episode": 40,
        "gamma": 0.99,
        "n_updates_per_iteration": 5,
        "lr": 3e-4,
        "clip": 0.2,
        "render": False,
    }
    model = PPO(FeedForwardNN, env, **hyperparameters)

    # Warm-up
    for _ in range(warmup):
        model.get_action(obs)

    # Timed get_action calls
    total = 0.0
    for _ in range(num_steps):
        t0 = time.perf_counter()
        model.get_action(obs)
        total += time.perf_counter() - t0

    env.close()
    return total / num_steps


def bench_full_rollout(warmup_rollouts: int = 1) -> float:
    env = make_env(flatten=True)

    hyperparameters = {
        "timesteps_per_batch": 1000,
        "max_timesteps_per_episode": 40,
        "gamma": 0.99,
        "n_updates_per_iteration": 5,
        "lr": 3e-4,
        "clip": 0.2,
        "render": False,
    }
    model = PPO(FeedForwardNN, env, **hyperparameters)

    # Warm-up rollout(s)
    for _ in range(warmup_rollouts):
        model.rollout()

    # Timed rollout
    t0 = time.perf_counter()
    _, _, _, _, batch_lens = model.rollout()
    elapsed = time.perf_counter() - t0
    steps = int(np.sum(batch_lens))

    env.close()
    return elapsed / steps


def main():
    config_name = os.environ.get("HIGHWAY_CONFIG", DEFAULT_HIGHWAY_CONFIG)
    print(f"Using highway config: {config_name}")

    num_steps = 1000
    warmup = 100

    # 1) Raw highway-env step time (no FlattenObservation)
    env_raw = make_env(flatten=False)
    raw_env_step = bench_env_step(env_raw, num_steps=num_steps, warmup=warmup)
    env_raw.close()

    # 2) highway-env step time with FlattenObservation
    env_flat = make_env(flatten=True)
    flat_env_step = bench_env_step(env_flat, num_steps=num_steps, warmup=warmup)
    env_flat.close()

    # 3) PPO get_action overhead
    get_action_time = bench_get_action(num_steps=num_steps, warmup=warmup)

    # 4) Full PPO rollout time per step
    rollout_time = bench_full_rollout(warmup_rollouts=1)

    print(f"raw_env_step: {raw_env_step:.6f} seconds per step")
    print(f"flattened_env_step: {flat_env_step:.6f} seconds per step")
    print(f"get_action: {get_action_time:.6f} seconds per call")
    print(f"full_rollout: {rollout_time:.6f} seconds per step")


if __name__ == "__main__":
    main()
