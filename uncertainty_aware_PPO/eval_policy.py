"""
    This file is used only to evaluate our trained policy/actor after
    training. It shows that the trained policy exists independently of
    the learning algorithm in ppo.py.
"""

import numpy as np
import torch
from torch.distributions import MultivariateNormal


def _log_summary(ep_len, ep_ret, ep_num, crashed):
    """
    Print episode statistics.
    """
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))
    crash_status = "Yes" if crashed else "No"

    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"Crashed: {crash_status}", flush=True)
    print(f"-----------------------------------------------------------", flush=True)
    print(flush=True)


def eval_policy(policy, env, num_episodes=5, render=False):
    """
    Evaluate a trained policy for a fixed number of episodes.

    Parameters:
        policy - trained actor network
        env - environment to evaluate on
        num_episodes - number of evaluation episodes
        render - whether to render episodes

    Return:
        avg_ep_len - average episode length
        avg_ep_ret - average episode return
        collision_rate - fraction of episodes ending in collision
    """
    collision_count = 0
    episode_returns = []
    episode_lengths = []

    # evaluation mode is helpful if you use dropout in your actor
    policy.eval()

    try:
        device = next(policy.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    act_dim = env.action_space.shape[0]
    cov_var = torch.full(size=(act_dim,), fill_value=0.05, device=device)
    cov_mat = torch.diag(cov_var)

    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            collided = False
            ep_ret = 0.0
            ep_len = 0

            while not done:
                if render:
                    env.render()

                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                mean = policy(obs_t)
                dist = MultivariateNormal(mean, cov_mat)
                action = dist.sample().detach().cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_ret += rew
                ep_len += 1

                if env.unwrapped.vehicle.crashed:
                    collided = True

            if collided:
                collision_count += 1

            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)

            _log_summary(ep_len, ep_ret, ep + 1, collided)

    avg_ep_ret = float(np.mean(episode_returns))
    avg_ep_len = float(np.mean(episode_lengths))
    collision_rate = collision_count / num_episodes

    print("=============== Evaluation Summary ===============", flush=True)
    print(f"Average Episodic Return: {avg_ep_ret:.3f}", flush=True)
    print(f"Average Episodic Length: {avg_ep_len:.3f}", flush=True)
    print(f"Collision Rate: {collision_rate:.3f}", flush=True)
    print("==================================================", flush=True)

    return avg_ep_len, avg_ep_ret, collision_rate
