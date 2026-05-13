"""
    This file is used only to evaluate our trained policy/actor after
    training. It shows that the trained policy exists independently of
    the learning algorithm in ppo.py.
"""

import numpy as np
import torch


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


def eval_policy(
    agent,
    env,
    num_episodes=5,
    render=False,
    max_steps=None,
    label="Evaluation Summary",
    threshold=0.0005,
    mc_samples=5,
    eval_seeds=None,
):
    """
    Evaluate a trained policy for a fixed number of episodes.

    Parameters:
        agent - trained agent network
        env - environment to evaluate on
        num_episodes - number of evaluation episodes
        render - whether to render episodes
        eval_seeds - optional fixed reset seeds for deterministic evaluation

    Return:
        avg_ep_len - average episode length
        avg_ep_ret - average episode return
        collision_rate - fraction of episodes ending in collision
    """
    collision_count = 0
    episode_returns = []
    episode_lengths = []
    avg_uncertainties_per_episode = []
    max_uncertainty_per_episode = []
    activation_counts_per_episode = []

    # evaluation mode is helpful if you use dropout in your actor
    agent.actor.eval()

    try:
        device = next(agent.actor.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.no_grad():
        for ep in range(num_episodes):
            if eval_seeds is None:
                obs, _ = env.reset()
            else:
                obs, _ = env.reset(seed=int(eval_seeds[ep % len(eval_seeds)]))
            done = False
            collided = False
            ep_ret = 0.0
            ep_len = 0
            uncertainties = []
            activations = 0

            while not done:
                if render:
                    env.render()

                # mean = agent.actor(obs_t)
                action, uncertainty, activated = agent.get_safeguarded_action(obs, threshold=threshold, mc_samples=mc_samples)  # Get the mean action from the agent
                uncertainties.append(uncertainty)
                if activated:
                    activations += 1

                obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_ret += rew
                ep_len += 1

                if env.unwrapped.vehicle.crashed:
                    collided = True

                if max_steps is not None and ep_len >= max_steps:
                    done = True

            if collided:
                collision_count += 1

            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            avg_uncertainties_per_episode.append(np.mean(uncertainties))
            max_uncertainty_per_episode.append(np.max(uncertainties))
            activation_counts_per_episode.append(activations)
            _log_summary(ep_len, ep_ret, ep + 1, collided)

    avg_ep_ret = float(np.mean(episode_returns))
    avg_ep_len = float(np.mean(episode_lengths))
    collision_rate = collision_count / num_episodes

    print(f"=============== {label} ===============", flush=True)
    print(f"Average Episodic Return: {avg_ep_ret:.3f}", flush=True)
    print(f"Average Episodic Length: {avg_ep_len:.3f}", flush=True)
    print(f"Collision Rate: {collision_rate:.3f}", flush=True)
    print(f"Average Uncertainty per Episode: {np.mean(avg_uncertainties_per_episode):.3f}", flush=True)
    print(f"Max Uncertainty per Episode: {np.mean(max_uncertainty_per_episode):.3f}", flush=True)
    print(f"Average Activations per Episode: {np.mean(activation_counts_per_episode):.3f}", flush=True)
    print("==================================================", flush=True)

    return avg_ep_len, avg_ep_ret, collision_rate
