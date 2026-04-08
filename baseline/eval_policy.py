"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""
import numpy as np
import torch
from torch.distributions import MultivariateNormal

def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def eval_policy(policy, env, num_episodes, render=False):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			num_episodes - The number of episodes to evaluate
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
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
				action = mean.detach().cpu().numpy()
				# dist = MultivariateNormal(mean, cov_mat)  # making it use actor's mean action instead of sampling from the distribution
				# action = dist.sample().detach().cpu().numpy()
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

			_log_summary(ep_len, ep_ret, ep + 1)

	avg_ep_ret = float(np.mean(episode_returns))
	avg_ep_len = float(np.mean(episode_lengths))
	collision_rate = collision_count / num_episodes

	print("=============== Evaluation Summary ===============", flush=True)
	print(f"Average Episodic Return: {avg_ep_ret:.3f}", flush=True)
	print(f"Average Episodic Length: {avg_ep_len:.3f}", flush=True)
	print(f"Collision Rate: {collision_rate:.3f}", flush=True)
	print("==================================================", flush=True)

	return avg_ep_len, avg_ep_ret, collision_rate
			
