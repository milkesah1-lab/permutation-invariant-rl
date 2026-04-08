"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gymnasium as gym
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from network import FeedForwardNN, DropoutCritic

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, env, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - retained for backward compatibility (unused in current implementation).
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) == gym.spaces.Box)
		assert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		# Set device for all tensors and networks
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		 # Initialize actor and critic networks
		self.actor = FeedForwardNN(self.obs_dim, self.act_dim)                                                                   # ALG STEP 1
		self.critic = DropoutCritic(self.obs_dim, hidden_dim=64, dropout_p=self.dropout_p)                                      # ALG STEP 1
		self.actor.to(self.device)
		self.critic.to(self.device)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.05, device=self.device)
		self.cov_mat = torch.diag(self.cov_var)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'critic_uncertainties': [], # mean critic uncertainty per update
			'rollout_uncertainties': [], # critic uncertainties collected during rollout
			'batch_raw_rews': [],   # raw rewards in batch, without uncertainty penalty
		}

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		print(f"the uncertainty penalty lambda is {self.lambda_u}")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V_mean, V_var, _ = self.evaluate(batch_obs, batch_acts, training=False)
			A_k = batch_rtgs - V_mean.detach()                                                                  # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V_mean, V_var, curr_log_probs = self.evaluate(batch_obs, batch_acts, training=True)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)

				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V_mean, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward()
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())
				self.logger['critic_uncertainties'].append(V_var.mean().detach())

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

	def rollout(self):
		"""
			This is where we collect the batch of data from simulation.
			Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_raw_rews = []
		batch_rtgs = []
		batch_lens = []
		batch_uncertainties = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode
			ep_raw_rews = [] # raw rewards collected per episode, without uncertainty penalty

			# Reset the environment. sNote that obs is short for observation. 
			obs, _ = self.env.reset()
			done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
				# if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					# self.env.render()

				t += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				action, log_prob = self.get_action(obs)
				obs, rew, terminated, truncated, _ = self.env.step(action)

				offroad = hasattr(self.env.unwrapped, "vehicle") and not self.env.unwrapped.vehicle.on_road
				if offroad:
					rew -= 2.0
					terminated = True

				_, uncertainty_estimation = self.estimate_critic_uncertainty(obs, mc_samples=self.mc_samples)
				uncertainty_estimation = uncertainty_estimation.detach().item()
				raw_rew = rew

				batch_uncertainties.append(uncertainty_estimation)

				if len(batch_uncertainties) < 20:
					unc_norm = 0.0
				else:
					unc_mean = np.mean(batch_uncertainties)
					unc_std = np.std(batch_uncertainties) + 1e-8
					unc_norm = (uncertainty_estimation - unc_mean) / unc_std
					unc_norm = np.clip(unc_norm, 0, 3)

				rew = rew - (self.lambda_u * unc_norm)



				# Don't really care about the difference between terminated or truncated in this, so just combine them
				done = terminated | truncated

				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				ep_raw_rews.append(raw_rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
				if done:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)
			batch_raw_rews.append(ep_raw_rews)
			

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.from_numpy(np.array(batch_obs, dtype=np.float32)).to(self.device)
		batch_acts = torch.from_numpy(np.array(batch_acts, dtype=np.float32)).to(self.device)
		batch_log_probs = torch.stack(batch_log_probs).to(self.device)
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_raw_rews'] = batch_raw_rews
		self.logger['batch_lens'] = batch_lens
		self.logger['rollout_uncertainties'] = batch_uncertainties

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32, device=self.device)

		return batch_rtgs

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Convert observation to tensor if needed and move to device
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float32)
		elif not torch.is_tensor(obs):
			obs = torch.tensor(obs, dtype=torch.float32)
		obs = obs.to(self.device)

		# Query the actor network for a mean action
		mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()
		action = torch.clamp(action, torch.as_tensor(self.env.action_space.low).to(self.device), torch.as_tensor(self.env.action_space.high).to(self.device))

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().cpu().numpy(), log_prob.detach()

	def estimate_critic_uncertainty(self, obs, mc_samples=10, return_samples=False, require_grad=False):
		"""
			Estimate predictive mean and variance of the critic using MC Dropout.

			Parameters:
				obs - observation(s), single state or batch (numpy or torch)
				mc_samples - number of MC dropout samples
				return_samples - if True, also return raw MC predictions
				require_grad - if True, allow gradients through MC sampling (used during training)

			Return:
				V_mean - predictive mean
				V_var - predictive variance
				samples (optional) - raw MC predictions
		"""
		# Convert observation to tensor if needed
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float32)
		elif not torch.is_tensor(obs):
			obs = torch.tensor(obs, dtype=torch.float32)

		# Move to device and ensure batch dimension
		obs = obs.to(self.device)
		single_input = False
		if obs.dim() == 1:
			obs = obs.unsqueeze(0)
			single_input = True

		was_training = self.critic.training
		self.critic.train()

		grad_context = torch.enable_grad() if require_grad else torch.no_grad()
		with grad_context:
			samples = []
			for _ in range(mc_samples):
				pred = self.critic(obs).squeeze(-1)
				samples.append(pred)
			samples = torch.stack(samples, dim=0)
			V_mean = samples.mean(dim=0)
			V_var = samples.var(dim=0, unbiased=False)

		# Restore original mode
		if was_training:
			self.critic.train()
		else:
			self.critic.eval()

		if single_input:
			V_mean = V_mean.squeeze(0)
			V_var = V_var.squeeze(0)
			samples = samples.squeeze(1)

		if return_samples:
			return V_mean, V_var, samples
		return V_mean, V_var

	def evaluate(self, batch_obs, batch_acts, mc_samples=None, training=True):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V_mean - the mean predicted values of batch_obs
				V_var - the predicted variance of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Convert inputs to tensors if needed and move to device
		if isinstance(batch_obs, np.ndarray):
			batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
		elif not torch.is_tensor(batch_obs):
			batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
		batch_obs = batch_obs.to(self.device)

		if isinstance(batch_acts, np.ndarray):
			batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
		elif not torch.is_tensor(batch_acts):
			batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
		batch_acts = batch_acts.to(self.device)

		# Query critic network for MC mean and variance. Shape should match batch_rtgs
		if mc_samples is None:
			mc_samples = self.mc_samples
		V_mean, V_var = self.estimate_critic_uncertainty(
			batch_obs,
			mc_samples=mc_samples,
			return_samples=False,
			require_grad=training
		)

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V_mean, V_var, log_probs

	def inspect_uncertainty_on_states(self, states, mc_samples=10, state_labels=None, verbose=True):
		"""
			Run a simple per-state uncertainty inspection using MC Dropout.

			Parameters:
				states - iterable of states (numpy or torch)
				mc_samples - number of MC dropout samples per state
				state_labels - optional list of labels per state
				verbose - if True, prints a formatted summary

			Return:
				summaries - list of dicts with index, label, mean, and variance

			Example:
				states = [s0, s1, s2, s3]
				labels = ["normal", "dense traffic", "near collision", "late episode"]
				summaries = agent.inspect_uncertainty_on_states(
					states,
					mc_samples=20,
					state_labels=labels,
					verbose=True
				)
		"""
		summaries = []
		for i, state in enumerate(states):
			label = None
			if state_labels is not None and i < len(state_labels):
				label = state_labels[i]

			V_mean, V_var = self.estimate_critic_uncertainty(state, mc_samples=mc_samples, return_samples=False)

			if torch.is_tensor(V_mean) and V_mean.numel() == 1:
				mean_val = float(V_mean.detach().cpu().item())
			else:
				mean_val = V_mean.detach().cpu().numpy()

			if torch.is_tensor(V_var) and V_var.numel() == 1:
				var_val = float(V_var.detach().cpu().item())
			else:
				var_val = V_var.detach().cpu().numpy()

			summary = {
				'index': i,
				'label': label,
				'mean': mean_val,
				'variance': var_val
			}
			summaries.append(summary)

			if verbose:
				label_str = f" | label: {label}" if label is not None else ""
				print(f"[{i}] mean: {mean_val:.6f} | var: {var_val:.6f}{label_str}")

		return summaries

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 3e-4                              # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# MC huperparameters
		self.dropout_p = 0.1
		self.mc_samples = 5
		self.lambda_u = 0.01

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			setattr(self, param, val)

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_raw_ep_rews = np.mean([np.sum(ep_raw_rews) for ep_raw_rews in self.logger['batch_raw_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
		avg_critic_uncertainty =  float(np.mean(self.logger['rollout_uncertainties']))

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 3))
		avg_raw_ep_rews = str(round(avg_raw_ep_rews, 3))
		avg_actor_loss = str(round(avg_actor_loss, 5))
		avg_critic_uncertainty = str(round(avg_critic_uncertainty, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Raw Episodic Return (without uncertainty penalty): {avg_raw_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Average Critic Uncertainty: {avg_critic_uncertainty}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		self.logger['critic_uncertainties'] = [] # deprecated, now using rollout_uncertainties
		self.logger['rollout_uncertainties'] = []
		self.logger['batch_raw_rews'] = []
