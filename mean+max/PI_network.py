"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class PIFeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self,out_dim, num_of_features, num_of_vehicles):
		"""
			Initialize the network and set up the layers.

			Parameters:
				out_dim - output dimensions as an int
				num_of_features - the number of features each vehicle has
				num_of_vehicles - the number of vehicles in the observation matrix

			Return:
				None
		"""
		super(PIFeedForwardNN, self).__init__()

		self.num_of_features = num_of_features
		self.num_of_vehicles = num_of_vehicles
		self.vehicle_obs_dim = num_of_features * num_of_vehicles

		self.urgency_dim = 8
		self.urgency_hidden_dim = 32

		self.urgency_layer1 = nn.Linear(self.urgency_dim, self.urgency_hidden_dim)
		self.urgency_layer2 = nn.Linear(self.urgency_hidden_dim, self.urgency_hidden_dim)


		self.layer1 = nn.Linear(num_of_features, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(192 + self.urgency_hidden_dim, out_dim)


	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float32)
		elif not torch.is_tensor(obs):
			obs = torch.tensor(obs, dtype=torch.float32)
		else:
			obs = obs.float()

		single_input = (obs.dim() == 1)

		if single_input:
			obs = obs.unsqueeze(0)

		vehicle_obs = obs[:, :self.vehicle_obs_dim]
		urgency_obs = obs[:,self.vehicle_obs_dim:]

		if urgency_obs.size(1) != self.urgency_dim:
			raise ValueError(f"Expected urgency_obs to have {self.urgency_dim} features, but got {urgency_obs.size(1)}")


		obs = vehicle_obs.reshape(-1,self.num_of_vehicles,self.num_of_features)
		prescence_mask = obs[:, 1:, 0]
		num_of_real_vehicles = prescence_mask.sum(dim=1)
		prescence_mask = prescence_mask.unsqueeze(-1).repeat(1,1,64)

		ego_row = obs[:,0,:]
		non_ego_rows = obs[:, 1:, ]
		set_encoders = []

		for i in range(non_ego_rows.size(1)):
			vehicle = non_ego_rows[:, i, :]
			vehicle_embedding = self.set_encoder_value(vehicle)
			set_encoders.append(vehicle_embedding)

		ego_set_encoder = self.set_encoder_value(ego_row)
		set_encoders = torch.stack(set_encoders, dim=1)
		masked_set_encoders = set_encoders * prescence_mask
		mean_traffic_embedding = masked_set_encoders.sum(dim=1) / num_of_real_vehicles.unsqueeze(-1).clamp(min=1.0)

		negative_matrix = torch.full_like(set_encoders,float("-inf"))
		masked_max_input = torch.where(prescence_mask.bool(), set_encoders, negative_matrix) 
		max_traffic_embedding = masked_max_input.max(dim=1)[0]

		has_vehicle = (num_of_real_vehicles > 0).unsqueeze(-1)
		max_traffic_embedding = torch.where(
			has_vehicle,
			max_traffic_embedding,
			torch.zeros_like(max_traffic_embedding)
		)

		traffic_embedding = torch.cat((mean_traffic_embedding, max_traffic_embedding), dim=1)
		pooled_embedding = torch.cat((ego_set_encoder, traffic_embedding), dim=1)

		urgency_embedding = F.relu(self.urgency_layer1(urgency_obs))
		urgency_embedding = F.relu(self.urgency_layer2(urgency_embedding))

		combined_embedding = torch.cat((pooled_embedding, urgency_embedding), dim=1)

		output = self.layer3(combined_embedding)

		if single_input:
			output = output.squeeze(0)
			
		return output


	def set_encoder_value(self,vehicle):
		activation1 = F.relu(self.layer1(vehicle))
		activation2 = F.relu(self.layer2(activation1))
		return activation2
