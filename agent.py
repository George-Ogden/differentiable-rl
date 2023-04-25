import torch.nn as nn
import numpy as np
import torch

from typing import Union
from env import EnvInteractor, EnvSpec

class Agent(nn.Module, EnvInteractor):
    def __init__(self, env_spec: EnvSpec):
        super().__init__()
        self.setup_spec(env_spec)
        
        # create the network
        self._network = torch.nn.Sequential(
            nn.BatchNorm1d(self._observation_size),
            nn.Linear(self._observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, self._action_size),
            nn.Sigmoid(),
        )
        self.device = torch.device("cpu")
    
    def to(self, device: torch.device):
        # store the device
        self.device = device
        return super().to(device)
    
    def forward(self, observation: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        # convert the observation to a tensor
        numpy = isinstance(observation, np.ndarray)
        if numpy:
            observation = torch.tensor(observation, device=self.device, dtype=torch.float32)
        assert isinstance(observation, torch.Tensor), "observation must be a tensor"

        # normalize the observation
        if self._observation_range:
            observation = (observation - self._observation_range[1]) / torch.diff(self._observation_range, dim=0)
        
        # flatten the observation
        batch_shape = observation.shape[:-len(self._observation_shape)]
        flat_observation = observation.reshape((-1, self._observation_size))
        
        # compute the actions
        actions = self._network(flat_observation)
        
        # unflatten the actions
        actions = actions * torch.diff(self._action_range, dim=0) + self._action_range[0]
        actions = actions.reshape(*batch_shape, *self._action_shape)
        if numpy:
            actions = actions.detach().cpu().numpy()
        return actions