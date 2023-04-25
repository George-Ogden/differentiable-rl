import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from dm_env.specs import BoundedArray
from typing import Union
from env import EnvSpec

class Agent(nn.Module):
    def __init__(self, game_spec: EnvSpec):
        super().__init__()
        # store the observation and action specs
        self._observation_spec = game_spec.observation_spec
        self._action_spec = game_spec.action_spec
        # save the shape of the observation and action
        self._observation_shape = self._observation_spec.shape
        self._action_shape = self._action_spec.shape
        # precompute the size of the observation and action
        self._observation_size = np.prod(self._observation_shape)
        self._action_size = np.prod(self._action_shape)
        # save observation and action ranges
        self._observation_range = nn.Parameter(
            torch.tensor(
                np.stack(
                    (self._observation_spec.minimum, self._observation_spec.maximum),
                    axis=0
                )
            ),
            requires_grad=False
        ) if isinstance(self._observation_spec, BoundedArray) else None
        self._action_range = nn.Parameter(
            torch.tensor(
                np.stack(
                    (self._action_spec.minimum, self._action_spec.maximum),
                    axis=0
                ),
                dtype=torch.float32
            ),
            requires_grad=False
        )
        
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