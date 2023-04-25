import torch.nn as nn
import numpy as np
import torch

from dm_env.specs import BoundedArray
from .spec import EnvSpec

class EnvInteractor:
    def setup_spec(self, spec: EnvSpec):
        assert isinstance(self, nn.Module), "EnvInteractor must be a nn.Module"
        # store the observation and action specs
        self._observation_spec = spec.observation_spec
        self._action_spec = spec.action_spec
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