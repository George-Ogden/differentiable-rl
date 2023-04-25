import numpy as np

from dm_env.specs import Array, BoundedArray
from dataclasses import dataclass
from typing import Union

@dataclass
class EnvSpec:
    action_spec: BoundedArray
    observation_spec: Union[Array, BoundedArray]
    def validate_action(self, action: np.ndarray) -> np.ndarray:
        assert action.shape == self.action_spec.shape
        if isinstance(self.action_spec, BoundedArray):
            assert (action >= self.action_spec.minimum).all()
            assert (action <= self.action_spec.maximum).all()
        return action.astype(self.action_spec.dtype)

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        assert observation.shape == self.observation_spec.shape
        if isinstance(self.observation_spec, BoundedArray):
            assert (observation >= self.observation_spec.minimum).all()
            assert (observation <= self.observation_spec.maximum).all()
        return observation.astype(self.observation_spec.dtype)