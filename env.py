import torch.nn as nn
import torch

from copy import deepcopy
import numpy as np

from dm_control.rl.control import Environment, flatten_observation, FLAT_OBSERVATION_KEY
from dm_env import StepType, TimeStep
from dm_control import suite

from typing import List, Optional, Tuple, Union
from dm_env.specs import Array, BoundedArray
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

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

class Env(metaclass=ABCMeta):
    spec: EnvSpec = None
    max_round = 0
    discount: Optional[float] = None
    @abstractmethod
    def _get_observation(self)-> np.ndarray:
        ...

    def get_observation(self) -> np.ndarray:
        return self.validate_observation(self._get_observation())

    @abstractmethod
    def _step(self, action: np.ndarray, display: bool = False) -> float:
        ...

    def step(self, action: np.ndarray, display: bool = False) -> TimeStep:
        self.pre_step(action)
        reward = self._step(action, display=display)
        return self.post_step(reward)

    def pre_step(self, action: np.ndarray):
        action = self.validate_action(action)

    def post_step(self, reward: float) -> TimeStep:
        self.current_round += 1
        return TimeStep(
            step_type=self.get_step_type(),
            reward=reward,
            observation=self.get_observation(),
            discount=None if reward is None else self.discount,
        )

    @abstractmethod
    def _reset(self):
        ...

    def reset(self) -> TimeStep:
        self.pre_reset()
        self._reset()
        return self.post_reset()

    def pre_reset(self):
        self.current_round = 0

    def post_reset(self) -> TimeStep:
        return TimeStep(
            step_type=StepType.FIRST,
            reward=None,
            observation=self.get_observation(),
            discount=None
        )

    def clone(self) -> "Self":
        return deepcopy(self)

    def get_symmetries(self, observation: np.ndarray, action: np.ndarray, reward: float) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        return Env.no_symmetries(observation, action, reward)

    @staticmethod
    def no_symmetries(observation: np.ndarray, action: np.ndarray, reward: float) -> List[Tuple[int, np.ndarray, np.ndarray, float]]:
        return [(observation, action, reward)]

    def validate_observation(self, observation: np.ndarray)-> np.ndarray:
        return self.spec.validate_observation(observation)

    def validate_action(self, action: np.ndarray)-> np.ndarray:
        return self.spec.validate_action(action)

    def get_step_type(self, round: Optional[int] = None) -> StepType:
        if round is None:
            round = self.current_round
        if self.current_round >= self.max_round:
            return StepType.LAST
        else:
            return StepType.MID

    def get_random_move(self):
        return np.random.uniform(low=self.spec.action_spec.minimum, high=self.spec.action_spec.maximum)

class MujocoEnv(Env):
    discount = .9
    time_limit = 20.
    timestep = .1
    def __init__(self, domain_name: str, task_name: str):
        super().__init__()
        self.name = (domain_name, task_name)
        self.env: Environment = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={
                "random": True,
                "time_limit": self.time_limit
            },
            environment_kwargs={
                "control_timestep": self.timestep,
                "flat_observation": True
            }
        )
        self.spec = EnvSpec(
            action_spec=self.env.action_spec(),
            observation_spec=self.env.observation_spec()[FLAT_OBSERVATION_KEY]
        )
        self.max_round = int(self.env._step_limit)
        self.reset()
    
    def _reset(self):
        self.env.reset()
    
    def _get_observation(self) -> np.ndarray:
        observation = self.env.task.get_observation(self.env.physics)
        flat_observation = flatten_observation(observation)[FLAT_OBSERVATION_KEY]
        return flat_observation

    def _step(self, action: np.ndarray, display: bool = False) -> float:
        timestep = self.env.step(action)
        if display:
            print(timestep)
        return (timestep.reward or 0)

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