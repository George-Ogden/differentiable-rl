from copy import deepcopy
import numpy as np

from typing import List, Optional, Tuple
from abc import ABCMeta, abstractmethod
from dm_env import StepType, TimeStep

from .spec import EnvSpec

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