from dm_control.rl.control import Environment, flatten_observation, FLAT_OBSERVATION_KEY
from dm_control import suite
import numpy as np

from .spec import EnvSpec
from .env import Env

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