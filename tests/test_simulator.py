import numpy as np
import torch

from src.simulator import Simulator
from src.env import MujocoEnv

env = MujocoEnv("point_mass", "easy")
spec = env.spec
simulator = Simulator(spec)

history = []
timestep = env.reset()
history.append((timestep, None))
while not timestep.last():
    action = np.random.uniform(
        low=spec.action_spec.minimum,
        high=spec.action_spec.maximum,
    )
    timestep = env.step(action)
    history.append((timestep, action))

simulator.fit([history] * 10)

def test_simulation_observations():
    observations = simulator.reset(4)
    for _ in range(100):
        assert len(observations) == 4
        for observation in observations:
            spec.validate_observation(observation.detach().numpy())
        observations, rewards, values = simulator.step(
            torch.tensor(
                np.random.uniform(
                    low=spec.action_spec.minimum,
                    high=spec.action_spec.maximum,
                    size=(4,) + spec.action_spec.shape,
                ),
                dtype=torch.float32
            )
        )

def test_simulation_rewards():
    observations = simulator.reset(4)
    for _ in range(100):
        observations, rewards, values = simulator.step(
            torch.tensor(
                np.random.uniform(
                    low=spec.action_spec.minimum,
                    high=spec.action_spec.maximum,
                    size=(4,) + spec.action_spec.shape,
                ),
                dtype=torch.float32
            )
        )
        assert rewards.shape == (4,)
        for reward in rewards:
            assert 0 <= reward and reward <= 1

def test_simulation_values():
    observations = simulator.reset(4)
    for _ in range(100):
        observations, rewards, values = simulator.step(
            torch.tensor(
                np.random.uniform(
                    low=spec.action_spec.minimum,
                    high=spec.action_spec.maximum,
                    size=(4,) + spec.action_spec.shape,
                ),
                dtype=torch.float32
            )
        )
        assert values.shape == (4,)
        for value in values:
            assert 0 <= value and value <= 1 / (1 - simulator.discount)