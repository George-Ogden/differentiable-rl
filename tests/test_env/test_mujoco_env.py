import numpy as np

import pytest

from src.env import Env, MujocoEnv
from src.agent import Agent

easy_env = MujocoEnv("point_mass", "easy")
hard_env = MujocoEnv("cartpole", "swingup_sparse")
env = MujocoEnv("pendulum", "swingup")

agent = Agent(env.spec)

def test_env_is_env():
    assert isinstance(env, Env)

def test_env_terminates():
    timestep = env.reset()
    assert timestep.step_type.first()
    for _ in range(env.max_round - 1):
        actions = agent(timestep.observation)
        timestep = env.step(actions)
        assert timestep.step_type.mid()
    actions = agent(timestep.observation)
    timestep = env.step(actions)
    assert timestep.step_type.last()

def test_discount():
    timestep = env.reset()
    assert timestep.discount is None
    for _ in range(env.max_round):
        timestep = env.step(agent(timestep.observation))
        assert timestep.discount < 1.

def test_valid_actions_are_valid():
    env = easy_env
    agent = Agent(env.spec)
    for i in range(1000):
        env.validate_action(agent(env.get_observation()))
    pytest.raises(AssertionError, env.validate_action, np.array((.5,)))
    pytest.raises(AssertionError, env.validate_action, np.array(.5))
    pytest.raises(AssertionError, env.validate_action, np.array((0, 1., 0)))
    pytest.raises(AssertionError, env.validate_action, np.array((-2., 0.)))
    pytest.raises(AssertionError, env.validate_action, np.array((0., -2.)))
    pytest.raises(AssertionError, env.validate_action, np.array((2., 0)))
    pytest.raises(AssertionError, env.validate_action, np.array((0., 2.)))
    pytest.raises(AssertionError, env.validate_action, np.array((-2., 2.)))
    pytest.raises(AssertionError, env.validate_action, np.array((-2., -2.)))
    pytest.raises(AssertionError, env.validate_action, np.array((2., -2.)))
    pytest.raises(AssertionError, env.validate_action, np.array((2., 2.)))

def test_valid_observations_are_valid():
    observation = env.reset().observation
    assert (observation == env.get_observation()).all()
    env.validate_observation(observation)
    assert observation.dtype == env.spec.observation_spec.dtype

    assert env.get_observation().dtype == env.spec.observation_spec.dtype
    env.validate_observation(env.get_observation())

    observation = env.step(agent(env.get_observation())).observation
    assert (observation == env.get_observation()).all()
    env.validate_observation(observation)
    assert observation.dtype == env.spec.observation_spec.dtype

    assert env.get_observation().dtype == env.spec.observation_spec.dtype
    env.validate_observation(env.get_observation())