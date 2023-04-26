import numpy as np

from src.env import MujocoEnv
from src.agent import Agent

env = MujocoEnv("point_mass", "easy")
agent = Agent(env.spec)

def test_agent_env_interaction():
    timestep = env.reset()
    while not timestep.step_type.last():
        actions = agent(timestep.observation)
        timestep = env.step(actions)

def test_agent_deterministic_during_eval():
    agent.eval()
    observation = env.get_observation()
    action = agent(observation)
    assert (action == agent(observation)).all()

def test_agent_nondeterministic_during_train():
    agent.train()
    observations = np.array([env.get_observation()] * 2)
    action = agent(observations)
    assert not (action == agent(observations)).all()