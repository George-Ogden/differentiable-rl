from src.simulator import Simulator
from src.agent import Agent
from src.env import MujocoEnv

import numpy as np
import torch
import wandb

# load environment
env = MujocoEnv("cartpole", "swingup")
# create agent
agent = Agent(env.spec).to("cuda")
# create simulator
simulator = Simulator(env.spec).to("cuda")

wandb.init(project="simulator")

for iteration in range(5):
    agent.eval()
    # gather experience using agent
    experience_history = []
    for episode in range(10):
        episode_history = []
        timestep = env.reset()
        episode_history.append((timestep, None))

        while not timestep.last():
            action = agent(timestep.observation)
            timestep = env.step(action)
            episode_history.append((timestep, action))

        experience_history.append(episode_history)
    rewards = [
        sum([(timestep.reward or 0.) for timestep, _ in episode_history])
    ]
    wandb.log({"env_reward": np.mean(rewards)})

    # train simulator
    simulator.fit(experience_history)
    agent.learn(simulator)
    torch.save(agent, "agent.pth")