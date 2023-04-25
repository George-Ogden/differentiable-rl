from dm_control import viewer

from env import MujocoEnv

import torch

agent = torch.load("agent.pth")
env = MujocoEnv("cartpole", "swingup")
env.reset()
agent.eval()

reward = 0
episode_history = []
timestep = env.reset()
episode_history.append((timestep, None))

while not timestep.last():
    action = agent(timestep.observation)
    timestep = env.step(action)
    reward += timestep.reward or 0.
print(reward)

viewer.launch(env.env, policy=lambda observation: agent(observation.observation["observations"]))