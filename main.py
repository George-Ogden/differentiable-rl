from simulator import Simulator
from agent import Agent
from env import MujocoEnv

import wandb

# load environment
env = MujocoEnv("cartpole", "swingup")
# create agent
agent = Agent(env.spec).to("cuda").eval()
# create simulator
simulator = Simulator(env.spec)

wandb.init(project="simulator")

for iteration in range(100):
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
        wandb.log({"reward": sum([(timestep.reward or 0.) for timestep, _ in episode_history])}, step=iteration)
    
    # train simulator
    simulator.fit(experience_history)