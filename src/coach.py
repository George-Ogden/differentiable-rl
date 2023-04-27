import numpy as np
import torch

from tqdm import tqdm, trange
import wandb
import os

from dataclasses import dataclass, field
from typing import Union

from .simulator import Simulator, SimulatorConfig
from .utils import Config, TrainingConfig
from .agent import Agent, AgentConfig
from .env import MujocoEnv

@dataclass
class CoachConfig(Config):
    num_iterations: int = 100
    """number of iterations to train for"""
    agent_env_experiences: int = 25
    """number of experiences the agent should gather before training"""
    simulator_epochs: int = 30
    """number of epochs to train the simulator for"""
    agent_sim_experiences: int = 20
    """number of experiences the agent should run in the simulator"""
    evaluation_runs: int = 10
    """number of simulations to run when evaluating"""
    save_directory: str = "output"
    """directory to save logs, model, files, etc. to"""
    sub_directories: str = "iteration-{iteration:04}"
    best_checkpoint_path: str = "best"
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    simulator_config: SimulatorConfig = field(default_factory=SimulatorConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        assert self.agent_env_experiences > 1, "`agent_env_experiences` must be greater than 1 (batch norm errors)"

class Coach:
    def __init__(self, env: MujocoEnv, config: CoachConfig=CoachConfig()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.agent = Agent(env.spec, config=config.agent_config).to(self.device)
        self.simulator = Simulator(env.spec, config=config.simulator_config).to(self.device)
        self.best_reward = -np.inf
        
        # set config
        self.config = config
        self.eval_envs = [self.env.clone() for _ in range(config.evaluation_runs)]

        self.num_iterations = config.num_iterations
        self.agent_env_experiences = config.agent_env_experiences
        self.agent_sim_experiences = config.agent_sim_experiences
        
        self.save_directory = config.save_directory
        self.sub_directories = config.sub_directories
        self.best_checkpoint_path = config.best_checkpoint_path
        
        self.training_config = config.training_config
        self.simulator_epochs = config.simulator_epochs

    def run(self):
        self.agent.eval()
        self.save_agent(0)
        self.save_simulator(0)
        for iteration in range(self.num_iterations):
            # gather experience using agent
            experience_history = []
            for episode in trange(self.agent_env_experiences, desc="Collecting experience in environment"):
                episode_history = []
                timestep = self.env.reset()
                episode_history.append((timestep, None))

                while not timestep.last():
                    action = self.agent(timestep.observation)
                    timestep = self.env.step(action)
                    episode_history.append((timestep, action))

                experience_history.append(episode_history)
            rewards = [
                sum([(timestep.reward or 0.) for timestep, _ in episode_history])
            ]
            wandb.log({"env_reward": np.mean(rewards)})

            # train simulator
            self.training_config.epochs = self.simulator_epochs
            self.simulator.fit(experience_history, training_config=self.training_config)
            self.save_simulator(iteration + 1)

            # train agent
            self.training_config.epochs = self.agent_sim_experiences
            self.agent.learn(self.simulator, training_config=self.training_config)
            self.save_agent(iteration + 1)

            # evaluate
            eval_envs = [env.clone() for env in self.eval_envs]
            rewards = []
            self.agent.eval()
            for env in tqdm(eval_envs, desc="Evaluating in environment"):
                timestep = env.reset()
                total_reward = 0.
                while not timestep.last():
                    action = self.agent(timestep.observation)
                    timestep = env.step(action)
                    total_reward += timestep.reward or 0.
                rewards.append(total_reward)
                wandb.log({"eval_reward": np.mean(total_reward)})
            
            # save best model
            if np.mean(rewards) > self.best_reward:
                self.best_reward = np.mean(rewards)
                self.save_agent(self.best_checkpoint_path)
                self.save_simulator(self.best_checkpoint_path)

    def save_agent(self, iteration: Union[int, str]):
        directory = os.path.join(self.save_directory, self.sub_directories.format(iteration=iteration))
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "agent.pth")
        torch.save(self.agent, path)
    
    def save_simulator(self, iteration: Union[int, str]):
        directory = os.path.join(self.save_directory, self.sub_directories.format(iteration=iteration))
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "simulator.pth")
        torch.save(self.simulator, path)