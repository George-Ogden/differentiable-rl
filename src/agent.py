import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from dataclasses import dataclass
from typing import Union

from tqdm import trange
import wandb

from .env import EnvInteractor, EnvSpec, MujocoEnv
from .utils import Config, TrainingConfig
from .simulator import Simulator

@dataclass
class AgentConfig(Config):
    """configuration for the agent"""
    hidden_size: int = 64
    dropout: float = .1
    similarity_coeff: float = 1.
    selection_length: int = 10

class Agent(nn.Module, EnvInteractor):
    def __init__(
        self,
        env_spec: EnvSpec,
        config: AgentConfig=AgentConfig()
    ):
        super().__init__()
        self.setup_spec(env_spec)

        # create the network
        self._feature_extractor = torch.nn.Sequential(
            nn.BatchNorm1d(self._observation_size),
            nn.Linear(self._observation_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self._action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self._action_size),
            nn.Sigmoid(),
        )
        self.device = torch.device("cpu")
        self.eval()

        # store the config
        self.similarity_coeff = config.similarity_coeff
        self.selection_length = config.selection_length

    def to(self, device: torch.device):
        # store the device
        self.device = device
        return super().to(device)

    def forward(self, observation: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        # convert the observation to a tensor
        numpy = isinstance(observation, np.ndarray)
        if numpy:
            observation = torch.tensor(observation, device=self.device, dtype=torch.float32)
        assert isinstance(observation, torch.Tensor), "observation must be a tensor"

        # normalize the observation
        if self._observation_range:
            # scale the observation to [-1, 1]
            observation = (observation - self._observation_range[1]) / torch.diff(self._observation_range, dim=0).squeeze(0) * 2 - 1

        # flatten the observation
        batch_shape = observation.shape[:-len(self._observation_shape)]
        flat_observation = observation.reshape((-1, self._observation_size))

        # compute the actions
        features = self._feature_extractor(flat_observation)
        actions = self._action_head(features)

        # unflatten the actions
        actions = actions.reshape(*batch_shape, *self._action_shape)
        actions = actions * torch.diff(self._action_range, dim=0).squeeze(0) + self._action_range[0]
        if numpy:
            actions = actions.detach().cpu().numpy()
        return actions

    def learn(
        self,
        simulator: Simulator,
        training_config: TrainingConfig=TrainingConfig(),
    ):
        """train the agent on the simulator"""
        simulator.eval()
        self.train()
        optimizer = training_config.optimizer(self.parameters())

        # create a target model
        target_model = Agent(self.spec).to(self.device)
        target_model.load_state_dict(self.state_dict())
        target_model.eval()

        batch_size = training_config.batch_size
        discount = simulator.discount

        for epoch in trange(training_config.iterations, desc="Practising in simulator"):
            optimizer.zero_grad()
            # reset the simulator
            observations = simulator.reset(training_config.batch_size)
            num_steps = int(MujocoEnv.time_limit / MujocoEnv.timestep)
            episode_lengths = torch.randint(self.selection_length, num_steps, (training_config.batch_size,), device=self.device)
            reward_coefficients = torch.zeros((num_steps, batch_size), device=self.device)
            reward_coefficients[:self.selection_length] = torch.stack([discount ** torch.arange(self.selection_length, device=self.device)] * batch_size, axis=1)
            for i in range(batch_size):
                reward_coefficients[:, i] = torch.roll(reward_coefficients[:, i], (episode_lengths[i] - self.selection_length).item())
            
            total_value = 0
            divergence_loss = 0
            for i in range(episode_lengths.max() + 1):
                # sample actions
                actions = self(observations)
                with torch.no_grad():
                    target_actions = target_model(observations)
                observations, rewards, values = simulator.step(actions)
                divergence_loss += F.mse_loss(actions, target_actions)
                total_value += (rewards * reward_coefficients[i]).mean()
                total_value += values[episode_lengths == i].sum() / batch_size * discount ** (self.selection_length + 1)

            # update agent
            loss = -total_value / num_steps + divergence_loss * self.similarity_coeff
            loss.backward()

            optimizer.step()

            wandb.log({"sim_value": total_value.item() / num_steps})