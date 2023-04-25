import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import torch

from env import EnvInteractor

from typing import Optional, List, Tuple
from training import TrainingConfig
from dm_env import TimeStep
from env import EnvSpec
from tqdm import trange

import wandb

class Simulator(nn.Module, EnvInteractor):
    """a game that uses an RNN to predict the next state and reward"""
    def __init__(self, env_spec: EnvSpec):
        super().__init__()
        self.setup_spec(env_spec)

        self._state_encoder = torch.nn.Sequential(
            nn.BatchNorm1d(self._observation_size),
            nn.Linear(self._observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self._action_encoder = torch.nn.Sequential(
            nn.BatchNorm1d(self._action_size),
            nn.Linear(self._action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self._rnn = nn.GRU(64, 64, batch_first=True)
        self._reward_decoder = torch.nn.Sequential(
            nn.Linear(64, 1),
        )
        self._state_decoder = torch.nn.Sequential(
            nn.Linear(64, self._observation_size),
        )
        # store initial states that have been encountered in the past
        self.initial_observations = []
        self.device = torch.device("cpu")

    def prepare(self, *data):
        """convert data to tensors"""
        # convert to numpy array first for speedup
        return (torch.tensor(np.array(d)).float() for d in data)

    def to(self, device: torch.device):
        # store the device
        self.device = device
        return super().to(device)

    def reset(self, batch_size: int=1) -> torch.Tensor:
        """reset the simulator to a random initial state

        Args:
            batch_size (int, optional): number of parallel simulations to run. Defaults to 1.

        Returns:
            torch.Tensor: observation
        """
        # RNN backward can only be called in training mode
        self._rnn.train()

        # pick random starting states
        starting_states = torch.tensor(
            np.array([
                self.initial_observations[
                    np.random.randint(len(self.initial_observations))
                ] for _ in range(batch_size)
            ]),
            device=self.device,
            dtype=torch.float32
        )
        # encode the starting states
        self._hidden_states = self._state_encoder(starting_states).unsqueeze(0)
        return starting_states

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """advance the simulation

        Args:
            actions (torch.Tensor): agent's actions

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: observations and rewards from the environment
        """
        # encode actions
        actions = actions.to(self.device)
        actions = actions.reshape(-1, self._action_size)
        action_encodings = self._action_encoder(actions)
        action_encodings = action_encodings.unsqueeze(1)

        # pass through RNN
        output_states, self._hidden_states = self._rnn(action_encodings, self._hidden_states)

        # decode outputs
        output_states = output_states.reshape(-1, output_states.shape[-1])
        predicted_observations = self._state_decoder(output_states)
        predicted_rewards = self._reward_decoder(output_states).squeeze(-1)

        return predicted_observations, predicted_rewards

    def forward(
        self,
        initial_observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """generate a sequence of states and rewards given a sequence of actions and initial states

        Args:
            initial_observations (torch.Tensor): observations from env.reset()
            actions (torch.Tensor): actions from env.step()

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: states and rewards
        """
        batch_shape = actions.shape[:-1]
        initial_observations = initial_observations.to(self.device)
        actions = actions.to(self.device)

        # encode inputs
        hidden_states = self._state_encoder(initial_observations).unsqueeze(0)
        actions = actions.reshape(-1, self._action_size)
        action_encodings = self._action_encoder(actions)
        action_encodings = action_encodings.reshape(*batch_shape, action_encodings.shape[-1])

        # pass through RNN
        output_states, _ = self._rnn(action_encodings, hidden_states)

        # decode outputs
        output_states = output_states.reshape(-1, output_states.shape[-1])
        predicted_observations = self._state_decoder(output_states)
        predicted_observations = predicted_observations.reshape(*batch_shape, *self._observation_shape)
        predicted_rewards = self._reward_decoder(output_states).squeeze(-1)
        predicted_rewards = predicted_rewards.reshape(*batch_shape)
        return predicted_observations, predicted_rewards

    def fit(
        self,
        experience_history: List[List[Tuple[TimeStep, Optional[np.ndarray]]]],
        training_config: TrainingConfig=TrainingConfig()
    ):
        """fit the model to the experience history"""
        # convert raw experiences into dataset
        initial_observations = [
            timestep.observation for episode in experience_history
            for timestep, _ in [episode.pop(0)]
        ]
        # use initial states for running simulations
        self.initial_observations.extend(initial_observations)

        # convert data to tensors
        initial_observations, actions, observations, rewards = self.prepare(
            initial_observations,
            *zip(*(
                zip(*(
                    (action, timestep.observation, timestep.reward)
                    for timestep, action in episode
                ))
                for episode in experience_history
            ))
        )
        dataset = torch.utils.data.TensorDataset(
            initial_observations, actions, observations, rewards
        )

        # create optimiser and loss
        optimiser = training_config.optimizer(self.parameters())
        loss_fn = training_config.loss_fn

        # create dataloaders
        indices = np.random.permutation(len(dataset))
        train_dataset = data.Subset(
            dataset, indices[int(len(dataset) * training_config.validation_split):]
        )
        val_dataset = data.Subset(
            dataset, indices[:int(len(dataset) * training_config.validation_split)]
        )
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=training_config.batch_size, shuffle=True
        )
        val_dataloader = data.DataLoader(
            val_dataset, batch_size=training_config.batch_size, shuffle=False
        )

        for epoch in trange(training_config.epochs, desc="Improving simulator"):
            self.train()
            train_loss = 0.
            for initial_observations, actions, observations, rewards in train_dataloader:
                # train on batch
                optimiser.zero_grad()

                predicted_observations, predicted_rewards = self(initial_observations, actions)

                # calculate loss
                loss = loss_fn(
                    predicted_observations,
                    observations.to(predicted_observations.device)
                ) + loss_fn(
                    predicted_rewards,
                    rewards.to(predicted_rewards.device)
                )
                train_loss += loss.item()
                loss.backward()

                optimiser.step()
            train_loss /= len(train_dataloader)

            val_loss = 0.
            self.eval()
            with torch.no_grad():
                for initial_observations, actions, observations, rewards in val_dataloader:
                    # validate on batch
                    predicted_observations, predicted_rewards = self(initial_observations, actions)

                    loss = loss_fn(
                        predicted_observations,
                        observations.to(predicted_observations.device)
                    ) + loss_fn(
                        predicted_rewards,
                        rewards.to(predicted_rewards.device)
                    )
                    val_loss += loss.item()
            val_loss /= len(val_dataloader)
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})