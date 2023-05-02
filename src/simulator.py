import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import torch

from tqdm import trange
import wandb

from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass
from dm_env import TimeStep

from .utils import Config, TrainingConfig
from .env import EnvInteractor, EnvSpec
from .buffer import Buffer

@dataclass
class SimulatorConfig(Config):
    """configuration for the simulator"""
    hidden_size: int = 64
    dropout: float = .1
    discount: float = .9
    gae_lambda: float = .95
    temporary_location: str = "/tmp/model_best.pth"
    buffer_size: int = 200
    epsilon = 1e-5

class Simulator(nn.Module, EnvInteractor):
    """a game that uses an RNN to predict the next state and reward"""
    def __init__(self, env_spec: EnvSpec, config: SimulatorConfig=SimulatorConfig()):
        super().__init__()
        self.setup_spec(env_spec)

        # create the network
        hidden_size = config.hidden_size
        dropout = config.dropout
        self._state_encoder = torch.nn.Sequential(
            nn.BatchNorm1d(self._observation_size),
            nn.Linear(self._observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Dropout(dropout)
        )
        self._action_encoder = torch.nn.Sequential(
            nn.BatchNorm1d(self._action_size),
            nn.Linear(self._action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self._rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self._reward_decoder = torch.nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self._state_decoder = torch.nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self._observation_size),
        )
        self._value_decoder = torch.nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        # store initial states that have been encountered in the past
        self.initial_observations = []
        self.device = torch.device("cpu")

        # store config
        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.temporary_location = config.temporary_location

        self.train_buffer = Buffer(config.buffer_size)
        self.eps = config.epsilon

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
        self._hidden_states = self._state_encoder(starting_states).reshape(2, 1, batch_size, -1)
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
        output_states, self._hidden_states = self._rnn(action_encodings, tuple(self._hidden_states))
        self._hidden_states = [state.detach() * (1 - self.discount) + state * self.discount for state in self._hidden_states]

        # decode outputs
        output_states = output_states.reshape(-1, output_states.shape[-1])
        predicted_observations = self._state_decoder(output_states)
        predicted_rewards = self._reward_decoder(output_states).squeeze(-1)
        predicted_values = self._value_decoder(output_states).squeeze(-1) / (1 - self.discount)

        return predicted_observations, predicted_rewards, predicted_values

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
        hidden_states = self._state_encoder(initial_observations).reshape(2, 1, np.prod(batch_shape[:-1]), -1)
        actions = actions.reshape(-1, self._action_size)
        action_encodings = self._action_encoder(actions)
        action_encodings = action_encodings.reshape(*batch_shape, action_encodings.shape[-1])

        # pass through RNN
        output_states, _ = self._rnn(action_encodings, tuple(hidden_states))

        # decode outputs
        output_states = output_states.reshape(-1, output_states.shape[-1])
        predicted_observations = self._state_decoder(output_states)
        predicted_observations = predicted_observations.reshape(*batch_shape, *self._observation_shape)
        if self._observation_range is not None:
            predicted_observations = F.sigmoid(predicted_observations) * torch.diff(self._observation_range, dim=0).squeeze(0) + self._observation_range[0]
        predicted_rewards = self._reward_decoder(output_states).squeeze(-1)
        predicted_rewards = predicted_rewards.reshape(*batch_shape)
        predicted_values = self._value_decoder(output_states).squeeze(-1)
        predicted_values = predicted_values.reshape(*batch_shape)
        return predicted_observations, predicted_rewards, predicted_values / (1 - self.discount)

    def compute_loss(
        self,
        batch: Tuple[torch.Tensor, ...],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """compute the loss on a batch of data

        Args:
            batch (Tuple[torch.Tensor, ...]): batch of data
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): loss function

        Returns:
            torch.Tensor: loss
        """
        initial_observations, actions, observations, rewards, values = batch
        predicted_observations, predicted_rewards, predicted_values = self(initial_observations, actions)
        # calculate loss
        return loss_fn(
            predicted_observations,
            observations.to(predicted_observations.device),
            reduction="none"
        ).mean(axis=tuple(range(1, observations.ndim))) + loss_fn(
            predicted_rewards,
            rewards.to(predicted_rewards.device),
            reduction="none"
        ).mean(axis=tuple(range(1, rewards.ndim))) + loss_fn(
            # normalise values
            predicted_values * (1 - self.discount),
            values.to(predicted_values.device) * (1 - self.discount),
            reduction="none"
        ).mean(axis=tuple(range(1, values.ndim)))

    def fit(
        self,
        experience_history: List[List[Tuple[TimeStep, Optional[np.ndarray]]]],
        training_config: TrainingConfig=TrainingConfig()
    ):
        """fit the model to the experience history"""
        # convert raw experiences into dataset
        initial_observations = [
            timestep.observation for episode in experience_history
            for timestep, _ in episode[:1]
        ]
        # use initial states for running simulations
        self.initial_observations.extend(initial_observations)

        # convert data to tensors
        initial_observations, actions, observations, rewards = self.prepare(
            initial_observations,
            *zip(*(
                zip(*(
                    (action, timestep.observation, timestep.reward)
                    for timestep, action in episode[1:]
                ))
                for episode in experience_history
            ))
        )
        
        # create dataset
        dataset = data.TensorDataset(
            initial_observations, actions, observations, rewards
        )

        batch_size = training_config.batch_size
        pre_dataloader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        self.eval()
        with torch.no_grad():
            # calculate TD-lambda targets
            *_, target_values, = zip(*[
                self(initial_observations, actions)
                for initial_observations, actions, *_ in pre_dataloader
            ])
            target_values = torch.cat(target_values, dim=0).T
            
            # compute rewards with GAE lambda
            lambda_reward = target_values[-1]
            values = torch.stack([
                (
                    lambda_reward := (
                        reward.to(self.device) + self.discount * (
                            self.gae_lambda * lambda_reward
                            + next_value_prediction
                        ) - value_prediction
                    ) - value_prediction
                ) + value_prediction
                for reward, value_prediction, next_value_prediction
                in reversed(
                    list(
                        zip(
                            rewards.T,
                            target_values,
                            torch.cat([target_values[1:], target_values[-1:]], dim=0)
                        )
                    )
                )
            ])
        dataset = data.TensorDataset(
            initial_observations, actions, observations, rewards, values.T
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
        pretrain_dataloader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        with torch.no_grad():
            for batch in pretrain_dataloader:
                loss = self.compute_loss(batch, loss_fn)
                self.train_buffer.add(loss.cpu().numpy() + self.eps, list(zip(*batch)))
        val_dataloader = data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        best_val_loss = float("inf")
        # save as a precaution
        torch.save(self.state_dict(), self.temporary_location)
        self.train()
        train_loss = 0.
        for _ in trange(training_config.iterations, desc="Improving simulator"):
            batch, ids, importances = self.train_buffer.sample(batch_size)
            # prepare data
            batch = data.default_collate(list(batch))
            importances = torch.tensor(importances, dtype=torch.float32, device=self.device)
            # train on batch
            optimiser.zero_grad()
            loss = (self.compute_loss(batch, loss_fn) * importances).sum() / importances.sum()
            train_loss += loss.item()
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                updated_loss = self.compute_loss(batch, loss_fn)
                self.train_buffer.update(ids, updated_loss.cpu().numpy() + self.eps)
        train_loss /= training_config.iterations or 1.

        val_loss = 0.
        self.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                # validate on batch
                loss = self.compute_loss(batch, loss_fn).mean()
                val_loss += loss.item()
        val_loss /= len(val_dataloader) or 1.

        # save best model to temporary location
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(self.state_dict(), self.temporary_location)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        # load best model
        self.load_state_dict(torch.load(self.temporary_location))