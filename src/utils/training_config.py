import torch.nn.functional as F
import torch.optim as optim

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config

@dataclass
class TrainingConfig(Config):
    iterations: int = 100
    batch_size: int = 1024
    """training batch size"""
    lr: float = 1e-3
    """model learning rate"""
    validation_split: float = 0.1
    """proportion of data to validate on"""
    loss: str = "mse"
    optimizer_type: str = "Adam"
    weight_decay: float = 1e-5

    def optimizer(self, parameters) -> optim.Optimizer:
        """optimizer to use for training"""
        optimizer_class = getattr(optim, self.optimizer_type)
        return optimizer_class(
            params=parameters,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    @property
    def loss_fn(self):
        """loss function to use for training"""
        return getattr(F, f"{self.loss}_loss")