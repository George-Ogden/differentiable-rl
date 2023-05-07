import torch.nn.functional as F
import torch.optim as optim

from ml_utils import DefaultTrainingConfig
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class TrainingConfig(DefaultTrainingConfig):
    epochs: ClassVar[int] = 0
    iterations: int = 100
    batch_size: int = 1024
    """training batch size"""
    loss: str = "mse"
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