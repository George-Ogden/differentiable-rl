from dataclasses import dataclass, field
from typing import List, Optional

import torch.optim as optim

@dataclass
class TrainingConfig:
    training_epochs: int = 20
    """number of epochs to train each model for"""
    batch_size: int = 64
    """training batch size"""
    training_patience: Optional[int] = 7
    """number of epochs without improvement during training (0 to ignore)"""
    lr: float = 1e-2
    """model learning rate"""
    validation_split: float = 0.1
    """proportion of data to validate on"""
    loss: str = "mse"
    optimizer_type: str = "Adam"
    metrics: List[str] = field(default_factory=lambda: ["mae"])

    @property
    def optimizer(self, parameters) -> optim.Optimizer:
        """optimizer to use for training"""
        optimizer_class = getattr(optim, self.optimizer_type)
        return optimizer_class(self.optimizer_type)(
            params=parameters,
            learning_rate=self.lr
        )
