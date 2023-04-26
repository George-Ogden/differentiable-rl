import numpy as np
import torch
import wandb

def pytest_configure(config):
    # disabel wandb logging
    wandb.init("test run", mode="disabled")

    # seed for repeatability
    np.random.seed(0)
    torch.manual_seed(0)

    # remove cuda
    torch.backends.cudnn.deterministic = True
    torch.cuda.is_available = lambda: False