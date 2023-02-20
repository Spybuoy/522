from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)


class CONFIG:
    batch_size = 256
    num_epochs = 7
    initial_learning_rate = 0.006
    initial_weight_decay = 0.0001
    lr_factor = 0.001
    lr_patience = 10

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "lr_factor": lr_factor,
        "lr_patience": lr_patience,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # RandomHorizontalFlip(),
            # RandomCrop(32, padding=4),
        ]
    )
