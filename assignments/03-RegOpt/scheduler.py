from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    This is the CustomLRScheduler class
    """

    def __init__(self, optimizer, last_epoch=-1, lr_factor=0.1, lr_patience=10):
        """
        Create a new scheduler.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            last_epoch (int): The index of last epoch.
            lr_factor (float): The factor by which the learning rate will be reduced.
            lr_patience (int): The number of epochs with no improvement after which learning rate will be reduced.
        """
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.counter = 0
        self.best_loss = float("inf")
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        This is the get_lr method
        """
        """
        Compute the learning rate for each parameter group.

        Returns:
            List[float]: The computed learning rates.

        """
        if self.last_epoch == 0:
            return self.base_lrs

        if self.counter >= self.lr_patience:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.lr_factor
            self.counter = 0
        print([group["lr"] for group in self.optimizer.param_groups])
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch=None, metrics=None):
        """
        Update the learning rate for the next epoch.

        Args:
            epoch (int): The index of the current epoch. If None, uses self.last_epoch + 1. Default: None.
            metrics (dict): A dictionary of metrics for the current epoch. Default: None.

        """
        if metrics is None:
            self.last_epoch += 1
        else:
            loss = metrics.get("loss")
            if loss < self.best_loss:
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += 1

            self.last_epoch = epoch

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
