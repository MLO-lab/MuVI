"""Collection of training callbacks."""
import numpy as np


class EarlyStoppingCallback:
    def __init__(
        self,
        min_epochs: int = 100,
        tolerance: float = 1e-5,
        patience: int = 10,
    ):
        """Early stopping callback.

        Parameters
        ----------
        min_epochs : int, optional
            Minimal number of epochs before deploying early stopping,
            by default 100
        tolerance : float, optional
            Improvement ratio between two consecutive evaluations,
            by default 1e-5
        patience : int, optional
            Number of patience steps before terminating training,
            by default 10
        """
        self.min_epochs = min_epochs
        self.tolerance = tolerance
        self.patience = patience

        self.early_stop_counter = 0
        self.min_loss = np.inf

    def __call__(self, loss_history):
        epoch_idx = len(loss_history) - 1
        if epoch_idx < self.min_epochs:
            return False

        current_loss = loss_history[epoch_idx]
        if self.min_loss == np.inf:
            self.min_loss = current_loss
            return False

        stop_early = False

        relative_improvement = (self.min_loss - current_loss) / np.abs(self.min_loss)
        self.min_loss = min(self.min_loss, current_loss)

        if relative_improvement < self.tolerance:
            self.early_stop_counter += 1
        else:
            self.early_stop_counter = 0
        stop_early = self.early_stop_counter >= self.patience

        if stop_early:
            print(
                "Relative improvement of "
                f"{relative_improvement:0.4g} < {self.tolerance:0.4g} "
                f"for {self.patience} step(s) in a row, stopping early."
            )

        return stop_early
