"""Definition of the sparse factor analysis as a pyro module."""
import numpy as np

from spexlvm import config

# logging stuff
logger = config.logger


class EarlyStoppingCallback:
    def __init__(
        self,
        n_iterations,
        min_iterations=1000,
        window_size=10,
        tolerance=1e-3,
        patience=1,
    ):
        self.n_iterations = n_iterations
        self.min_iterations = min_iterations
        self.tolerance = tolerance
        self.patience = patience

        self.window_size = window_size
        self.early_stop_counter = 0
        self.min_avg_loss = np.inf

    def __call__(self, loss_history):
        stop_early = False

        iteration_idx = len(loss_history) - 1

        if iteration_idx % self.window_size == 0 and iteration_idx > self.min_iterations:
            current_window_avg_loss = np.mean(loss_history[-self.window_size :])

            relative_improvement = (self.min_avg_loss - current_window_avg_loss) / np.abs(
                current_window_avg_loss
            )

            self.min_avg_loss = min(current_window_avg_loss, self.min_avg_loss)

            if relative_improvement < self.tolerance:
                self.early_stop_counter += 1
            else:
                self.early_stop_counter = 0

            stop_early = self.early_stop_counter >= self.patience

            if stop_early:
                logger.info(
                    f"Relative improvement of {relative_improvement:0.4g} < {self.tolerance:0.4g} for {self.patience} step(s) in a row, stopping early."
                )

        return stop_early
