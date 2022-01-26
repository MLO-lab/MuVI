from collections import defaultdict

import numpy as np

from spexlvm import config
from spexlvm.utils import compute_cf_scores_at, compute_factor_relevance

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

        if (
            iteration_idx % self.window_size == 0
            and iteration_idx > self.min_iterations
        ):
            current_window_avg_loss = np.mean(loss_history[-self.window_size :])

            relative_improvement = (
                self.min_avg_loss - current_window_avg_loss
            ) / np.abs(current_window_avg_loss)

            self.min_avg_loss = min(current_window_avg_loss, self.min_avg_loss)

            if relative_improvement < self.tolerance:
                self.early_stop_counter += 1
            else:
                self.early_stop_counter = 0

            stop_early = self.early_stop_counter >= self.patience

            if stop_early:
                logger.info(
                    f"Relative improvement of {relative_improvement:0.4g} < "
                    f"{self.tolerance:0.4g} for {self.patience} step(s) "
                    "in a row, stopping early."
                )

        return stop_early


class CheckpointCallback:
    def __init__(
        self,
        model,
        n_iterations,
        n_checkpoints=10,
        callback=None,
    ):
        self.model = model
        self.n_iterations = n_iterations
        self.n_checkpoints = n_checkpoints
        self.window_size = int(n_iterations / n_checkpoints)
        self.callback = callback

    def __call__(self, loss_history):
        iteration_idx = len(loss_history)

        if iteration_idx % self.window_size == 0 or iteration_idx == self.n_iterations:
            self.callback()

        return False


class DebugCallback:
    # _CALLBACK_KEYS = ["sparsity", "view_sigma", "r2", "binary_scores",  "log"]
    _CALLBACK_KEYS = ["sparsity", "r2", "binary_scores", "log"]

    def __init__(
        self, model, n_iterations, n_checkpoints=10, active_callbacks=None, **kwargs
    ) -> None:
        self.kwargs = kwargs
        self.model = model
        self.n_iterations = n_iterations
        self.n_checkpoints = n_checkpoints

        if active_callbacks is None:
            active_callbacks = self._CALLBACK_KEYS
        self.callback_config = {k: self._default_config[k] for k in active_callbacks}
        self.callback_dict = {
            k: CheckpointCallback(
                self.model, self.n_iterations, v["n_checkpoints"], v["callback"]
            )
            for k, v in self.callback_config.items()
        }
        self.threshold = kwargs.get("threshold", 0.05)
        self.should_log = kwargs.get("log", False)

        self.ws = []
        self.xs = []
        self.scores = defaultdict(list)

    @property
    def _default_config(self):
        return {
            # "plot": {"n_checkpoints": self.n_checkpoints, "callback": self.plot},
            "sparsity": {
                "n_checkpoints": self.n_checkpoints,
                "callback": self.sparsity,
            },
            "view_sigma": {
                "n_checkpoints": self.n_checkpoints,
                "callback": self.view_sigma,
            },
            "r2": {"n_checkpoints": self.n_checkpoints, "callback": self.r2},
            "binary_scores": {
                "n_checkpoints": self.n_checkpoints,
                "callback": self.binary_scores,
            },
            "log": {
                "n_checkpoints": self.kwargs.get("log_frequency", 100),
                "callback": self.log,
            },
        }

    @property
    def w(self):
        return self.model.get_w()

    def get_w_activations(self, threshold=0.05):
        return np.abs(self.w) > threshold

    @property
    def x(self):
        return self.model.get_x()

    @property
    def mask(self):
        local_prior_scales = self.model.local_prior_scales
        if local_prior_scales is None:
            return None
        return local_prior_scales.cpu().detach().numpy() >= 1.0

    @property
    def sigma(self):
        return 1.0 / np.sqrt(self.model.get_precision())

    def sparsity(self):
        w_activations = self.get_w_activations(self.threshold)
        frac_inactive = 1 - w_activations.mean()
        self.scores["sparsity"].append(frac_inactive)
        print(f"Average fraction of inactive genes: {frac_inactive:.3f}")

        return frac_inactive

    def view_sigma(self):
        sigma = self.sigma
        self.scores["sigma"].append(sigma.mean())
        print(f"Average sigma: {sigma.mean():.3f}")

    def binary_scores(self):
        mask = self.kwargs.get("mask", self.mask)
        at = self.kwargs.get("binary_scores_at", 100)

        if mask is None:
            print("Mask is none...")
            return

        # thresholds = [self.threshold]
        # thresholds = [0.05, 0.1, 0.5]
        n_annotated = self.kwargs.get("n_annotated", self.model.n_factors)
        print(
            f"Binary scores between prior mask and learned activations "
            f"with threshold {self.threshold} of top {at} (abs) weights:"
        )
        for threshold in [self.threshold]:
            accuracy, precision, recall, f1_score = compute_cf_scores_at(
                mask[:n_annotated, :],
                self.w[:n_annotated, :],
                threshold=self.threshold,
                at=at,
            )
            print(
                f"accuracy: {accuracy:.3f}, "
                f"precision: {precision:.3f}, "
                f"recall: {recall:.3f}, "
                f"F1: {f1_score:.3f}"
            )
            self.scores[f"accuracy_{threshold}"].append(accuracy)
            self.scores[f"precision_{threshold}"].append(precision)
            self.scores[f"recall_{threshold}"].append(recall)
            self.scores[f"f1_score_{threshold}"].append(f1_score)
        # precisions[t].append(precision)
        # recalls[t].append(recall)
        # f1s[t].append(f1)
        # ws.append(w)

    def r2(self):
        y_true = self.kwargs["y_true"]
        _, r2_scores_acc = compute_factor_relevance(y_true, self.x, self.w, which="acc")
        print(f"Variance explained (R2): {r2_scores_acc:.3f}")
        self.scores["r2"].append(r2_scores_acc)

    def log(self):
        if self.should_log:
            self.ws.append(self.w)
            self.xs.append(self.x)

    def __call__(self, loss_history):

        for key, cb in self.callback_dict.items():
            cb(loss_history)

        return False
