"""Collection of MuVI callbacks."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_squared_error

from muvi.utils import compute_cf_scores_at


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


class LogCallback:
    _CALLBACK_KEYS = [
        "sparsity",
        "binary_scores",
        "rmse",
        "log",
    ]

    def __init__(
        self,
        model,
        n_iterations,
        n_checkpoints=10,
        active_callbacks=None,
        view_wise=True,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.model = model
        self.n_iterations = n_iterations
        self.n_checkpoints = n_checkpoints
        self.view_wise = view_wise

        if active_callbacks is None:
            active_callbacks = self._CALLBACK_KEYS
        self.callback_config = {k: self._default_config[k] for k in active_callbacks}
        self.callback_dict = {
            k: CheckpointCallback(
                self.model, self.n_iterations, v["n_checkpoints"], v["callback"]
            )
            for k, v in self.callback_config.items()
        }
        self.threshold = kwargs.get("threshold", 0.1)
        self.should_log = kwargs.get("log", False)

        self.ws = []
        self.zs = []
        self.fs = []
        self.scores = defaultdict(list)

    @property
    def _default_config(self):
        return {
            "sparsity": {
                "n_checkpoints": self.n_checkpoints,
                "callback": self.sparsity,
            },
            "binary_scores": {
                "n_checkpoints": self.n_checkpoints,
                "callback": self.binary_scores,
            },
            "rmse": {"n_checkpoints": self.n_checkpoints, "callback": self.rmse},
            "log": {
                "n_checkpoints": self.kwargs.get("log_frequency", 100),
                "callback": self.log,
            },
        }

    @property
    def betas(self):
        return self.model.get_beta(as_list=self.view_wise)

    @property
    def w(self):
        return self.model.get_w(as_list=self.view_wise)

    def get_w_activations(self, threshold):
        if self.view_wise:
            return [np.abs(w) > threshold for w in self.w]
        return np.abs(self.w) > threshold

    @property
    def z(self):
        return self.model.get_z()

    @property
    def factor_scale(self):
        return self.model.get_factor_scale()

    @property
    def mask(self):
        prior_scales = self.model.prior_scales
        if prior_scales is None:
            return None
        return [ps >= 1.0 for ps in prior_scales]

    @property
    def sigma(self):
        return self.model.get_sigma(as_list=self.view_wise)

    def sparsity(self):
        w_activations = self.get_w_activations(self.threshold)
        frac_inactive = []
        if self.view_wise:
            frac_inactive = [1 - wa.mean() for wa in w_activations]
        else:
            frac_inactive = [1 - w_activations.mean()]

        for m, fi in enumerate(frac_inactive):
            self.scores[f"sparsity_{m}"].append(fi)

        print(
            "Average fraction of inactive loadings for each view: ",
            " ".join(f"{fi:.3f}" for fi in frac_inactive),
        )
        return frac_inactive

    def binary_scores(self):
        mask = self.kwargs.get("mask", self.mask)
        at = self.kwargs.get("binary_scores_at", 100)

        if mask is None:
            print("Mask is none...")
            return

        thresholds = [self.threshold]
        n_annotated = self.kwargs.get("n_annotated", self.model.n_factors)
        for threshold in thresholds:
            print(
                "Binary scores between prior mask and learned activations "
                f"with a threshold of {threshold} for top {at} (abs) weights:"
                f"\nview #: (acc, prec, rec, f1)"
            )
            if self.view_wise:
                informed_views = self.kwargs.get(
                    "informed_views", range(self.model.n_views)
                )
                for m in informed_views:
                    acc, prec, rec, f1 = compute_cf_scores_at(
                        mask[m][:n_annotated, :],
                        self.w[m][:n_annotated, :],
                        threshold=threshold,
                        at=at,
                    )
                    print(f"view {m}: ({acc:.3f}, {prec:.3f}, {rec:.3f}, {f1:.3f})")
                    self.scores[f"accuracy_{m}"].append(acc)
                    self.scores[f"precision_{m}"].append(prec)
                    self.scores[f"recall_{m}"].append(rec)
                    self.scores[f"f1_{m}"].append(f1)

    def rmse(self):
        y_true = self.kwargs["y_true"]
        rmses = []
        for m in range(self.model.n_views):
            y_pred = self.z @ self.w[m]
            if self.model.n_covariates > 0:
                y_pred += self.model.covariates @ self.betas[m]
            rmse = mean_squared_error(y_true[m], y_pred, squared=False)
            rmses.append(rmse)
            self.scores[f"rmse_{m}"].append(rmse)

        print("Overall rmse for each view: ", " ".join(f"{rmse:.3f}" for rmse in rmses))

    def log(self):
        if self.should_log:
            self.ws.append(self.w)
            self.zs.append(self.z)
            self.fs.append(self.factor_scale)

    def __call__(self, loss_history):

        for key, cb in self.callback_dict.items():
            cb(loss_history)

        return False


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
                print(
                    f"Relative improvement of "
                    f"{relative_improvement:0.4g} < {self.tolerance:0.4g} "
                    f"for {self.patience} step(s) in a row, stopping early."
                )

        return stop_early
