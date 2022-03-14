"""Collection of MuVI callbacks."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from muvi.tools.utils import rmse, variance_explained


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
        "r2",
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
            "rmse": {"n_checkpoints": self.n_checkpoints, "callback": self._rmse},
            "r2": {
                "n_checkpoints": self.n_checkpoints,
                "callback": self._variance_explained,
            },
            "log": {
                "n_checkpoints": self.kwargs.get("log_frequency", 100),
                "callback": self.log,
            },
        }

    @property
    def factor_loadings(self):
        return self.model.get_factor_loadings()

    def get_activations(self, threshold):
        return {vn: np.abs(w) > threshold for vn, w in self.factor_loadings.items()}

    @property
    def factor_scores(self):
        return self.model.get_factor_scores()

    @property
    def masks(self):
        return self.kwargs.get("masks", self.model.get_prior_masks())

    def sparsity(self):
        activations = self.get_activations(self.threshold)
        frac_inactive = {vn: 1 - a.mean() for vn, a in activations.items()}

        str_result = "Average fraction of inactive loadings:\n"
        for vn, fi in frac_inactive.items():
            str_result += f"{vn}: {fi:.3f}, "
            self.scores[f"sparsity_{vn}"].append(fi)
        str_result = str_result[:-2]
        print(str_result)
        return frac_inactive

    def binary_scores(self):
        masks = self.masks
        at = self.kwargs.get("binary_scores_at", 100)

        if masks is None:
            print("Mask is none...")
            return

        thresholds = [self.threshold]
        n_annotated = self.kwargs.get("n_annotated", self.model.n_factors)
        for threshold in thresholds:
            str_result = "Binary scores between prior mask and learned activations "
            f"with a threshold of {threshold} for top {at} (abs) weights:"
            "\n`view_name`: (prec, rec, f1)"

            informed_views = self.kwargs.get("informed_views", self.model.view_names)
            for vn in informed_views:
                if isinstance(vn, int):
                    vn = self.model.view_names[vn]
                prec, rec, f1, _ = compute_binary_scores_at(
                    masks[vn][:n_annotated, :],
                    self.factor_loadings[vn][:n_annotated, :],
                    threshold=threshold,
                    at=at,
                )
                self.scores[f"precision_{vn}"].append(prec)
                self.scores[f"recall_{vn}"].append(rec)
                self.scores[f"f1_{vn}"].append(f1)
                str_result += f"view {vn}: ({prec:.3f}, {rec:.3f}, {f1:.3f})"

    def _rmse(self):
        scores = rmse(self.model)[0]
        str_result = "RMSE for each view:\n"
        for vn, s in scores.items():
            str_result += f"{vn}: {s:.3f}, "
            self.scores[f"rmse_{vn}"].append(s)
        str_result = str_result[:-2]
        print(str_result)
        return scores

    def _variance_explained(self):
        scores = variance_explained(self.model)[0]
        str_result = "Variance explained for each view:\n"
        for vn, s in scores.items():
            str_result += f"{vn}: {s:.3f}, "
            self.scores[f"r2_{vn}"].append(s)
        str_result = str_result[:-2]
        print(str_result)
        return scores

    def log(self):
        if self.should_log:
            self.ws.append(self.factor_loadings)
            self.zs.append(self.factor_scores)

    def __call__(self, loss_history):

        for key, cb in self.callback_dict.items():
            cb(loss_history)

        return False


def compute_binary_scores_at(true_mask, learned_w, threshold=0.05, at=None):
    if at is None:
        at = true_mask.shape[1]

    learned_w_abs = np.abs(learned_w)
    # descending order
    argsort_indices = np.argsort(-learned_w_abs, axis=1)

    sorted_true_mask = np.array(
        list(map(lambda x, y: y[x], argsort_indices, true_mask))
    )
    sorted_learned_mask = (
        np.array(list(map(lambda x, y: y[x], argsort_indices, learned_w_abs)))
        > threshold
    )
    sorted_true_mask = sorted_true_mask[:, :at].flatten()
    sorted_learned_mask = sorted_learned_mask[:, :at].flatten()

    return precision_recall_fscore_support(
        sorted_true_mask, sorted_learned_mask, average="binary"
    )
