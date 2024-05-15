"""Collection of training callbacks."""
from collections import defaultdict
from typing import Callable
from typing import Optional

import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from muvi.tools.utils import rmse
from muvi.tools.utils import variance_explained


class CheckpointCallback:
    def __init__(
        self,
        model,
        n_epochs: int,
        n_checkpoints: int = 10,
        callback: Optional[Callable] = None,
    ):
        """Checkpoint callback.

        Parameters
        ----------
        model : MuVI
            A MuVI object
        n_epochs : int
            Number of training epochs
        n_checkpoints : int, optional
            Number of times to execute during training,
            by default 10
        callback : Callable, optional
            A function that accepts the loss history,
            by default None
        """
        self.model = model
        self.n_epochs = n_epochs
        self.n_checkpoints = n_checkpoints
        self.window_size = n_epochs // n_checkpoints
        self.callback = callback

    def __call__(self, loss_history):
        epoch_idx = len(loss_history)

        if epoch_idx % self.window_size == 0 or epoch_idx == self.n_epochs:
            self.callback()

        return False


class LogCallback:
    from typing import ClassVar

    _CALLBACK_KEYS: ClassVar[list] = [
        "sparsity",
        "binary_scores",
        "rmse",
        "r2",
        "log",
    ]

    def __init__(
        self,
        model,
        n_epochs: int,
        n_checkpoints: int = 10,
        active_callbacks: Optional[list[str]] = None,
        view_wise: bool = True,
        **kwargs,
    ) -> None:
        """Log callback.

        Parameters
        ----------
        model : MuVI
            A MuVI object
        n_epochs : int
            Number of training epochs
        n_checkpoints : int, optional
            Number of times to execute during training,
            by default 10
        active_callbacks : list[str], optional
            List of callback names from LogCallback._CALLBACK_KEYS,
            by default None (all)
        view_wise : bool, optional
            Whether to report scores for each view,
            by default True
        """
        self.kwargs = kwargs
        self.model = model
        self.n_epochs = n_epochs
        self.n_checkpoints = n_checkpoints
        self.view_wise = view_wise

        if active_callbacks is None:
            active_callbacks = self._CALLBACK_KEYS
        self.callback_config = {k: self._default_config[k] for k in active_callbacks}
        self.callback_dict = {
            k: CheckpointCallback(
                self.model,
                self.n_epochs,
                n_checkpoints=v["n_checkpoints"],
                callback=v["callback"],
            )
            for k, v in self.callback_config.items()
        }
        self.threshold = kwargs.get("threshold", 0.1)
        self.should_log = kwargs.get("log", False)

        self.ws = []
        self.zs = []
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
                "n_checkpoints": self.n_checkpoints,
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
            str_result = (
                "Binary scores between prior mask and learned activations "
                + f"with a threshold of {threshold} for top {at} (abs) weights:"
                + "\n`view_name`: (prec, rec, f1)\n"
            )

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
                str_result += f"{vn}: ({prec:.3f}, {rec:.3f}, {f1:.3f})\n"
            str_result = str_result[:-1]
            print(str_result)

    def _rmse(self):
        cov_idx = "all"
        if self.model.n_covariates == 0:
            cov_idx = None
        scores = rmse(
            self.model, cov_idx=cov_idx, factor_wise=False, cov_wise=False, cache=False
        )[0]
        str_result = "RMSE for each view:\n"
        for vn, s in scores.items():
            str_result += f"{vn}: {s:.3f}, "
            self.scores[f"rmse_{vn}"].append(s)
        str_result = str_result[:-2]
        print(str_result)
        return scores

    def _variance_explained(self):
        cov_idx = "all"
        if self.model.n_covariates == 0:
            cov_idx = None
        scores = variance_explained(
            self.model, cov_idx=cov_idx, factor_wise=False, cov_wise=False, cache=False
        )[0]
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
        for _key, cb in self.callback_dict.items():
            cb(loss_history)

        return False


def compute_binary_scores_at(
    true_mask: np.ndarray,
    learned_w: np.ndarray,
    threshold: float = 0.05,
    at: Optional[int] = None,
):
    """Compute binary scores (precision, recall, F1).

    Parameters
    ----------
    true_mask : np.ndarray
        True underlying mask or the prior mask
    learned_w : np.ndarray
        Factor loadings
    threshold : float, optional
        Threshold to determine active and inactive loadings,
        by default 0.05
    at : int, optional
        Number of the factor loadings to consider,
        sorted by abs value,
        by default None (all)

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support : None (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    """
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
