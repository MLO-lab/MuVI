"""Collection of helper functions."""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)


def compute_confusion_matrix(true_x, pred_x):
    """Compute a confusion matrix between two binary vectors."""
    return confusion_matrix(true_x, pred_x)


def compute_factor_relevance(y, X, W, factor_indices=None):
    """Compute factor relevance as the coefficient of determination (R^2)."""
    if factor_indices is None:
        factor_indices = list(range(W.shape[0]))
    ss_tot = np.sum(np.power(y, 2))
    r2_scores = []
    for k in tqdm(factor_indices):
        y_hat = np.outer(X[:, k], W[k, :])
        ss_res = np.sum(np.power(y - y_hat, 2))
        r2_scores.append((k, 1 - (ss_res / ss_tot)))

    return pd.DataFrame(r2_scores, columns=["Index", "R2"])
