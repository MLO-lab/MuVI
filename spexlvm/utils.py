"""Collection of helper functions."""
import os
import time

import numpy as np
from scipy.optimize import linprog
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


def generate_filename() -> str:
    """Create unique filename for both the logfile and model related output.

    Returns
    -------
    str
        filename
    """
    return time.strftime("%Y%m%d-%H%M%S")


def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)


def compute_confusion_matrix(true_x, pred_x):
    """Compute a confusion matrix between two binary vectors."""
    return confusion_matrix(true_x, pred_x)


def compute_cf_scores_at(true_mask, learned_w, threshold=0.05, at=None):
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
    sorted_true_mask = sorted_true_mask[:, :at]
    sorted_learned_mask = sorted_learned_mask[:, :at]

    tp = sorted_true_mask & sorted_learned_mask
    tn = ~sorted_true_mask & ~sorted_learned_mask
    fp = ~sorted_true_mask & sorted_learned_mask
    fn = sorted_true_mask & ~sorted_learned_mask

    ntp = tp.sum()
    nfp = fp.sum()
    nfn = fn.sum()
    ntn = tn.sum()

    accuracy = (ntp + ntn) / np.prod(sorted_true_mask.shape)
    precision = ntp / (ntp + nfp)
    recall = ntp / (ntp + nfn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


def compute_cosine_similarity(x, y=None):
    return cosine_similarity(x, y)


def optim_perm(A):
    n, n = A.shape
    res = linprog(
        -A.ravel(),
        A_eq=np.r_[
            np.kron(np.identity(n), np.ones((1, n))),
            np.kron(np.ones((1, n)), np.identity(n)),
        ],
        b_eq=np.ones((2 * n,)),
        bounds=n * n * [(0, None)],
    )
    assert res.success
    shuffle = res.x.reshape(n, n).T
    shuffle[np.abs(shuffle) < 1e-2] = 0.0
    return shuffle
