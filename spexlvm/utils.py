"""Collection of helper functions."""
import os
from typing import List

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import confusion_matrix
from statsmodels.stats import multitest
from tqdm import tqdm

from spexlvm.data import Pathways, load_pathways


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


def compute_factor_relevance(
    y: np.ndarray, x: np.ndarray, w: np.ndarray, which: str = "both"
):
    """Compute factor relevance based on variance explained ($R^2$).

    Parameters
    ----------
    y : np.ndarray
        True data matrix of samples times genes
    x : np.ndarray
        Latent factor scores matrix of samples times features
    w : np.ndarray
        Factor loadings matrix of factors times features
    which : str, optional
        Whether to compute variance explained per factor, in total, or both,
        by default "both"

    Returns
    -------
    tuple
        (scores per factor, total score)
    """

    presence_mask = ~np.isnan(y)

    n_factors = x.shape[1]

    r2_score_acc = 0.0
    r2_scores = np.zeros(n_factors)

    y_true = np.nan_to_num(y, nan=0.0)
    ss_tot = np.square(y_true).sum()

    if which == "acc" or which == "both":
        y_hat = np.where(presence_mask, x @ w, np.zeros_like(y_true))
        ss_res = np.square(y_true - y_hat).sum()
        r2_score_acc = 1 - (ss_res / ss_tot)

    if which == "fac" or which == "both":
        for k in tqdm(range(n_factors)):
            y_hat_fac_k = np.where(
                presence_mask, np.outer(x[:, k], w[k, :]), np.zeros_like(y_true)
            )
            ss_res = np.square(y_true - y_hat_fac_k).sum()
            r2_scores[k] = 1 - (ss_res / ss_tot)
    return r2_scores, r2_score_acc

    # """Helper function to load pathways from msigdb.

    # Parameters
    # ----------
    # genes : List[str]
    #     List of genes with which to subset the pathways
    # pathways : List[str]
    #     List of the names of pathway collections
    # pathway_min_gene_fraction : float, optional
    #     Min fraction genes available in a pathway, by default 0.1
    # pathway_min_gene_count : int, optional
    #     Min number genes available in a pathway, by default 15
    # max_gene_count : int, optional
    #     asd, by default 350
    # redundant_pathways : list, optional
    #     asd, by default []

    # Returns
    # -------
    # [type]
    #     [description]
    # """


def get_pathways(
    genes: List[str],
    collections: List[str],
    min_gene_fraction: List[float] = None,
    min_gene_count: List[int] = None,
    max_gene_count: List[int] = None,
    redundant_pathways: List[str] = None,
):
    """Helper function to load pathways from msigdb.

    Parameters
    ----------
    genes : List[str]
        List of genes with which to filter the pathways
    collections : List[str]
        List of the names of pathway collections
    min_gene_fraction : List[float], optional
        Min fraction genes available in a pathway collection, by default 0.1
    min_gene_count : List[int], optional
        Min number genes available in a pathway collection, by default 15
    max_gene_count : List[int], optional
        Min number genes available in a pathway collection, by default 350
    redundant_pathways : List[str], optional
        List of pathways to skip

    Returns
    -------
    Pathways
        A collection of pathways
    """

    n_collections = len(collections)
    if min_gene_fraction is None:
        min_gene_fraction = [0.1 for _ in range(n_collections)]
    if min_gene_count is None:
        min_gene_count = [15 for _ in range(n_collections)]
    if max_gene_count is None:
        max_gene_count = [350 for _ in range(n_collections)]
    if redundant_pathways is None:
        redundant_pathways = []

    # load pathways
    pathways = tuple()
    for i, c in enumerate(collections):
        print(
            f"Loading collection {c} with at least "
            f"{(min_gene_fraction[i] * 100):2.1f}% of genes "
            f"available and at least {min_gene_count[i]} genes"
        )
        keep = []
        tmp_pathways = (
            load_pathways(keep=[c])
            .subset(
                genes,
                fraction_available=min_gene_fraction[i],
                min_gene_count=min_gene_count[i],
                max_gene_count=max_gene_count[i],
                keep=keep,
            )
            .gene_sets
        )
        pathways += tmp_pathways

        print(
            f"Adding {len(tmp_pathways)} pathways from "
            f"{c} collection with median size of "
            f"{np.median([len(gs.genes) for gs in tmp_pathways])} genes"
        )

    measured_pathways = Pathways(gene_sets=pathways)
    for k, v in measured_pathways.find_redundant().items():
        redundant_pathways += v[1:]

    if len(redundant_pathways) > 0:
        print(f"Removing following redundant pathways {', '.join(redundant_pathways)}")
    measured_pathways = measured_pathways.remove(redundant_pathways)
    assert len(measured_pathways.find_redundant()) == 0
    # print(measured_pathways.info(verbose=1))
    print(
        f"Loaded in total {len(measured_pathways)} pathways "
        "with a median size of "
        f"{np.median([len(gs.genes) for gs in measured_pathways.gene_sets])}"
    )
    return measured_pathways


# assume format: COLLECTION_SOME_LONG_GENE_SET_NAME
# return format: Some Long...Name (C)
def prettify(name, str_len_threshold=32):
    gsn_parts = [part.capitalize() for part in name.split("_")]
    if gsn_parts[0] not in ["Sparse", "Dense"]:
        gsn_parts[0] = "(" + gsn_parts[0][0] + ")"
        gsn_parts = gsn_parts[1:] + gsn_parts[:1]

    new_name = " ".join(gsn_parts)
    if len(new_name) > str_len_threshold:
        half_str_len_threshold = (str_len_threshold - 4) // 2
        new_name = (
            new_name[:half_str_len_threshold]
            + "..."
            + new_name[-half_str_len_threshold - 3 :]
        )

    return new_name


def test_refinement(
    adata,
    factors="all",
    sign="all",
    corr_adjust=False,
    p_adj_method="hs",
):

    Y = adata.to_df()
    X = adata.obsm["X"]
    W = adata.varm["W"]
    pathways = adata.varm["pathway_mask"].astype(bool)

    # subset available features only
    feature_intersection = pathways.index.intersection(W.index)
    Y = Y.loc[:, feature_intersection]
    W = W.loc[feature_intersection, :]
    pathways = pathways.loc[feature_intersection, :]
    pathways = pathways.loc[:, pathways.sum(axis=0) > 0]

    if "pos" in sign.lower():
        W[W < 0] = 0.0
    if "neg" in sign.lower():
        W[W > 0] = 0.0

    W = W.abs()

    if factors == "all":
        factors = X.columns

    t_stat_dict = {}
    prob_dict = {}

    for pathway in pathways.columns:
        pathway_genes = pathways.loc[:, pathway]
        features_in = W.loc[pathway_genes, factors]
        features_out = W.loc[~pathway_genes, factors]
        n_in = len(features_in)
        n_out = len(features_out)

        df = n_in + n_out - 2.0

        mean_diff = features_in.mean() - features_out.mean()
        # why divide here by df and not denom later?
        svar = ((n_in - 1) * features_in.var() + (n_out - 1) * features_out.var()) / df

        vif = 1.0
        if corr_adjust:
            corr_df = Y.loc[:, pathway_genes].corr()
            mean_corr = (corr_df.to_numpy().sum() - n_in) / (n_in * (n_in - 1))
            vif = 1 + (n_in - 1) * mean_corr
            df = Y.shape[0] - 2
        denom = np.sqrt(svar * (vif / n_in + 1.0 / n_out))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.divide(mean_diff, denom)
        prob = t_stat.apply(lambda t: scipy.stats.t.sf(np.abs(t), df) * 2)

        t_stat_dict[pathway] = t_stat
        prob_dict[pathway] = prob

    t_stat_df = pd.DataFrame(t_stat_dict, index=factors)
    prob_df = pd.DataFrame(prob_dict, index=factors)
    prob_adj_df = prob_df.apply(
        lambda p: multitest.multipletests(p, method=p_adj_method)[1],
        axis=1,
        result_type="broadcast",
    )

    return t_stat_df, prob_df, prob_adj_df
