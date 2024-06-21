import io
import logging

from pathlib import Path
from typing import Optional
from typing import Union

import anndata as ad
import dill as pickle
import mudata as md
import numpy as np
import pandas as pd
import pyro
import scanpy as sc
import scipy
import torch

from kneed import KneeLocator
from scipy.optimize import linprog
from sklearn.metrics import root_mean_squared_error
from statsmodels.stats import multitest
from tqdm import tqdm

from muvi.core.index import _normalize_index
from muvi.core.models import MuVI
from muvi.tools import feature_sets as fs
from muvi.tools.cache import Cache


logger = logging.getLogger(__name__)

Index = Union[int, str, list[int], list[str], np.ndarray, pd.Index]


def setup_cache(model, overwrite: bool = False):
    """Setup model cache."""
    # check if model has been trained?
    if overwrite:
        model._cache = None

    if not model._cache:
        if not overwrite:
            logger.warning("Cache has not yet been setup, initialising model cache.")
        model._cache = Cache(model)

    return model._cache


def get_metadata(model, name):
    model_cache = setup_cache(model)
    if name not in model_cache.factor_adata.obs.columns:
        raise ValueError(
            f"`{name}` not found in the metadata, "
            "call `muvi.tl.add_metadata` to add new metadata."
        )

    return model_cache.factor_adata.obs[name].copy()


def add_metadata(model, name, metadata, overwrite=False):
    model_cache = setup_cache(model)
    if name in model_cache.factor_adata.obs.columns and not overwrite:
        raise ValueError(
            f"`{name}` already found in the metadata, "
            "set `overwrite=True` to replace the existing values."
        )

    model_cache.factor_adata.obs[name] = metadata
    return model_cache.factor_adata.obs[name].copy()


def _filter_factors(model, factor_idx: Index):
    """Filter factors for the current analysis."""
    factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)
    if len(factor_idx) == 0:
        raise ValueError("`factor_idx` is empty.")
    model_cache = setup_cache(model)
    return model_cache.filter_factors(factor_idx)


def filter_factors(model, r2_thresh: Union[int, float] = 0.95):
    """Filter factors based on variance explained.

    Parameters
    ----------
    model : MuVI
        A MuVI model.
    r2_thresh : Union[int, float], optional
        Threshold for the (rel) cumulative sum of variance explained,
        or the number of top factors, sorted by variance explained,
        by default 0.95

    Returns
    -------
    bool
        True if succesfully filtered factors.
    """
    model_cache = setup_cache(model)

    r2_cols = [f"{Cache.METRIC_R2}_{vn}" for vn in model.view_names]
    r2_df = model_cache.factor_metadata[r2_cols]

    if r2_df.isna().all(None):
        logger.warning(
            "Unable to filter factors based on variance explained.\n"
            "Run `muvi.tl.variance_explained` to compute "
            "the variance explained by each factor first."
        )
        return False

    r2_sorted = r2_df.sum(1).sort_values(ascending=False)

    if int(r2_thresh) == 1:
        logger.warning(
            "Unable to filter factors based on variance explained.\n"
            f"`r2_thresh` of `{r2_thresh}` is ambiguous, `r2_thresh` must be "
            "less than 1.0 or an integer greater than 1."
        )
        return False

    factor_subset = r2_sorted.index
    if r2_thresh < 1.0:
        r2_thresh = (r2_sorted.cumsum() / r2_sorted.sum() < r2_thresh).sum() + 1

    factor_subset = r2_sorted.iloc[: int(r2_thresh)].index

    factor_subset = factor_subset.tolist()
    logger.info(f"Filtering down to {len(factor_subset)} factors.")
    return _filter_factors(model, factor_subset)


def _recon_error(
    model,
    view_idx,
    sample_idx,
    feature_idx,
    factor_idx,
    cov_idx,
    factor_wise,
    cov_wise,
    subsample,
    metric_label,
    metric_fn,
    cache,
    sort,
):
    if view_idx is None:
        raise ValueError("`view_idx` cannot be None.")

    if model.n_factors == 0:
        factor_idx = None
    if model.n_covariates == 0:
        cov_idx = None

    if factor_idx is None and cov_idx is None:
        raise ValueError("`factor_idx` and `cov_idx` cannot be both None.")

    try:
        factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)
    except IndexError:
        if factor_idx is not None and model.n_factors > 0:
            logger.warning(f"Invalid factor index: `{factor_idx}`.")
        factor_idx = None

    try:
        cov_idx = _normalize_index(cov_idx, model.covariate_names, as_idx=False)
    except IndexError:
        if cov_idx is not None and model.n_covariates > 0:
            logger.warning(f"Invalid covariance index: `{cov_idx}`.")
        cov_idx = None

    if factor_idx is None and cov_idx is None:
        raise ValueError(
            "Both `factor_idx` and `cov_idx` are invalid, at least one is required."
        )

    factor_wise &= factor_idx is not None
    cov_wise &= cov_idx is not None

    if sample_idx is None:
        sample_idx = "all"

    sample_idx = _normalize_index(sample_idx, model.sample_names)

    if subsample is not None and subsample > 0 and subsample < len(sample_idx):
        logger.info(
            f"Estimating `{metric_label}` with a random sample of {subsample} samples."
        )
        sample_idx = np.random.choice(sample_idx, subsample, replace=False)

    ys = model.get_observations(
        view_idx, sample_idx=sample_idx, feature_idx=feature_idx
    )
    view_names = list(ys.keys())

    view_scores_key = f"view_{metric_label}"
    cache_columns = [f"{metric_label}_{vn}" for vn in model.view_names]

    model_cache = setup_cache(model)

    n_samples = next(iter(ys.values())).shape[0]
    if subsample is None and n_samples > 10000 and (factor_wise or cov_wise):
        logger.warning(
            f"Computing `{metric_label}` with `{n_samples}` samples, "
            "this may take some time. "
            f"Consider estimating `{metric_label}` by setting `subsample` "
            "to a smaller number."
        )

    n_factors = 0
    if factor_idx is not None:
        z = model.get_factor_scores(sample_idx=sample_idx, factor_idx=factor_idx)
        ws = model.get_factor_loadings(view_idx, factor_idx, feature_idx)
        n_factors = len(factor_idx)
    n_covariates = 0
    if cov_idx is not None:
        x = model.get_covariates(sample_idx=sample_idx, cov_idx=cov_idx)
        betas = model.get_covariate_coefficients(view_idx, cov_idx, feature_idx)
        n_covariates = len(cov_idx)

    view_scores = {}
    factor_scores = {}
    cov_scores = {}

    for m, vn in enumerate(view_names):
        score_key = cache_columns[m]
        y_true_view = ys[vn]
        y_pred_view = np.zeros_like(y_true_view)
        if n_factors > 0:
            y_pred_view += model._mode(z @ ws[vn], model.likelihoods[vn])
        if n_covariates > 0:
            y_pred_view += x @ betas[vn]
        view_scores[vn] = metric_fn(y_true_view, y_pred_view)

        factor_scores[score_key] = np.zeros(n_factors)
        cov_scores[score_key] = np.zeros(n_covariates)
        if factor_wise:
            for k in range(n_factors):
                y_pred_fac_k = model._mode(
                    np.outer(z[:, k], ws[vn][k, :]), model.likelihoods[vn]
                )
                factor_scores[score_key][k] = metric_fn(y_true_view, y_pred_fac_k)
        if cov_wise:
            for k in range(n_covariates):
                y_pred_cov_k = model._mode(
                    np.outer(x[:, k], betas[vn][k, :]), model.likelihoods[vn]
                )
                cov_scores[score_key][k] = metric_fn(y_true_view, y_pred_cov_k)

    factor_scores = pd.DataFrame(
        factor_scores, index=[] if factor_idx is None else factor_idx
    )
    cov_scores = pd.DataFrame(cov_scores, index=[] if cov_idx is None else cov_idx)
    if cache:
        model_cache.update_uns(view_scores_key, view_scores)
        model_cache.update_factor_metadata(factor_scores)
        model_cache.update_cov_metadata(cov_scores)
        if sort and sort in ["ascending", "descending"]:
            order = (
                factor_scores.sum(1).sort_values(ascending=sort == "ascending").index
            )
            order = _normalize_index(order, model.factor_names, as_idx=True)
            model.factor_order = order
    return view_scores, factor_scores, cov_scores


def rmse(
    model,
    view_idx: Index = "all",
    sample_idx: Index = "all",
    feature_idx: Index = "all",
    factor_idx: Index = "all",
    cov_idx: Index = "all",
    factor_wise: bool = True,
    cov_wise: bool = True,
    subsample: int = 0,
    cache: bool = True,
    sort: bool = True,
):
    """Compute RMSE.

    Parameters
    ----------
    model : MuVI
        A MuVI model
    view_idx : Index, optional
        View index, by default "all"
    sample_idx : Index, optional
        Sample index, by default "all"
    feature_idx : Index, optional
        Feature index, by default "all"
    factor_idx : Index, optional
        Factor index, by default "all"
    cov_idx : Index, optional
        Covariate index, by default "all"
    factor_wise : bool, optional
        Whether to compute factor-wise RMSE, by default True
    cov_wise : bool, optional
        Whether to compute covariate-wise RMSE, by default True
    subsample : int, optional
        Number of samples to estimate RMSE, by default 0 (all samples)
    cache : bool, optional
        Whether to store results in the model cache, by default True
    sort : bool, optional
        Whether to sort factors by RMSE, by default True
    """

    def _rmse(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)

    if sort:
        sort = "ascending"

    return _recon_error(
        model,
        view_idx,
        sample_idx,
        feature_idx,
        factor_idx,
        cov_idx,
        factor_wise,
        cov_wise,
        subsample,
        metric_label=Cache.METRIC_RMSE,
        metric_fn=_rmse,
        cache=cache,
        sort=sort,
    )


def variance_explained(
    model,
    view_idx: Index = "all",
    sample_idx: Index = "all",
    feature_idx: Index = "all",
    factor_idx: Index = "all",
    cov_idx: Index = "all",
    factor_wise: bool = True,
    cov_wise: bool = True,
    subsample: int = 0,
    cache: bool = True,
    sort: bool = True,
):
    """Compute R2.

    Parameters
    ----------
    model : MuVI
        A MuVI model
    view_idx : Index, optional
        View index, by default "all"
    sample_idx : Index, optional
        Sample index, by default "all"
    feature_idx : Index, optional
        Feature index, by default "all"
    factor_idx : Index, optional
        Factor index, by default "all"
    cov_idx : Index, optional
        Covariate index, by default "all"
    factor_wise : bool, optional
        Whether to compute factor-wise R2, by default True
    cov_wise : bool, optional
        Whether to compute covariate-wise R2, by default True
    subsample : int, optional
        Number of samples to estimate R2, by default 0 (all samples)
    cache : bool, optional
        Whether to store results in the model cache, by default True
    sort : bool, optional
        Whether to sort factors by R2, by default True
    """

    def _r2(y_true, y_pred):
        ss_res = np.nansum(np.square(y_true - y_pred))
        ss_tot = np.nansum(np.square(y_true))
        return 1.0 - (ss_res / ss_tot)

    if sort:
        sort = "descending"

    return _recon_error(
        model,
        view_idx,
        sample_idx,
        feature_idx,
        factor_idx,
        cov_idx,
        factor_wise,
        cov_wise,
        subsample,
        metric_label=Cache.METRIC_R2,
        metric_fn=_r2,
        cache=cache,
        sort=sort,
    )


def variance_explained_grouped(model, groupby, factor_idx: Index = "all", **kwargs):
    model_cache = setup_cache(model)
    if groupby not in model_cache.factor_adata.obs.columns:
        raise ValueError(
            f"`{groupby}` not found in the metadata, "
            " add a new column onto `model._cache.factor_adata.obs`."
        )

    # TODO: at some point extend with covariates
    metadata = (
        model_cache.factor_adata.obs[groupby]
        .astype("category")
        .cat.remove_unused_categories()
    )
    group_wise_r2 = (
        metadata.groupby(metadata)
        .apply(
            lambda group_df: variance_explained(
                model,
                sample_idx=group_df.index,
                factor_idx=factor_idx,
                cache=False,
                sort=False,
                **kwargs,
            )[1].copy()
        )
        .reset_index()
    )

    model._cache.uns[Cache.UNS_GROUPED_R2] = group_wise_r2.rename(
        columns={"level_1": "Factor"}
    )
    return model._cache.uns[Cache.UNS_GROUPED_R2]


def _test_single_view(
    model,
    view_idx: Union[str, int] = 0,
    factor_idx: Index = "all",
    feature_sets: pd.DataFrame = None,
    sign: str = "all",
    corr_adjust: bool = True,
    p_adj_method: str = "fdr_bh",
    min_size: int = 10,
    cache: bool = True,
):
    """Perform significance test of factor loadings against feature sets.

    Parameters
    ----------
    model : MuVI
        A MuVI model
    view_idx : Union[str, int]
        Single view index
    factor_idx : Index, optional
        Factor index, by default "all"
    feature_sets : pd.DataFrame, optional
        Boolean dataframe with feature sets in each row, by default None
    sign : str, optional
        Two sided ("all") or one-sided ("neg" or "pos"), by default "all"
    corr_adjust : bool, optional
        Whether to adjust for multiple testing, by default True
    p_adj_method : str, optional
        Adjustment method for multiple testing, by default "fdr_bh"
    min_size : int, optional
        Lower size limit for feature sets to be considered, by default 10
    cache : bool, optional
        Whether to store results in the model cache, by default True

    Returns
    -------
    dict
        Dictionary of test results with "t", "p" and "p_adj" keys
        and pd.DataFrame values with factor_idx as index,
        and index of feature_sets as columns
    """
    use_prior_mask = feature_sets is None
    adjust_p = p_adj_method is not None

    if not isinstance(view_idx, (str, int)) and view_idx != "all":
        raise IndexError(
            f"Invalid `view_idx`, `{view_idx}` must be a string or an integer."
        )
    if isinstance(view_idx, int):
        view_idx = model.view_names[view_idx]
    if view_idx not in model.view_names:
        raise IndexError(f"`{view_idx}` not found in the view names.")

    if use_prior_mask and not model._informed:
        raise ValueError(
            "`feature_sets` is None, no feature sets provided for uninformed model."
        )

    model_cache = setup_cache(model)

    sign = sign.lower().strip()
    allowed_signs = [Cache.TEST_ALL, Cache.TEST_POS, Cache.TEST_NEG]
    if sign not in allowed_signs:
        raise ValueError(f"sign `{sign}` must be one of `{', '.join(allowed_signs)}`.")

    if use_prior_mask:
        logger.warning(
            f"No feature sets provided for `{view_idx}`, "
            "extracting feature sets from the prior mask."
        )
        feature_sets = model.get_prior_masks(
            view_idx, factor_idx=factor_idx, as_df=True
        )[view_idx]
        if not feature_sets.any(axis=None):
            raise ValueError(
                f"Empty `feature_sets`, view `{view_idx}` "
                "has not been informed prior to training."
            )

    feature_sets = feature_sets.astype(bool)
    if not feature_sets.any(axis=None):
        raise ValueError("Empty `feature_sets`.")
    feature_sets = feature_sets.loc[feature_sets.sum(axis=1) >= min_size, :]

    if not feature_sets.any(axis=None):
        raise ValueError(
            "Empty `feature_sets` after filtering feature sets "
            f"of fewer than {min_size} features."
        )

    feature_sets = feature_sets.loc[~(feature_sets.all(axis=1)), feature_sets.any()]
    if not feature_sets.any(axis=None):
        raise ValueError(
            "Empty `feature_sets` after filtering feature sets "
            f"of fewer than {min_size} features."
        )

    # subset available features only
    feature_intersection = feature_sets.columns.intersection(
        model.feature_names[view_idx]
    )
    feature_sets = feature_sets.loc[:, feature_intersection]

    if not feature_sets.any(axis=None):
        raise ValueError(
            "Empty `feature_sets` after feature intersection with the observations."
        )

    y = model.get_observations(view_idx, feature_idx=feature_intersection, as_df=True)[
        view_idx
    ]
    factor_loadings = model.get_factor_loadings(
        view_idx, factor_idx=factor_idx, feature_idx=feature_intersection, as_df=True
    )[view_idx]
    factor_loadings /= np.max(np.abs(factor_loadings.to_numpy()))

    if Cache.TEST_POS in sign:
        factor_loadings[factor_loadings < 0] = 0.0
    if Cache.TEST_NEG in sign:
        factor_loadings[factor_loadings > 0] = 0.0
    factor_loadings = factor_loadings.abs()

    factor_names = factor_loadings.index

    t_stat_dict = {}
    prob_dict = {}

    for feature_set in tqdm(feature_sets.index.tolist()):
        fs_features = feature_sets.loc[feature_set, :]

        features_in = factor_loadings.loc[:, fs_features]
        features_out = factor_loadings.loc[:, ~fs_features]

        n_in = features_in.shape[1]
        n_out = features_out.shape[1]

        df = n_in + n_out - 2.0
        mean_diff = features_in.mean(axis=1) - features_out.mean(axis=1)
        # why divide here by df and not denom later?
        svar = (
            (n_in - 1) * features_in.var(axis=1)
            + (n_out - 1) * features_out.var(axis=1)
        ) / df

        vif = 1.0
        if corr_adjust:
            corr_df = y.loc[:, fs_features].corr()
            mean_corr = (np.nansum(corr_df.to_numpy()) - n_in) / (n_in * (n_in - 1))
            vif = 1 + (n_in - 1) * mean_corr
            df = y.shape[0] - 2
        denom = np.sqrt(svar * (vif / n_in + 1.0 / n_out))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.divide(mean_diff, denom)
        prob = t_stat.apply(lambda t: scipy.stats.t.sf(np.abs(t), df) * 2)  # noqa: B023

        t_stat_dict[feature_set] = t_stat
        prob_dict[feature_set] = prob

    t_stat_df = pd.DataFrame(t_stat_dict, index=factor_names)
    prob_df = pd.DataFrame(prob_dict, index=factor_names)
    t_stat_df.fillna(0.0, inplace=True)
    prob_df.fillna(1.0, inplace=True)
    if adjust_p:
        prob_adj_df = prob_df.apply(
            lambda p: multitest.multipletests(p, method=p_adj_method)[1],
            axis=1,
            result_type="broadcast",
        )

    if "all" not in sign:
        prob_df[t_stat_df < 0.0] = 1.0
        if adjust_p:
            prob_adj_df[t_stat_df < 0.0] = 1.0
        t_stat_df[t_stat_df < 0.0] = 0.0

    result = {Cache.TEST_T: t_stat_df, Cache.TEST_P: prob_df}
    if adjust_p:
        result[Cache.TEST_P_ADJ] = prob_adj_df

    if use_prior_mask and cache:
        for key, rdf in result.items():
            factor_names = rdf.columns
            model_cache.update_factor_metadata(
                pd.DataFrame(
                    np.diag(rdf.loc[factor_names, factor_names]),
                    index=factor_names,
                    columns=[f"{key}_{sign}_{view_idx}"],
                )
            )

    return result


def test(
    model,
    view_idx: Index = "all",
    factor_idx: Index = "all",
    feature_sets: pd.DataFrame = None,
    sign: Optional[str] = None,
    corr_adjust: bool = True,
    p_adj_method: str = "fdr_bh",
    min_size: int = 10,
    cache: bool = True,
    rename: bool = True,
):
    """Perform significance test of factor loadings against feature sets.

    Parameters
    ----------
    model : MuVI
        A MuVI model
    view_idx : Index, optional
        View index, by default "all"
    factor_idx : Index, optional
        Factor index, by default "all"
    feature_sets : pd.DataFrame, optional
        Boolean dataframe with feature sets in each row, by default None
    sign : str, optional
        Two sided ("all") or one-sided ("neg", "pos" or None for both directions),
        by default None ("neg" and "pos")
    corr_adjust : bool, optional
        Whether to adjust for multiple testing, by default True
    p_adj_method : str, optional
        Adjustment method for multiple testing, by default "fdr_bh"
    min_size : int, optional
        Lower size limit for feature sets to be considered, by default 10
    cache : bool, optional
        Whether to store results in the model cache, by default True
    rename : bool, optional
        Whether to rename overwritten factors (FDR > 0.05), by default True

    Returns
    -------
    dict
        Dictionary of test results with "t", "p" and "p_adj" keys
        and pd.DataFrame values with factor_idx as index,
        and index of feature_sets as columns
    """

    view_indices = _normalize_index(view_idx, model.view_names, as_idx=False)
    use_prior_mask = feature_sets is None
    if use_prior_mask:
        view_indices = [vi for vi in view_indices if vi in model.informed_views]

    if len(view_indices) == 0:
        if use_prior_mask:
            raise ValueError(
                "`feature_sets` is None, and none of the selected views are informed."
            )
        raise ValueError(f"No valid views found for `view_idx={view_idx}`.")

    signs = [sign]
    if sign is None:
        signs = ["neg", "pos"]

    results = {}

    for sign in signs:
        results[sign] = {}
        for view_idx in view_indices:
            try:
                results[sign][view_idx] = _test_single_view(
                    model,
                    view_idx=view_idx,
                    factor_idx=factor_idx,
                    feature_sets=feature_sets,
                    sign=sign,
                    corr_adjust=corr_adjust,
                    p_adj_method=p_adj_method,
                    min_size=min_size,
                    cache=cache,
                )
            except ValueError as e:
                logger.warning(e)
                results[sign][view_idx] = {
                    Cache.TEST_T: pd.DataFrame(),
                    Cache.TEST_P: pd.DataFrame(),
                }
                if p_adj_method is not None:
                    results[sign][view_idx][Cache.TEST_P_ADJ] = pd.DataFrame()
                continue

    if cache and rename:
        dfs = []
        for _, sign_results in results.items():
            for view_name, view_results in sign_results.items():
                p_adj = view_results[Cache.TEST_P_ADJ].copy()
                p_adj = p_adj.loc[p_adj.columns, p_adj.columns].copy()
                dfs.append(
                    pd.DataFrame(
                        np.diag(p_adj), index=[p_adj.index], columns=[view_name]
                    )
                )
        df = pd.concat(dfs, axis=1)

        new_factor_names = []
        overwritten_idx = 0
        for k in model.factor_names:
            if k not in df.index:
                new_factor_names.append(k)
                continue
            if (df.loc[k, :] > 0.05).all(None):
                new_factor_names.append(f"factor_{overwritten_idx}")
                overwritten_idx += 1
            else:
                new_factor_names.append(k)

        model.factor_names = pd.Index(new_factor_names)[
            invert_permutation(model.factor_order)
        ]
    return results


def invert_permutation(p):
    p = np.asanyarray(p)
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


# scanpy
def _optional_neighbors(model, **kwargs):
    model_cache = setup_cache(model)
    if "neighbors" not in model_cache.factor_adata.uns:
        logger.warning("Computing a neighborhood graph first.")
        neighbors(model, **kwargs)


def neighbors(model, **kwargs):
    """Compute a neighborhood graph of observations."""
    model_cache = setup_cache(model)
    kwargs["use_rep"] = model_cache.use_rep
    return sc.pp.neighbors(model_cache.factor_adata, **kwargs)


def _cluster(model, cluster_fn, **kwargs):
    _optional_neighbors(model)
    return cluster_fn(setup_cache(model).factor_adata, **kwargs)


def leiden(model, **kwargs):
    """Cluster samples according to leiden algorithm."""
    return _cluster(model, cluster_fn=sc.tl.leiden, **kwargs)


def louvain(model, **kwargs):
    """Cluster samples according to louvain algorithm."""
    return _cluster(model, cluster_fn=sc.tl.louvain, **kwargs)


def tsne(model, **kwargs):
    """Compute tSNE embeddings."""
    model_cache = setup_cache(model)
    kwargs["use_rep"] = model_cache.use_rep
    return sc.tl.tsne(model_cache.factor_adata, **kwargs)


def umap(model, **kwargs):
    """Compute UMAP embeddings."""
    _optional_neighbors(model)
    return sc.tl.umap(setup_cache(model).factor_adata, **kwargs)


def rank(model, groupby, method="wilcoxon", **kwargs):
    """Rank factors for characterizing groups."""
    if "rankby_abs" not in kwargs:
        kwargs["rankby_abs"] = True
    return sc.tl.rank_genes_groups(
        setup_cache(model).factor_adata, groupby, method=method, **kwargs
    )


def dendrogram(model, groupby, **kwargs):
    """Compute hierarchical clustering for the given `groupby` categories."""
    model_cache = setup_cache(model)
    kwargs["use_rep"] = model_cache.use_rep
    kwargs["n_pcs"] = None
    return sc.tl.dendrogram(model_cache.factor_adata, groupby, **kwargs)


def posterior_feature_sets(
    model,
    view_idx: Index = "all",
    factor_idx: Index = "all",
    r2_thresh: Union[int, float] = 0.95,
    knee_sensitivity: float = 1.0,
    dir_path=None,
    **kwargs,
):
    model_cache = setup_cache(model)

    r2_cols = [f"{Cache.METRIC_R2}_{vn}" for vn in model.view_names]
    r2_df = model_cache.factor_metadata[r2_cols]

    if r2_df.isna().all(None):
        logger.warning(
            "Unable to filter factors based on variance explained.\n"
            "Run `muvi.tl.variance_explained` to compute "
            "the variance explained by each factor first. "
            "Extracting the posterior feature sets across all factors."
        )
        r2_thresh = model.n_factors

    if int(r2_thresh) == 1:
        logger.warning(
            "Unable to filter factors based on variance explained.\n"
            f"`r2_thresh` of `{r2_thresh}` is ambiguous, `r2_thresh` must be "
            "less than 1.0 or an integer greater than 1. "
            "Extracting the posterior feature sets across all factors."
        )
        r2_thresh = model.n_factors

    ws = model.get_factor_loadings(view_idx, factor_idx, as_df=True)

    posterior_feature_sets = {}

    for view_name, view_loadings in ws.items():
        _r2_thresh = r2_thresh
        view_feature_sets = {}
        r2_sorted = r2_df[f"{Cache.METRIC_R2}_{view_name}"].sort_values(ascending=False)
        factor_subset = r2_sorted.index
        if _r2_thresh < 1.0:
            _r2_thresh = (r2_sorted.cumsum() / r2_sorted.sum() < _r2_thresh).sum() + 1

        factor_subset = r2_sorted.iloc[: int(_r2_thresh)].index

        factor_subset = factor_subset.tolist()
        if len(factor_subset) < model.n_factors:
            logger.info(
                "Extracting the posterior feature sets from "
                f"{len(factor_subset)}/{model.n_factors} "
                f"factors for view {view_name}."
            )
        for factor_name in factor_subset:
            loadings = (
                view_loadings.loc[factor_name, :].abs().sort_values(ascending=False)
            )
            kn = KneeLocator(
                range(len(loadings)),
                loadings,
                S=knee_sensitivity,
                curve="convex",
                direction="decreasing",
                **kwargs,
            )
            view_feature_sets[factor_name] = loadings.iloc[: kn.knee].index.tolist()
        posterior_feature_sets[view_name] = view_feature_sets

    if dir_path is not None:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        pfs_df = pd.DataFrame(posterior_feature_sets)
        pfs_df["name"] = pfs_df.index.astype(str)
        for view_name in ws:
            feature_sets = fs.from_dataframe(
                pfs_df.loc[~pfs_df[view_name].isna()],
                name=f"muvi_posterior_{view_name}",
                features_col=view_name,
            )
            feature_sets.to_gmt(Path(dir_path) / f"muvi_posterior_{view_name}.gmt")

    return posterior_feature_sets


def from_adata(
    adata,
    obs_key: Optional[str] = None,
    prior_mask_key: Optional[str] = None,
    covariate_key: Optional[str] = None,
    **kwargs,
):
    if obs_key is None:
        observations = [adata.to_df().copy()]
    else:
        if obs_key not in adata.obsm:
            raise ValueError(f"Invalid `obs_key`, `{obs_key}` not found in `obsm`.")
        observations = [adata.obsm[obs_key].copy()]
    prior_masks = None
    if prior_mask_key is not None:
        if prior_mask_key not in adata.varm:
            logger.warning(
                f"Invalid `prior_mask_key`, `{prior_mask_key}` not found in `varm`."
            )
        else:
            prior_masks = [adata.varm[prior_mask_key].T.copy()]

    covariates = None
    if covariate_key is not None:
        if covariate_key not in adata.obsm:
            raise ValueError(
                f"Invalid `covariate_key`, `{covariate_key}` not found in `obsm`."
            )
        covariates = adata.obsm[covariate_key].copy()

    return MuVI(observations, prior_masks=prior_masks, covariates=covariates, **kwargs)


def from_mdata(
    mdata,
    obs_key: Optional[str] = None,
    prior_mask_key: Optional[str] = None,
    covariate_key: Optional[str] = None,
    **kwargs,
):
    view_names = sorted(mdata.mod.keys())

    observations = {}
    prior_masks = {}

    for view_name in view_names:
        if obs_key is None:
            observations[view_name] = mdata.mod[view_name].to_df().copy()
        else:
            if obs_key not in mdata.mod[view_name].obsm:
                raise ValueError(f"Invalid `obs_key`, `{obs_key}` not found in `obsm`.")
            observations[view_name] = mdata.mod[view_name].obsm[obs_key].copy()

        if prior_mask_key is not None:
            if prior_mask_key not in mdata.mod[view_name].varm:
                logger.warning(
                    f"Invalid `prior_mask_key`, `{prior_mask_key}` not found in `varm`."
                )
            else:
                prior_masks[view_name] = (
                    mdata.mod[view_name].varm[prior_mask_key].T.copy()
                )

    if len(prior_masks) == 0:
        prior_masks = None

    covariates = None
    if covariate_key is not None:
        if covariate_key not in mdata.obsm:
            raise ValueError(
                f"Invalid `covariate_key`, `{covariate_key}` not found in `obsm`."
            )
        covariates = mdata.obsm[covariate_key].copy()

    return MuVI(observations, prior_masks=prior_masks, covariates=covariates, **kwargs)


def to_mdata(
    model,
    factor_scores_key="Z",
    covariates_key="X",
    loadings_key="W",
    betas_key="B",
    mask_key="mask",
):
    """WIP"""
    obs_dict = model.get_observations(as_df=True)
    loadings_dict = model.get_factor_loadings(as_df=True)

    betas_dict = {}
    if model.n_covariates > 0:
        betas_dict = model.get_covariate_coefficients(as_df=True)

    masks_dict = {}
    if model._informed:
        masks_dict = model.get_prior_masks(as_df=True)

    adata_dict = {}
    for vn in model.view_names:
        adata = ad.AnnData(
            obs_dict[vn], varm={loadings_key: loadings_dict[vn].T}, dtype=np.float32
        )
        if betas_dict:
            adata.varm[betas_key] = betas_dict[vn].T
        if masks_dict:
            adata.varm[mask_key] = masks_dict[vn].T
        adata_dict[vn] = adata

    mdata = md.MuData(adata_dict)
    if model.n_factors > 0:
        mdata.obsm[factor_scores_key] = model.get_factor_scores(as_df=True)

    if model.n_covariates > 0:
        mdata.obsm[covariates_key] = model.get_covariates(as_df=True)

    return mdata


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def save(model, dir_path="."):
    model_path = Path(dir_path) / "model.pkl"
    params_path = Path(dir_path) / "params.save"
    factor_adata_path = Path(dir_path) / "factor.h5ad"
    cov_adata_path = Path(dir_path) / "cov.h5ad"
    for pth in [model_path, params_path, factor_adata_path, cov_adata_path]:
        if pth.exists():
            logger.warning(f"`{pth}` already exists, overwriting.")

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    factor_adata = None
    cov_adata = None
    # clear cache
    if model._cache is not None:
        factor_adata = model._cache.factor_adata
        cov_adata = model._cache.cov_adata
        model._cache.factor_adata = None
        model._cache.cov_adata = None
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # fixes error of anndata that renames .obsm index
    for adata in [factor_adata, cov_adata]:
        if adata is not None:
            for key in adata.obsm:
                if isinstance(adata.obsm[key], pd.DataFrame):
                    adata.obsm[key].index = adata.obs_names.copy()

    # save adata(s) and restore cache
    try:
        if factor_adata is not None:
            factor_adata.write_h5ad(factor_adata_path)
    except Exception as e:
        logger.error(e)
    finally:
        if model._cache is not None:
            model._cache.factor_adata = factor_adata

    try:
        if cov_adata is not None:
            cov_adata.write_h5ad(cov_adata_path)
    except Exception as e:
        logger.error(e)
    finally:
        if model._cache is not None:
            model._cache.cov_adata = cov_adata

    pyro.get_param_store().save(params_path)
    return dir_path


def load(dir_path=".", with_params=True, map_location=None):
    model_path = Path(dir_path) / "model.pkl"
    params_path = Path(dir_path) / "params.save"
    factor_adata_path = Path(dir_path) / "factor.h5ad"
    cov_adata_path = Path(dir_path) / "cov.h5ad"

    with open(model_path, "rb") as f:
        model = pickle.load(f) if map_location is None else CPUUnpickler(f).load()

    if model._cache is not None:
        factor_adata = None
        cov_adata = None

        if factor_adata_path.exists():
            factor_adata = sc.read_h5ad(factor_adata_path)
        if cov_adata_path.exists():
            cov_adata = sc.read_h5ad(cov_adata_path)

        model._cache.factor_adata = factor_adata
        model._cache.cov_adata = cov_adata
    if with_params:
        pyro.get_param_store().load(params_path, map_location=map_location)
    # model = pyro.module("MuVI", model, update_module_params=True)  # noqa: ERA001
    return model


def optim_perm(matrix):
    n, n = matrix.shape
    res = linprog(
        -matrix.ravel(),
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
