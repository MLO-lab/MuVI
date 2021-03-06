import logging
from typing import List, Union

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.optimize import linprog
from sklearn.metrics import mean_squared_error
from statsmodels.stats import multitest
from tqdm import tqdm

from muvi.core.index import _normalize_index
from muvi.core.models import MuVI
from muvi.tools.cache import Cache

logger = logging.getLogger(__name__)

Index = Union[int, str, List[int], List[str], np.ndarray, pd.Index]


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


# def filter_factors(model, factor_idx=None, min_r2=None, top_r2=None):
#     pass


def filter_factors(model, factor_idx: Index):
    """Filter factors for the current analysis."""
    factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)
    if len(factor_idx) == 0:
        raise ValueError("`factor_idx` is empty.")
    model_cache = setup_cache(model)
    return model_cache.filter_factors(factor_idx)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _recon_error(
    model,
    view_idx,
    factor_idx,
    cov_idx,
    factor_wise,
    cov_wise,
    subsample,
    cache,
    metric_label,
    metric_fn,
):
    if view_idx is None:
        raise ValueError(f"Invalid view index: {view_idx}")

    valid_factor_idx = factor_idx is not None and model.n_factors > 0
    valid_cov_idx = cov_idx is not None and model.n_covariates > 0
    if not valid_factor_idx and not valid_cov_idx:
        raise ValueError(
            "Both `factor_idx` and `cov_idx` are None, at least one is required."
        )

    factor_wise &= valid_factor_idx
    cov_wise &= valid_cov_idx

    sample_idx = "all"
    if subsample is not None and subsample > 0 and subsample < model.n_samples:
        logger.info(
            "Estimating %s with a random sample of %s samples.", metric_label, subsample
        )
        sample_idx = np.random.choice(model.n_samples, subsample, replace=False)

    ys = model.get_observations(view_idx, sample_idx=sample_idx)
    view_names = list(ys.keys())

    view_scores_key = f"view_{metric_label}"
    cache_columns = [f"{metric_label}_{vn}" for vn in model.view_names]

    model_cache = setup_cache(model)

    # results not in cache, carry on computing
    n_samples = list(ys.values())[0].shape[0]
    if subsample is None and n_samples > 10000 and (factor_wise or cov_wise):
        logger.warning(
            "Computing %s with more %s samples, this may take some time. "
            "Consider estimating %s by setting `subsample` to a smaller number.",
            metric_label,
            n_samples,
            metric_label,
        )

    n_factors = 0
    n_covariates = 0
    factor_names = []
    cov_names = []
    if valid_factor_idx:
        z = model.get_factor_scores(
            sample_idx=sample_idx, factor_idx=factor_idx, as_df=True
        )
        ws = model.get_factor_loadings(view_idx, factor_idx)
        factor_names = z.columns
        n_factors = len(factor_names)
        z = z.to_numpy()
    if valid_cov_idx:
        x = model.get_covariates(sample_idx=sample_idx, cov_idx=cov_idx, as_df=True)
        betas = model.get_covariate_coefficients(view_idx, cov_idx)
        cov_names = x.columns
        n_covariates = len(cov_names)
        x = x.to_numpy()

    view_scores = {}
    factor_scores = {}
    cov_scores = {}

    for m, vn in enumerate(view_names):
        score_key = cache_columns[m]
        view_scores[vn] = 0.0
        factor_scores[score_key] = np.zeros(n_factors)
        cov_scores[score_key] = np.zeros(n_covariates)
        y_true_view = ys[vn]
        y_true_view = np.nan_to_num(y_true_view, nan=0.0)
        y_pred_view = np.zeros_like(y_true_view)
        if valid_factor_idx:
            y_pred_view += z @ ws[vn]
            if model.likelihoods[vn] == "bernoulli":
                y_pred_view = _sigmoid(y_pred_view)
        if valid_cov_idx:
            y_pred_view += x @ betas[vn]
        view_scores[vn] = metric_fn(y_true_view, y_pred_view)
        if factor_wise:
            for k in range(n_factors):
                y_pred_fac_k = np.outer(z[:, k], ws[vn][k, :])
                if model.likelihoods[vn] == "bernoulli":
                    y_pred_fac_k = _sigmoid(y_pred_fac_k)
                factor_scores[score_key][k] = metric_fn(y_true_view, y_pred_fac_k)
        if cov_wise:
            for k in range(n_covariates):
                y_pred_cov_k = np.outer(x[:, k], betas[vn][k, :])
                cov_scores[score_key][k] = metric_fn(y_true_view, y_pred_cov_k)

    factor_scores = pd.DataFrame(factor_scores, index=factor_names)
    cov_scores = pd.DataFrame(cov_scores, index=cov_names)
    if cache:
        model_cache.update_uns(view_scores_key, view_scores)
        model_cache.update_factor_metadata(factor_scores)
        model_cache.update_cov_metadata(cov_scores)
    return view_scores, factor_scores, cov_scores


def rmse(
    model,
    view_idx: Index = "all",
    factor_idx: Index = "all",
    cov_idx: Index = "all",
    factor_wise: bool = True,
    cov_wise: bool = True,
    subsample: int = 0,
    cache: bool = True,
):
    """Compute RMSE.

    Parameters
    ----------
    model : MuVI
        A MuVI model
    view_idx : Index, optional
        View index, by default "all"
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
    """

    def _rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

    return _recon_error(
        model,
        view_idx,
        factor_idx,
        cov_idx,
        factor_wise,
        cov_wise,
        subsample,
        cache,
        metric_label=Cache.METRIC_RMSE,
        metric_fn=_rmse,
    )


def variance_explained(
    model,
    view_idx: Index = "all",
    factor_idx: Index = "all",
    cov_idx: Index = "all",
    factor_wise: bool = True,
    cov_wise: bool = True,
    subsample: int = 0,
    cache: bool = True,
):
    """Compute R2.

    Parameters
    ----------
    model : MuVI
        A MuVI model
    view_idx : Index, optional
        View index, by default "all"
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
    """

    def _r2(y_true, y_pred):
        ss_res = np.square(y_true - y_pred).sum()
        ss_tot = np.square(y_true).sum()
        return 1 - (ss_res / ss_tot)

    return _recon_error(
        model,
        view_idx,
        factor_idx,
        cov_idx,
        factor_wise,
        cov_wise,
        subsample,
        cache,
        metric_label=Cache.METRIC_R2,
        metric_fn=_r2,
    )


def test(
    model,
    view_idx: Union[str, int],
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
            "No feature sets provided, extracting feature sets from prior mask."
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
    i = 0
    for feature_set in tqdm(feature_sets.index.tolist()):
        i += 1
        pathway_features = feature_sets.loc[feature_set, :]

        features_in = factor_loadings.loc[:, pathway_features]
        features_out = factor_loadings.loc[:, ~pathway_features]

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
            corr_df = y.loc[:, pathway_features].corr()
            mean_corr = (np.nansum(corr_df.to_numpy()) - n_in) / (n_in * (n_in - 1))
            vif = 1 + (n_in - 1) * mean_corr
            df = y.shape[0] - 2
        denom = np.sqrt(svar * (vif / n_in + 1.0 / n_out))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.divide(mean_diff, denom)
        prob = t_stat.apply(lambda t: scipy.stats.t.sf(np.abs(t), df) * 2)

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


def from_mdata(mdata, prior_mask_key: str = None, covariate_key: str = None, **kwargs):

    view_names = sorted(mdata.mod.keys())
    observations = {
        view_name: mdata.mod[view_name].to_df().copy() for view_name in view_names
    }
    prior_masks = {}
    if prior_mask_key is not None:
        for view_name in view_names:
            if prior_mask_key in mdata.mod[view_name].varm:
                prior_masks[view_name] = (
                    mdata.mod[view_name].varm[prior_mask_key].T.copy()
                )
            else:
                logger.warning(f"No prior information found for `{view_name}`.")

    covariates = None
    if covariate_key is not None:
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
        adata = ad.AnnData(obs_dict[vn], varm={loadings_key: loadings_dict[vn].T})
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
