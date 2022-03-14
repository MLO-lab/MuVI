import logging

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from sklearn.metrics import mean_squared_error
from statsmodels.stats import multitest
from tqdm import tqdm

logger = logging.getLogger(__name__)


RMSE = "rmse"
R2 = "r2"

TEST_POS = "pos"
TEST_NEG = "neg"
TEST_ALL = "all"
TEST_P = "p"
TEST_P_ADJ = "p_adj"
TEST_T = "t"

FAC_DF = "factors_df"
COV_DF = "covs_df"


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


def _setup_cache(model, overwrite=False):
    # check if model has been trained?
    if overwrite:
        model._cache = {}

    for view_score_key in [R2, RMSE]:
        if view_score_key not in model._cache:
            model._cache[f"view_{view_score_key}"] = {}

    columns = []
    for key in [R2, RMSE, TEST_P, TEST_P_ADJ, TEST_T]:
        for vn in model.view_names:
            columns.append(f"{key}_{vn}")
    if FAC_DF not in model._cache:
        model._cache[FAC_DF] = pd.DataFrame(
            np.nan, index=model.factor_names, columns=columns
        )
    if COV_DF not in model._cache:
        model._cache[COV_DF] = pd.DataFrame(
            np.nan, index=model.covariate_names, columns=columns
        )
    return model._cache


def _recon_error(
    model, view_idx, factor_idx, cov_idx, factor_wise, cov_wise, metric_label, metric_fn
):
    _setup_cache(model)

    valid_view_idx = view_idx is not None and len(view_idx) > 1
    if not valid_view_idx:
        raise ValueError(f"Invalid view index: {view_idx}")
    valid_factor_idx = factor_idx is not None and len(factor_idx) > 1
    valid_cov_idx = cov_idx is not None and len(cov_idx) > 1
    if not valid_factor_idx and not valid_cov_idx:
        raise ValueError(f"Invalid factor and covariate index: {factor_idx}, {cov_idx}")

    factor_wise &= valid_factor_idx
    cov_wise &= valid_cov_idx

    ys = model.get_observations(view_idx)

    n_factors = 0
    n_covariates = 0
    if valid_factor_idx:
        z = model.get_factor_scores(factor_idx=factor_idx, as_df=True)
        ws = model.get_factor_loadings(view_idx, factor_idx)
        factor_names = z.columns
        n_factors = len(factor_names)
        z = z.to_numpy()
    if valid_cov_idx:
        x = model.get_covariates(cov_idx=cov_idx, as_df=True)
        betas = model.get_covariate_coefficients(view_idx, cov_idx)
        cov_names = x.columns
        n_covariates = len(cov_names)
        x = x.to_numpy()

    view_scores = {}
    factor_scores = {}
    cov_scores = {}

    for vn in ys.keys():
        score_key = f"{metric_label}_{vn}"
        view_scores[vn] = 0.0
        factor_scores[score_key] = np.zeros(n_factors)
        cov_scores[score_key] = np.zeros(n_covariates)
        y_true_view = ys[vn]
        y_true_view = np.nan_to_num(y_true_view, nan=0.0)
        y_pred_view = np.zeros_like(y_true_view)
        if valid_factor_idx:
            y_pred_view += z @ ws[vn]
        if valid_factor_idx:
            y_pred_view += x @ betas[vn]
        view_scores[vn] = metric_fn(y_true_view, y_pred_view)
        if factor_wise:
            for k in range(n_factors):
                y_pred_fac_k = np.outer(z[:, k], ws[vn][k, :])
                factor_scores[score_key][k] = metric_fn(y_true_view, y_pred_fac_k)
        if cov_wise:
            for k in range(n_covariates):
                y_pred_cov_k = np.outer(x[:, k], betas[vn][k, :])
                cov_scores[score_key][k] = metric_fn(y_true_view, y_pred_cov_k)

    factor_scores = pd.DataFrame(factor_scores, index=factor_names)
    cov_scores = pd.DataFrame(cov_scores, index=cov_names)

    model._cache[f"view_{metric_label}"].update(view_scores)
    if factor_wise:
        model._cache[FAC_DF].update(factor_scores)
    if cov_wise:
        model._cache[COV_DF].update(cov_scores)
    return view_scores, factor_scores, cov_scores


def rmse(
    model,
    view_idx="all",
    factor_idx="all",
    cov_idx="all",
    factor_wise=False,
    cov_wise=False,
):
    def _rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

    return _recon_error(
        model,
        view_idx,
        factor_idx,
        cov_idx,
        factor_wise,
        cov_wise,
        metric_label=RMSE,
        metric_fn=_rmse,
    )


def variance_explained(
    model,
    view_idx="all",
    factor_idx="all",
    cov_idx="all",
    factor_wise=False,
    cov_wise=False,
):
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
        metric_label=R2,
        metric_fn=_r2,
    )


def inspect_factor(model, view_idx, factor_idx, threshold=0.05):
    if model.prior_masks is None:
        raise ValueError("Model has not been informed prior to training.")
    # validate view_idx, factor_idx

    factor_loadings = model.get_factor_loadings(view_idx, as_list=False)[factor_idx, :]
    factor_mask = model.get_prior_mask(view_idx, as_list=False)[factor_idx, :]
    factor_loadings_abs = np.abs(factor_loadings)

    df = pd.DataFrame(
        {
            "Feature": model.feature_names[view_idx],
            "Mask": factor_mask,
            "Loading": factor_loadings,
            "Loading (abs)": factor_loadings_abs,
            "FP": ~factor_mask & factor_loadings_abs > threshold,
        }
    )

    df["Type"] = df["FP"].map({False: "Annotated", True: "Inferred"})

    result = {model.factor_names[factor_idx]: df}
    model._cache.update(result)
    return result


def test(
    model,
    view_idx,
    factor_idx="all",
    feature_sets=None,
    sign="all",
    corr_adjust=False,
    p_adj_method="fdr_bh",
    min_size=10,
):

    _setup_cache(model)
    use_prior_mask = feature_sets is None

    if not isinstance(view_idx, (str, int)) and view_idx != "all":
        raise IndexError(
            f"Invalid `view_idx`, `{view_idx}` must be a string or an integer."
        )
    if isinstance(view_idx, int):
        view_idx = model.view_names[view_idx]
    if view_idx not in model.view_names:
        raise IndexError(f"`{view_idx}` not found in the view names.")

    if use_prior_mask:
        print("No feature sets provided, extracting feature sets from prior mask.")
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

    if "pos" in sign.lower():
        factor_loadings[factor_loadings < 0] = 0.0
    if "neg" in sign.lower():
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
    # prob_df.fillna(1.0, inplace=True)
    prob_adj_df = prob_df.apply(
        lambda p: multitest.multipletests(p, method=p_adj_method)[1],
        axis=1,
        result_type="broadcast",
    )

    if "all" not in sign.lower():
        prob_df[t_stat_df < 0.0] = 1.0
        prob_adj_df[t_stat_df < 0.0] = 1.0
        t_stat_df[t_stat_df < 0.0] = 0.0

    result = {TEST_T: t_stat_df, TEST_P: prob_df, TEST_P_ADJ: prob_adj_df}

    if use_prior_mask:
        for key, rdf in result.items():
            factor_names = rdf.columns
            df = pd.DataFrame(
                np.diag(rdf.loc[factor_names, factor_names]),
                index=factor_names,
                columns=[f"{key}_{view_idx}"],
            )
            if key != TEST_T:
                df = df.clip(1e-10, 1.0)
            model._cache[FAC_DF].update(df)

    return result
