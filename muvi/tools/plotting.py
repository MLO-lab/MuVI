"""Collection of plotting functions."""
import logging
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_cf

import muvi
from muvi.core.index import _normalize_index

sns.set_theme()
sns.set_style("whitegrid")

logger = logging.getLogger(__name__)


HEATMAP = "heatmap"
MATRIXPLOT = "matrixplot"
DOTPLOT = "dotplot"
TRACKSPLOT = "tracksplot"
VIOLIN = "violin"
STACKED_VIOLIN = "stacked_violin"
PL_TYPES = [HEATMAP, MATRIXPLOT, DOTPLOT, TRACKSPLOT, VIOLIN, STACKED_VIOLIN]


def savefig_or_show(
    writekey: str,
    show: bool = None,
    dpi: int = None,
    ext: str = None,
    save: Union[bool, str, None] = None,
):
    return sc.pl._utils.savefig_or_show(writekey, show, dpi, ext, save)


def _get_model_cache(model):
    cache = model._cache
    if cache is None:
        raise ValueError("Model cache not initialised, execute muvi.tl.* first.")
    return cache


def _lines(ax, positions, ymin, ymax, horizontal=False, **kwargs):
    if horizontal:
        ax.hlines(positions, ymin, ymax, **kwargs)
    else:
        ax.vlines(positions, ymin, ymax, **kwargs)
    return ax


def lined_heatmap(data, figsize=None, hlines=None, vlines=None, **kwargs):
    """Plot heatmap with horizontal or vertical lines."""
    if figsize is None:
        figsize = (20, 2)
    fig, g = plt.subplots(figsize=figsize)
    g = sns.heatmap(data, ax=g, **kwargs)
    if hlines is not None:
        _lines(
            g,
            hlines,
            *sorted(g.get_xlim()),
            horizontal=True,
            lw=1.0,
            linestyles="dashed",
        )
    if vlines is not None:
        _lines(
            g,
            vlines,
            *sorted(g.get_ylim()),
            horizontal=False,
            lw=1.0,
            linestyles="dashed",
        )
    return g


def variance_explained(
    model,
    sort=50,
    show: bool = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    """Heatmap of variance explained, see `muvi.tl.variance_explained`."""
    # _, r2_fac, r2_cov = muvi.tl.variance_explained(
    #     model,
    #     factor_wise=kwargs.pop("factor_wise", True),
    #     cov_wise=kwargs.pop("cov_wise", True),
    #     **kwargs,
    # )

    model_cache = _get_model_cache(model)
    r2_fac = model_cache.factor_metadata
    r2_cov = model_cache.cov_metadata
    if r2_fac is None and r2_cov is None:
        raise ValueError(
            "No scores found in model cache, rerun `muvi.tl.variance_explained`."
        )

    r2_columns = [f"{model_cache.METRIC_R2}_{vn}" for vn in model.view_names]
    if r2_fac is not None:
        r2_fac = r2_fac.loc[:, r2_columns].T
    if r2_cov is not None:
        r2_cov = r2_cov.loc[:, r2_columns].T

    if sort:
        r2_argsort = r2_fac.sum().argsort()[::-1]
        r2_fac = r2_fac.iloc[:, r2_argsort]
    if sort > 1:
        r2_fac = r2_fac.iloc[:, :sort]

    r2_joint = r2_fac
    if r2_cov is not None:
        r2_joint = pd.concat([r2_fac, r2_cov], axis=1)

    if r2_joint.isna().all(None):
        raise ValueError(
            "No scores found in model cache, rerun `muvi.tl.variance_explained`."
        )

    r2_joint.index = r2_joint.index.str[len(model_cache.METRIC_R2) + 1 :]

    figsize = (max(20, r2_joint.shape[1] // 5), r2_joint.shape[0])
    vlines = r2_fac.shape[1]
    if r2_cov is None or r2_cov.empty:
        vlines = None

    g = lined_heatmap(r2_joint, figsize=figsize, vlines=vlines, **kwargs)
    if not show:
        return g
    return savefig_or_show("variance_explained", show=show, save=save)


def factors_overview(
    model,
    view_idx,
    one_sided=True,
    alpha=0.1,
    sig_only=False,
    adjusted=False,
    sort=25,
    show: bool = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    """Scatterplot of variance explained along with significance test resutls,
    see `muvi.tl.test`."""
    if isinstance(view_idx, int):
        view_idx = model.view_names[view_idx]

    model_cache = _get_model_cache(model)
    df = model_cache.factor_metadata.copy()

    name_col = "Factor"
    df[name_col] = df.index.astype(str)

    size_col = None
    if model._informed:
        size_col = "Size"
        df[size_col] = model.get_prior_masks(view_idx, as_df=True)[view_idx].sum(1)
        df.loc[df[name_col].str.contains("dense", case=False), size_col] = 0

    sign_dict = {model_cache.TEST_ALL: " (*)"}
    if one_sided:
        sign_dict = {model_cache.TEST_NEG: " (-)", model_cache.TEST_POS: " (+)"}
    joint_p_col = "p_min"
    direction_col = "direction"

    p_col = "p"
    if adjusted:
        p_col = "p_adj"

    p_df = df[[f"{p_col}_{sign}_{view_idx}" for sign in sign_dict.keys()]]
    if p_df.isna().all(None):
        raise ValueError("No test results found in model cache, rerun `muvi.tl.test`.")

    p_df = p_df.fillna(1.0)
    df[joint_p_col] = p_df.min(axis=1).clip(1e-10, 1.0)
    df[direction_col] = p_df.idxmin(axis=1).str[len(p_col) + 1 : len(p_col) + 4]

    if alpha is None:
        alpha = 1.0
    if alpha <= 0:
        logger.warning("Negative or zero `alpha`, setting `alpha` to 0.01.")
        alpha = 0.01
    if alpha > 1.0:
        logger.warning("`alpha` larger than 1.0, setting `alpha` to 1.0.")
        alpha = 1.0
    df.loc[df[joint_p_col] > alpha, direction_col] = ""
    if sig_only:
        df = df.loc[df[direction_col] != "", :]

    df[name_col] = df[name_col] + df[direction_col].map(sign_dict).fillna("")

    neg_log_col = r"$-\log_{10}(FDR)$"
    df[neg_log_col] = -np.log10(df[joint_p_col])
    r2_col = f"r2_{view_idx}"
    if sort > 0:
        df = df.sort_values(r2_col, ascending=True)

    g = sns.scatterplot(
        data=df.iloc[-sort:],
        x=r2_col,
        y=name_col,
        hue=neg_log_col,
        palette=kwargs.pop("palette", "flare"),
        size=size_col,
        sizes=kwargs.pop("sizes", (50, 350)),
        **kwargs,
    )
    g.set_title(
        f"Overview top factors in {view_idx} "
        rf"($\alpha = {alpha:.{max(1, int(-np.log10(alpha)))}f}$)"
    )
    g.set(xlabel=r"$R^2$")
    if not show:
        return g
    return savefig_or_show(f"overview_view_{view_idx}", show=show, save=save)


def inspect_factor(
    model,
    view_idx,
    factor_idx,
    sort=25,
    threshold=0.05,
    show: bool = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    """Scatterplot of factor loadings for specific factors."""
    view_idx = _normalize_index(view_idx, model.view_names, as_idx=False)
    factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)

    if len(view_idx) > 1:
        logger.warning(
            "Currently supporting only one view, " "showing results for  `%s`.",
            view_idx,
        )
    view_idx = view_idx[0]

    if len(factor_idx) > 1:
        logger.warning(
            "Currently supporting only one factor, " "showing results for  `%s`.",
            factor_idx,
        )
    factor_idx = factor_idx[0]

    factor_loadings = model.get_factor_loadings(view_idx, factor_idx, as_df=True)[
        view_idx
    ].iloc[0, :]
    factor_mask = pd.Series(False, index=factor_loadings.index)
    if model._informed:
        factor_mask = model.get_prior_masks(view_idx, factor_idx, as_df=True)[
            view_idx
        ].iloc[0, :]
    else:
        logger.warning("Model not informed a priori, all features are inferred.")
    factor_loadings_abs = np.abs(factor_loadings)

    name_col = "Feature"
    loading_col = "Loading"
    abs_loading_col = "Loading (abs)"

    df = pd.DataFrame(
        {
            name_col: model.feature_names[view_idx],
            "Mask": factor_mask,
            loading_col: factor_loadings,
            abs_loading_col: factor_loadings_abs,
            "FP": ~factor_mask & factor_loadings_abs > threshold,
        }
    )

    type_col = "Type"
    df[type_col] = df["FP"].map({False: "Annotated", True: "Inferred"})

    if sort > 0:
        df = df.sort_values(abs_loading_col, ascending=True)
    g = sns.scatterplot(
        data=df.iloc[-sort:],
        x=loading_col,
        y=name_col,
        hue=type_col,
        hue_order=["Annotated", "Inferred"],
        palette={"Annotated": "black", "Inferred": "red"},
        s=kwargs.pop("s", (64)),
        **kwargs,
    )
    g.set_title(f"Overview factor {factor_idx} in {view_idx}")
    if not show:
        return g
    return savefig_or_show(f"overview_factor_{factor_idx}", show=show, save=save)


# source: https://github.com/DTrimarchi10/confusion_matrix
def confusion_matrix(
    model,
    view_idx,
    true_mask=None,
    threshold=0.1,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="binary",
    title=None,
):
    if true_mask is None:
        view_idx, true_mask = model.get_prior_masks(view_idx).popitem()
    cf = sklearn_cf(
        true_mask.flatten(),
        (
            np.abs(model.get_factor_loadings(view_idx).popitem()[1]).flatten()
            > threshold
        ),
    )
    if group_names is None:
        group_names = ["true neg", "false pos", "false neg", "true pos"]
    if categories is None:
        categories = ["0.0", "1.0"]
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}".format(
                accuracy
            ) + "\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    g = sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title is None:
        title = f"View {view_idx}"
    plt.title(title)

    return g
    # return accuracy, precision, recall, f1_score


def factor_activity(
    true_w,
    approx_w,
    true_mask,
    noisy_mask,
    factor_idx=0,
    ylim=None,
    **kwargs,
):

    true_w_col = true_w[factor_idx, :]
    w_col = approx_w[factor_idx, :]
    true_mask_col = true_mask[factor_idx, :]
    noisy_mask_col = noisy_mask[factor_idx, :]

    activity_df = pd.DataFrame(
        {
            "true_weight": true_w_col,
            "weight": w_col,
            "true_mask": true_mask_col,
            "noisy_mask": noisy_mask_col,
            "TP": true_mask_col * noisy_mask_col,
            "FP": (1 - true_mask_col) * noisy_mask_col,
            "TN": (1 - true_mask_col) * (1 - noisy_mask_col),
            "FN": true_mask_col * (1 - noisy_mask_col),
        }
    )
    activity_df.sort_values(["true_weight"], inplace=True)

    score_cols = ["TP", "FP", "TN", "FN"]

    assert (activity_df.loc[:, score_cols].values.sum(1) == 1).all()
    activity_df["state"] = (
        activity_df.loc[:, score_cols]
        .astype(np.int32)
        .dot(activity_df.loc[:, score_cols].columns + "+")
        .str[:-1]
    )
    activity_df["true state"] = [
        "on" if f > 0.5 else "off" for f in activity_df["true_mask"]
    ]
    activity_df["idx"] = list(range(len(w_col)))

    g = sns.scatterplot(
        data=activity_df,
        x="idx",
        y="weight",
        hue="state",
        hue_order=["TP", "FN", "TN", "FP"],
        style="true state",
        style_order=["on", "off"],
        size="state",
        sizes={"TP": 64, "FN": 64, "TN": 32, "FP": 32},
        linewidth=0.01,
        **kwargs,
    )
    g.set_xlabel("")
    joint_handles, joint_labels = g.get_legend_handles_labels()
    g.legend(
        # loc="lower right",
        handles=[h for i, h in enumerate(joint_handles) if i not in [0, 5]],
        labels=[h for i, h in enumerate(joint_labels) if i not in [0, 5]],
    )
    if ylim is not None:
        g.set(ylim=ylim)

    return g, activity_df


# scanpy plotting..

# plot latent embeddings
def _embed(model, color, pl_fn, **kwargs):
    return pl_fn(_get_model_cache(model).factor_adata, color=color, **kwargs)


def tsne(model, color, **kwargs):
    if "X_tsne" not in _get_model_cache(model).factor_adata.obsm:
        logger.warning("Computing tSNE embeddings first.")
        muvi.tl.tsne(model)
    return _embed(model, color, pl_fn=sc.pl.tsne, **kwargs)


def umap(model, color, **kwargs):
    if "X_umap" not in _get_model_cache(model).factor_adata.obsm:
        logger.warning("Computing UMAP embeddings first.")
        muvi.tl.umap(model)
    return _embed(model, color, pl_fn=sc.pl.umap, **kwargs)


# plot groups of observations against (subset of) factors
def group(model, factor_idx, groupby, pl_type=HEATMAP, **kwargs):

    pl_type = pl_type.lower().strip()

    if (pl_type in MATRIXPLOT or pl_type in DOTPLOT) and "colorbar_title" not in kwargs:
        kwargs["colorbar_title"] = "Average scores\nin group"

    if pl_type in DOTPLOT and "size_title" not in kwargs:
        kwargs["size_title"] = "Fraction of samples\nin group (%)"

    type_to_fn = {
        HEATMAP: sc.pl.heatmap,
        MATRIXPLOT: sc.pl.matrixplot,
        DOTPLOT: sc.pl.dotplot,
        TRACKSPLOT: sc.pl.tracksplot,
        VIOLIN: sc.pl.violin,
        STACKED_VIOLIN: sc.pl.stacked_violin,
    }

    factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)

    try:
        pl_fn = type_to_fn[pl_type]
    except KeyError as e:
        raise ValueError(
            f"`{pl_type}` is not valid. Select one of {','.join(PL_TYPES)}."
        ) from e

    return pl_fn(_get_model_cache(model).factor_adata, factor_idx, groupby, **kwargs)


# plot ranked factors against groups of observations
def rank(model, n_factors=10, pl_type=None, sep_groups=True, **kwargs):

    factor_adata = _get_model_cache(model).factor_adata
    if "rank_genes_groups" not in factor_adata.uns:
        raise ValueError(
            "No group-wise ranking found, run `muvi.tl.rank_genes_groups first.`"
        )

    groupby = factor_adata.uns["rank_genes_groups"]["params"]["groupby"]
    dendrogram_key = "dendrogram_" + groupby
    if dendrogram_key not in factor_adata.uns:
        logger.warning(
            "dendrogram data not found (using key=%s). "
            "Running `muvi.tl.dendrogram` with default parameters. "
            "For fine tuning "
            "it is recommended to run `muvi.tl.dendrogram` independently."
        )
        muvi.tl.dendrogram(model, groupby)

    if pl_type is None:
        logger.warning("`pl_type` is None, defaulting to `sc.pl.rank_genes_groups`.")
        pl_type = ""
    pl_type = pl_type.lower().strip()
    n_factors = kwargs.pop("n_genes", n_factors)

    type_to_fn = {
        "": sc.pl.rank_genes_groups,
        HEATMAP: sc.pl.rank_genes_groups_heatmap,
        MATRIXPLOT: sc.pl.rank_genes_groups_matrixplot,
        DOTPLOT: sc.pl.rank_genes_groups_dotplot,
        TRACKSPLOT: sc.pl.rank_genes_groups_tracksplot,
        VIOLIN: sc.pl.rank_genes_groups_violin,
        STACKED_VIOLIN: sc.pl.rank_genes_groups_stacked_violin,
    }

    n_groups = len(
        factor_adata.obs[
            factor_adata.uns["rank_genes_groups"]["params"]["groupby"]
        ].unique()
    )

    positions = np.linspace(
        n_factors, n_factors * n_groups, num=n_groups - 1, endpoint=False
    )

    try:
        pl_fn = type_to_fn[pl_type]
    except KeyError as e:
        raise ValueError(
            f"`{pl_type}` is not valid. Select one of {','.join(PL_TYPES)}."
        ) from e

    if not sep_groups:
        return pl_fn(
            factor_adata,
            n_genes=n_factors,
            **kwargs,
        )

    show = kwargs.pop("show", None)
    save = kwargs.pop("save", None)
    _pl = pl_fn(
        factor_adata,
        n_genes=n_factors,
        show=False,
        save=None,
        **kwargs,
    )

    # add line separation
    g = None
    if pl_type == HEATMAP:
        g = _pl["heatmap_ax"]
        ymin = -0.5
        ymax = factor_adata.n_obs
        positions -= 0.5
    if pl_type == MATRIXPLOT:
        g = _pl["mainplot_ax"]
        ymin = 0.0
        ymax = n_groups
    if pl_type == DOTPLOT or pl_type == STACKED_VIOLIN:
        g = _pl["mainplot_ax"]
        ymin = -0.5
        ymax = n_groups + 0.5

    if g is not None:
        g = _lines(
            g,
            positions,
            ymin=ymin,
            ymax=ymax,
            horizontal=kwargs.pop("swap_axes", False),
            lw=0.5,
            color="black",
            linestyles="dashed",
            zorder=10,
            clip_on=False,
        )
    writekey = "rank"
    if len(pl_type) > 0:
        writekey += f"_{pl_type}"
    return savefig_or_show(writekey, show=show, save=save)


def clustermap(model, factor_idx="all", **kwargs):
    factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)
    return sc.pl.clustermap(
        _get_model_cache(model).factor_adata[:, factor_idx], **kwargs
    )


def scatter(model, x, y, groupby=None, **kwargs):
    x = _normalize_index(x, model.factor_names, as_idx=False)[0]
    y = _normalize_index(y, model.factor_names, as_idx=False)[0]
    kwargs["color"] = groupby
    return sc.pl.scatter(_get_model_cache(model).factor_adata, x, y, **kwargs)
