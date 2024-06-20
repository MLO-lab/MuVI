"""Collection of plotting functions."""

import logging

from typing import Optional
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


STRIPPLOT = "stripplot"
BOXPLOT = "boxplot"
BOXENPLOT = "boxenplot"
VIOLINPLOT = "violinplot"
GROUP_PL_TYPES = [STRIPPLOT, BOXPLOT, BOXENPLOT, VIOLINPLOT]


def savefig_or_show(
    writekey: str,
    show: Optional[bool] = None,
    dpi: Optional[int] = None,
    ext: Optional[str] = None,
    save: Union[bool, str, None] = None,
):
    return sc.pl._utils.savefig_or_show(writekey, show, dpi, ext, save)


def _subset_df(data, groupby, groups, include_rest=True):
    if groups is None:
        return data

    _groups = groups.copy()

    if include_rest:
        data[groupby] = data[groupby].cat.add_categories(include_rest)
        data.loc[~data[groupby].isin(_groups), groupby] = include_rest
        _groups.append(include_rest)
    data = data.loc[data[groupby].isin(_groups), :].copy()
    data[groupby] = data[groupby].cat.remove_unused_categories()

    if data.empty:
        raise ValueError("Empty data, check whether the provided `groups` are correct.")

    return data


def _setup_legend(
    g,
    bbox_to_anchor=(1, 0.5),
    loc="center left",
    frameon=False,
    remove_last=False,
    fontsize=None,
):
    kwargs = {"bbox_to_anchor": bbox_to_anchor, "loc": loc, "frameon": frameon}

    if remove_last:
        handles, labels = g.get_legend_handles_labels()
        kwargs["handles"] = handles[:-1]
        kwargs["labels"] = labels[:-1]

    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    g.legend(**kwargs)

    return g


def _get_color_dict(factor_adata, groupby, include_rest=True):
    uns_colors_key = f"{groupby}_colors"
    if uns_colors_key not in factor_adata.uns:
        return None
    color_dict = dict(
        zip(
            factor_adata.obs[groupby].astype("category").cat.categories,
            factor_adata.uns[uns_colors_key],
        )
    )
    if include_rest:
        color_dict[include_rest] = "#D3D3D3"
    return color_dict


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


def missingness_overview(
    model,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    data = model.get_observations(as_df=True)

    g = muvi.pl.lined_heatmap(
        pd.DataFrame(
            {
                view_name: data[view_name].isna().mean(axis=1)
                for view_name in model.view_names
            }
        )
        .loc[model.sample_names, model.view_names]
        .T,
        vmin=0.0,
        vmax=1.0,
        cmap="gray",
        **kwargs,
    )
    savefig_or_show("missingness_overview", show=show, save=save)
    if not show:
        return g


def variance_explained(
    model,
    top=50,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    """Heatmap of variance explained, see `muvi.tl.variance_explained`."""

    model_cache = _get_model_cache(model)
    r2_fac = None
    if model_cache.factor_metadata is not None:
        r2_fac = model_cache.factor_metadata.copy()
    r2_cov = None
    if model_cache.cov_metadata is not None:
        r2_cov = model_cache.cov_metadata.copy()

    if r2_fac is None and r2_cov is None:
        raise ValueError(
            "No scores found in model cache, rerun `muvi.tl.variance_explained`."
        )

    r2_columns = [f"{model_cache.METRIC_R2}_{vn}" for vn in model.view_names]
    if r2_fac is not None:
        r2_fac = r2_fac.loc[:, r2_columns].T
    if r2_cov is not None:
        r2_cov = r2_cov.loc[:, r2_columns].T

    if top > 0:
        r2_argsort = r2_fac.sum().argsort()[::-1]
        r2_fac = r2_fac.iloc[:, r2_argsort]
    if top > 1:
        r2_fac = r2_fac.iloc[:, :top]

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

    kwargs["vmin"] = 0

    g = lined_heatmap(r2_joint, figsize=figsize, vlines=vlines, **kwargs)
    savefig_or_show("variance_explained", show=show, save=save)
    if not show:
        return g


def variance_explained_grouped(
    model,
    factor_idx,
    view_idx="all",
    groups=None,
    kind="bar",
    stacked=True,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    model_cache = _get_model_cache(model)
    if model_cache.UNS_GROUPED_R2 not in model_cache.uns:
        raise ValueError(
            f"`{model_cache.UNS_GROUPED_R2}` not found in model cache, "
            "rerun `muvi.tl.variance_explained_grouped`."
        )

    view_idx = _normalize_index(view_idx, model.view_names, as_idx=False)
    factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)

    data = model_cache.uns[model_cache.UNS_GROUPED_R2].copy()
    groupby = data.columns[0]

    data = data[data["Factor"].isin(factor_idx)]

    data[model_cache.METRIC_R2] = data[
        [f"{model_cache.METRIC_R2}_{vn}" for vn in view_idx]
    ].sum(1)

    data = _subset_df(data, groupby, groups, include_rest=False)

    data = data.pivot(
        index="Factor",
        columns=groupby,
        values=model_cache.METRIC_R2,
    )
    data = data.loc[factor_idx]

    legend_fontsize = kwargs.pop("legend_fontsize", None)

    g = data.plot(
        kind=kind,
        stacked=stacked,
        color=kwargs.pop("color", _get_color_dict(model_cache.factor_adata, groupby)),
        **kwargs,
    )
    g.set_ylabel(r"$R^2$")

    g = _setup_legend(g, fontsize=legend_fontsize)

    g.set_title(f"Variance explained across {groupby} groups in {', '.join(view_idx)}")
    g.set(xlabel="Factor")

    savefig_or_show("variance_explained_grouped", show=show, save=save)
    if not show:
        return g


def factors_overview(
    model,
    view_idx="all",
    one_sided=True,
    alpha=0.05,
    sig_only=False,
    prior_only=False,
    adjusted=True,
    top=25,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    """Scatterplot of variance explained along with significance test results,
    see `muvi.tl.test`."""
    if isinstance(view_idx, int):
        view_idx = model.view_names[view_idx]

    view_indices = _normalize_index(view_idx, model.view_names, as_idx=False)
    n_views = len(view_indices)

    model_cache = _get_model_cache(model)

    figsize = (8 * n_views, 8)
    fig, axs = plt.subplots(1, n_views, figsize=figsize, squeeze=False, sharey=False)

    factor_metadata = model_cache.factor_metadata.copy()

    name_col = "Factor"
    factor_metadata[name_col] = factor_metadata.index.astype(str)

    if prior_only:
        factor_metadata = factor_metadata.loc[
            ~factor_metadata[name_col].str.contains("dense", case=False), :
        ].copy()

    for m, view_idx in enumerate(view_indices):
        factor_metadata_view = factor_metadata.copy()
        size_col = None
        if model._informed:
            size_col = "Size"
            factor_metadata_view[size_col] = model.get_prior_masks(
                view_idx, as_df=True
            )[view_idx].sum(1)
            factor_metadata_view.loc[
                factor_metadata_view[name_col].str.contains("dense", case=False),
                size_col,
            ] = 0

        sign_dict = {model_cache.TEST_ALL: " (*)"}
        if one_sided:
            sign_dict = {model_cache.TEST_NEG: " (-)", model_cache.TEST_POS: " (+)"}
        joint_p_col = "p_min"
        direction_col = "direction"

        p_col = model_cache.TEST_P
        if adjusted:
            p_col = model_cache.TEST_P_ADJ

        p_df = factor_metadata_view[
            [f"{p_col}_{sign}_{view_idx}" for sign in sign_dict]
        ]
        if p_df.isna().all(None) and sig_only:
            raise ValueError(
                "No test results found in model cache, rerun `muvi.tl.test`."
            )

        p_df = p_df.fillna(1.0)
        factor_metadata_view[joint_p_col] = p_df.min(axis=1).clip(1e-10, 1.0)
        factor_metadata_view[direction_col] = p_df.idxmin(axis=1).str[
            len(p_col) + 1 : len(p_col) + 4
        ]

        if alpha is None:
            alpha = 1.0
        if alpha <= 0:
            logger.warning("Negative or zero `alpha`, setting `alpha` to 0.01.")
            alpha = 0.01
        if alpha > 1.0:
            logger.warning("`alpha` larger than 1.0, setting `alpha` to 1.0.")
            alpha = 1.0
        factor_metadata_view.loc[
            factor_metadata_view[joint_p_col] > alpha, direction_col
        ] = ""
        if sig_only:
            factor_metadata_view = factor_metadata_view.loc[
                factor_metadata_view[direction_col] != "", :
            ]

        factor_metadata_view[name_col] = factor_metadata_view[
            name_col
        ] + factor_metadata_view[direction_col].map(sign_dict).fillna("")

        neg_log_col = r"$-\log_{10}(FDR)$"
        factor_metadata_view[neg_log_col] = -np.log10(factor_metadata_view[joint_p_col])

        r2_col = f"{model_cache.METRIC_R2}_{view_idx}"
        if top > 0:
            factor_metadata_view = factor_metadata_view.sort_values(
                r2_col, ascending=True
            )

        g = sns.scatterplot(
            ax=axs[0][m],
            data=factor_metadata_view.iloc[-top:],
            x=r2_col,
            y=name_col,
            hue=neg_log_col,
            palette=kwargs.pop("palette", "flare"),
            size=size_col,
            sizes=kwargs.pop("sizes", (50, 350)),
            **kwargs,
        )
        g.set_title(
            rf"Overview top factors in {view_idx} $\alpha = {alpha}$"
            # rf"($\alpha = {alpha:.{max(1, int(-np.log10(alpha)))}f}$)"  # noqa: ERA001
        )
        g.set(xlabel=r"$R^2$")

        # Format the legend labels
        new_labels = []
        for text in g.legend_.get_texts():
            if "size" in text.get_text().lower():
                break
            try:
                new_label = float(text.get_text())
                new_label = f"{new_label:.3f}"
            except ValueError:
                new_label = text.get_text()
            new_labels.append(new_label)

        # Set the new labels
        for text, new_label in zip(g.legend_.get_texts(), new_labels):
            text.set_text(new_label)

    fig.tight_layout()
    savefig_or_show(f"overview_view_{view_idx}", show=show, save=save)
    if not show:
        return fig, axs


def inspect_factor(
    model,
    factor_idx,
    view_idx="all",
    top=25,
    ranked=True,
    figsize=None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    """Scatterplot of factor loadings for specific factors."""
    view_idx = _normalize_index(view_idx, model.view_names, as_idx=False)
    factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)

    n_views = len(view_idx)
    n_factors = len(factor_idx)

    if figsize is None:
        figsize = (8 * n_views, 8 * n_factors)

    fig, axs = plt.subplots(n_factors, n_views, squeeze=False, figsize=figsize)

    for m in range(n_views):
        view_name = view_idx[m]
        for k in range(n_factors):
            factor_name = factor_idx[k]
            i = k * n_views + m
            g = axs[k, m]
            # only last
            show_legend = i == n_views * n_factors - 1

            factor_loadings = model.get_factor_loadings(
                view_name, factor_name, as_df=True
            )[view_name].iloc[0, :]
            factor_mask = pd.Series(False, index=factor_loadings.index)
            if model._informed:
                factor_mask = model.get_prior_masks(view_name, factor_name, as_df=True)[
                    view_name
                ].iloc[0, :]
            else:
                logger.warning(
                    "Model not informed a priori, all features are inferred."
                )
            factor_loadings_abs = np.abs(factor_loadings)
            factor_loadings_rank = factor_loadings.rank(ascending=False)

            name_col = "Feature"
            loading_col = "Loading"
            rank_col = "Rank"
            abs_loading_col = "Loading (abs)"

            data = pd.DataFrame(
                {
                    name_col: model.feature_names[view_name],
                    "Mask": factor_mask,
                    loading_col: factor_loadings,
                    abs_loading_col: factor_loadings_abs,
                    rank_col: factor_loadings_rank,
                    "FP": ~factor_mask & (factor_loadings_abs > 0.0),
                }
            )

            type_col = "Type"
            data[type_col] = data["FP"].map({False: "Annotated", True: "Inferred"})

            if top > 0:
                data = data.sort_values(abs_loading_col, ascending=True)
            x = loading_col
            y = name_col
            kwargs["hue"] = type_col
            kwargs["hue_order"] = ["Annotated", "Inferred"]
            kwargs["palette"] = {"Annotated": "black", "Inferred": "red"}
            if ranked:
                x = rank_col
                y = loading_col
            g = sns.scatterplot(
                ax=g,
                data=data.iloc[-top:],
                x=x,
                y=y,
                s=kwargs.pop("s", (64)),
                legend=show_legend and not ranked,
                **kwargs,
            )
            if ranked:
                g = sns.scatterplot(
                    ax=g,
                    data=data.iloc[:-top],
                    x=rank_col,
                    y=loading_col,
                    s=10,
                    legend=show_legend,
                    **kwargs,
                )

                y_max = factor_loadings.max()
                y_min = factor_loadings.min()
                x_range = factor_loadings_rank.max()

                labeled_data = (
                    data.iloc[-top:].sort_values(loading_col, ascending=False).copy()
                )

                labeled_data["is_positive"] = labeled_data[loading_col] > 0

                n_positive = labeled_data["is_positive"].sum()
                n_negative = top - n_positive
                num = max(n_positive, n_negative)

                labeled_data["x_arrow_pos"] = labeled_data[rank_col] + 0.02 * x_range
                labeled_data["x_text_pos"] = (
                    labeled_data["x_arrow_pos"] + 0.15 * x_range
                )
                labeled_data["y_arrow_pos"] = labeled_data[loading_col]
                labeled_data["y_text_pos"] = (
                    np.linspace(y_max, 0.1 * y_max, num=num)[:n_positive].tolist()
                    + np.linspace(y_min, -0.1 * y_min, num=num)[:n_negative][
                        ::-1
                    ].tolist()
                )

                for _, row in labeled_data.iterrows():
                    g.text(
                        row["x_text_pos"],
                        row["y_text_pos"],
                        row[name_col],
                        color=kwargs["palette"][row[type_col]],
                        fontsize="medium",
                    )
                    g.annotate(
                        "",
                        (row["x_arrow_pos"], row["y_arrow_pos"]),
                        xytext=(row["x_text_pos"], row["y_text_pos"]),
                        # bbox=dict(boxstyle="round", alpha=0.1),  # noqa: ERA001
                        arrowprops={
                            "arrowstyle": "simple,tail_width=0.01,head_width=0.15",
                            "color": "black",
                        },
                    )

            g.set_title(f"{factor_name} ({view_name})")

    fig.tight_layout()
    savefig_or_show("overview_factors", show=show, save=save)
    if not show:
        return fig, axs


# source: https://github.com/DTrimarchi10/confusion_matrix
def confusion_matrix(
    model,
    view_idx=0,
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
        group_labels = [f"{value}\n" for value in group_names]
    else:
        group_labels = blanks

    group_counts = [f"{value:0.0f}\n" for value in cf.flatten()] if count else blanks

    if percent:
        group_percentages = [f"{value:.2%}" for value in cf.flatten() / np.sum(cf)]
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
            stats_text = (
                f"\n\nAccuracy={accuracy:0.3f}"
                + "\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    precision, recall, f1_score
                )
            )
        else:
            stats_text = f"\n\nAccuracy={accuracy:0.3f}"
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


def factor_activity(
    true_w,
    approx_w,
    true_mask,
    noisy_mask,
    factor_idx=0,
    ylim=None,
    top=None,
    **kwargs,
):
    true_w_col = true_w[factor_idx, :]
    w_col = approx_w[factor_idx, :]
    true_mask_col = true_mask[factor_idx, :]
    noisy_mask_col = noisy_mask[factor_idx, :]
    if top is not None:
        # descending order
        argsort_indices = np.argsort(-np.abs(w_col))[:top]
        w_col = w_col[argsort_indices]
        # remove zeros
        non_zero_indices = np.abs(w_col) > 0
        if non_zero_indices.sum() < top:
            logger.warning(
                f"Found {sum(non_zero_indices)} (< {top}) non-zero weights found,"
                " updating the `top` parameter."
            )
        top = min(top, sum(non_zero_indices))
        # subset again
        argsort_indices = argsort_indices[:top]
        w_col = w_col[:top]
        true_w_col = true_w_col[argsort_indices]
        true_mask_col = true_mask_col[argsort_indices]
        noisy_mask_col = noisy_mask_col[argsort_indices]

    activity_df = pd.DataFrame(
        {
            "true_weight": true_w_col,
            "weight": w_col,
            "true_mask": true_mask_col,
            "noisy_mask": noisy_mask_col,
            "TP": true_mask_col & noisy_mask_col,
            "FP": ~true_mask_col & noisy_mask_col,
            "TN": ~true_mask_col & ~noisy_mask_col,
            "FN": true_mask_col & ~noisy_mask_col,
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
        handles=[h for i, h in enumerate(joint_handles) if i not in [0, 5]],
        labels=[h for i, h in enumerate(joint_labels) if i not in [0, 5]],
    )
    if ylim is not None:
        g.set(ylim=ylim)

    return g, activity_df


# scanpy plotting


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
def group(
    model,
    factor_idx,
    groupby,
    groups=None,
    pl_type=HEATMAP,
    **kwargs,
):
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

    factor_adata = _get_model_cache(model).factor_adata.copy()

    return pl_fn(
        factor_adata[
            _subset_df(
                factor_adata.obs.copy(), groupby, groups, include_rest=False
            ).index,
            :,
        ],
        factor_idx,
        groupby,
        **kwargs,
    )


# plot ranked factors against groups of observations
def rank(model, n_factors=10, pl_type=None, sep_groups=True, **kwargs):
    factor_adata = _get_model_cache(model).factor_adata
    if "rank_genes_groups" not in factor_adata.uns:
        raise ValueError("No group-wise ranking found, run `muvi.tl.rank first.`")

    groupby = factor_adata.uns["rank_genes_groups"]["params"]["groupby"]
    dendrogram_key = "dendrogram_" + groupby
    if dendrogram_key not in factor_adata.uns:
        logger.warning(
            f"dendrogram data not found (using `{dendrogram_key}` as key). "
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
    if "groups" in kwargs and kwargs["groups"] is not None:
        n_groups = len(kwargs["groups"])

    positions = np.linspace(
        n_factors, n_factors * n_groups, num=n_groups - 1, endpoint=False
    )

    try:
        pl_fn = type_to_fn[pl_type]
    except KeyError as e:
        raise ValueError(
            f"`{pl_type}` is not valid. Select one of {', '.join(PL_TYPES)}."
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
    if pl_type in (DOTPLOT, STACKED_VIOLIN):
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
    savefig_or_show(writekey, show=show, save=save)
    if not show:
        return g


def clustermap(model, factor_idx="all", **kwargs):
    factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)
    return sc.pl.clustermap(
        _get_model_cache(model).factor_adata[:, factor_idx], **kwargs
    )


def _groupplot(
    model,
    factor_idx,
    groupby,
    pl_type=STRIPPLOT,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    factor_idx = _normalize_index(factor_idx, model.factor_names, as_idx=False)

    model_cache = _get_model_cache(model)
    if groupby not in model_cache.factor_adata.obs.columns:
        raise ValueError(
            f"`{groupby}` not found in the metadata, "
            " add a new column onto `model._cache.factor_adata.obs`."
        )
    data = pd.concat(
        [
            model_cache.factor_adata.to_df().loc[:, factor_idx],
            model_cache.factor_adata.obs[groupby],
        ],
        axis=1,
    )

    data = pd.melt(data, id_vars=[groupby], var_name="Factor", value_name="Score")

    data = _subset_df(data, groupby, groups, include_rest=include_rest)

    if pl_type is None:
        pl_type = STRIPPLOT
    pl_type = pl_type.lower().strip()

    type_to_fn = {
        STRIPPLOT: sns.stripplot,
        BOXPLOT: sns.boxplot,
        BOXENPLOT: sns.boxenplot,
        VIOLINPLOT: sns.violinplot,
    }

    try:
        pl_fn = type_to_fn[pl_type]
    except KeyError as e:
        raise ValueError(
            f"`{pl_type}` is not valid. Select one of {', '.join(GROUP_PL_TYPES)}."
        ) from e

    legend_fontsize = kwargs.pop("legend_fontsize", None)

    g = pl_fn(
        data=data,
        x="Factor",
        y="Score",
        hue=kwargs.pop("hue", groupby),
        palette=kwargs.pop(
            "palette",
            _get_color_dict(
                model_cache.factor_adata, groupby, include_rest=include_rest
            ),
        ),
        **kwargs,
    )
    if rot is not None:
        for label in g.get_xticklabels():
            label.set_rotation(rot)
    g = _setup_legend(
        g, remove_last=groups is not None and include_rest, fontsize=legend_fontsize
    )
    savefig_or_show(pl_type, show=show, save=save)
    if not show:
        return g


def stripplot(
    model,
    factor_idx,
    groupby,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    return _groupplot(
        model,
        factor_idx,
        groupby,
        pl_type=STRIPPLOT,
        groups=groups,
        include_rest=include_rest,
        rot=rot,
        show=show,
        save=save,
        **kwargs,
    )


def boxplot(
    model,
    factor_idx,
    groupby,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    return _groupplot(
        model,
        factor_idx,
        groupby,
        pl_type=BOXPLOT,
        groups=groups,
        include_rest=include_rest,
        rot=rot,
        show=show,
        save=save,
        **kwargs,
    )


def boxenplot(
    model,
    factor_idx,
    groupby,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    return _groupplot(
        model,
        factor_idx,
        groupby,
        pl_type=BOXENPLOT,
        groups=groups,
        include_rest=include_rest,
        rot=rot,
        show=show,
        save=save,
        **kwargs,
    )


def violinplot(
    model,
    factor_idx,
    groupby,
    groups=None,
    include_rest=True,
    rot: int = 45,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    **kwargs,
):
    return _groupplot(
        model,
        factor_idx,
        groupby,
        pl_type=VIOLINPLOT,
        groups=groups,
        include_rest=include_rest,
        rot=rot,
        show=show,
        save=save,
        **kwargs,
    )


def scatter(
    model,
    x,
    y,
    groupby=None,
    groups=None,
    include_rest=True,
    style=None,
    markers=True,
    **kwargs,
):
    x = _normalize_index(x, model.factor_names, as_idx=False)[0]
    y = _normalize_index(y, model.factor_names, as_idx=False)[0]
    kwargs["color"] = groupby
    model_cache = _get_model_cache(model)

    data = pd.concat(
        [model_cache.factor_adata.to_df(), model_cache.factor_adata.obs.copy()], axis=1
    )

    data = _subset_df(data, groupby, groups, include_rest=include_rest)
    palette = kwargs.pop(
        "palette",
        _get_color_dict(model_cache.factor_adata, groupby, include_rest=include_rest),
    )

    if style is None:
        factor_adata = model_cache.factor_adata.copy()
        if not include_rest:
            factor_adata = factor_adata[data.index, :]
        return sc.pl.scatter(
            factor_adata,
            x,
            y,
            groups=groups,
            **kwargs,
        )

    logger.warning(
        "Experimental! "
        "Passing a `style` argument does not rely on `sc.pl.scatter`, "
        "and may lead to undesired results."
    )

    size = kwargs.pop("size", None)
    show = kwargs.pop("show", None)
    save = kwargs.pop("save", None)
    legend_fontsize = kwargs.pop("legend_fontsize", None)

    kwargs = {}

    if size is None:
        size = 120000 / data.shape[0]
    g = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=groupby,
        style=style,
        markers=markers,
        s=size,
        palette=palette,
        linewidth=kwargs.pop("linewidth", 0),
        ax=kwargs.pop("ax", None),
        **kwargs,
    )

    # getting as close as possible to scanpy plotting style
    g = _setup_legend(
        g, remove_last=groups is not None and include_rest, fontsize=legend_fontsize
    )

    g.set_title(groupby)
    savefig_or_show("scatter", show=show, save=save)
    if not show:
        return g


def scatter_rank(model, groups=None, **kwargs):
    factor_adata = _get_model_cache(model).factor_adata
    try:
        groupby = factor_adata.uns["rank_genes_groups"]["params"]["groupby"]
    except KeyError as e:
        raise ValueError(
            "No group-wise ranking found, run `muvi.tl.rank first.`"
        ) from e
    group_df = sc.get.rank_genes_groups_df(factor_adata, group=groups)
    group_df["scores_abs"] = group_df["scores"].abs()

    relevant_factors_dict = {}
    for group in group_df["group"].unique():
        relevant_factors_dict[group] = (
            group_df[group_df["group"] == group]
            .sort_values("scores_abs", ascending=False)
            .iloc[:2]["names"]
            .tolist()
        )

    show = kwargs.pop("show", None)
    save = kwargs.pop("save", None)
    gs = {}

    for group, relevant_factors in relevant_factors_dict.items():
        g = muvi.pl.scatter(
            model,
            *relevant_factors[:2],
            groupby=groupby,
            groups=groups,
            show=False,
            save=False,
            **kwargs,
        )

        g.set_title(f"{groupby} ({group})")
        savefig_or_show(f"scatter_rank_{group}", show=show, save=save)
        gs[group] = g
    if not show:
        return gs


def groupplot_rank(model, groups=None, pl_type=STRIPPLOT, top=1, **kwargs):
    factor_adata = _get_model_cache(model).factor_adata
    try:
        groupby = factor_adata.uns["rank_genes_groups"]["params"]["groupby"]
    except KeyError as e:
        raise ValueError(
            "No group-wise ranking found, run `muvi.tl.rank first.`"
        ) from e
    group_df = sc.get.rank_genes_groups_df(factor_adata, group=groups)
    group_df["scores_abs"] = group_df["scores"].abs()

    relevant_factors = []
    for group in group_df["group"].unique():
        rfs = (
            group_df[group_df["group"] == group]
            .sort_values("scores_abs", ascending=False)
            .iloc[:top]["names"]
        )
        for rf in rfs:
            if rf not in relevant_factors:
                relevant_factors.append(rf)

    show = kwargs.pop("show", None)
    save = kwargs.pop("save", None)

    return relevant_factors, _groupplot(
        model,
        relevant_factors,
        groupby,
        pl_type=pl_type,
        groups=groups,
        show=show,
        save=save,
        **kwargs,
    )
