"""Collection of plotting functions."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()
sns.set_style("whitegrid")


def save_figure(file_path, fmt="png", close_fig=False):
    """Save the last figure that was plotted."""
    # for fmt in ["pdf", "pgf", "png"]:
    plt.savefig(
        file_path + "." + fmt,
        dpi=300,
        format=fmt,
        # transparent=True,
        facecolor=(1, 1, 1, 0),
        bbox_inches="tight",
    )
    if close_fig:
        plt.close()


def heatmap(data, figsize=(20, 10), annot=True, **kwargs):
    """Generate a heatmap of a matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    return sns.heatmap(data, annot=annot, ax=ax, **kwargs)


def lined_heatmap(
    data, figsize=(20, 2), annot=False, hlines=None, vlines=None, **kwargs
):
    ax = heatmap(data, figsize, annot, **kwargs)
    if hlines is not None:
        ax.hlines(hlines, *ax.get_xlim(), linestyles="dashed")
    if vlines is not None:
        ax.vlines(vlines, *ax.get_ylim(), linestyles="dashed")
    return ax


# source: https://github.com/DTrimarchi10/confusion_matrix
def confusion_matrix(
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="Blues",
    title=None,
):
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
    sns.heatmap(
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

    if title:
        plt.title(title)

    return accuracy, precision, recall, f1_score
