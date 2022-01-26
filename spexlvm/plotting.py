"""Collection of plotting functions."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


# source: https://github.com/DTrimarchi10/confusion_matrix
def plot_confusion_matrix(
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


def plot_factor_relevance(
    df, xlabel, ylabel, order_by=None, top_k=20, fig_path=None, **kwargs
):
    if order_by is None:
        order_by = xlabel
    df = df.sort_values([order_by], ascending=True)
    ax = sns.scatterplot(data=df.iloc[-top_k:], x=xlabel, y=ylabel, **kwargs)
    if fig_path is not None:
        title_parts = fig_path.split("/")[-1].split("_")
        if title_parts[-1][-1] == ")":
            title = " ".join(title_parts[2:])
        else:
            title = " ".join([title_part.capitalize() for title_part in title_parts])
        ax.set_title(title)
        save_figure(fig_path)
    return ax
