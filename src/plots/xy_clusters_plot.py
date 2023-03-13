import math

import matplotlib.pyplot as pyplot

from techminer.plots import expand_ax_limits

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "cornflowerblue",
    "lightsalmon",
    "limegreen",
    "tomato",
    "mediumvioletred",
    "darkgoldenrod",
    "lightcoral",
    "silver",
    "darkkhaki",
    "skyblue",
    "dodgerblue",
    "orangered",
    "turquoise",
    "crimson",
    "violet",
    "goldenrod",
    "thistle",
    "grey",
    "yellowgreen",
    "lightcyan",
]

COLORS += COLORS + COLORS

#
# Check in order:
#   - latent semantic analysis module
#
#
def _get_quadrant(x, y, x_axis_at, y_axis_at):
    if x >= x_axis_at and y >= y_axis_at:
        return 0
    if x < x_axis_at and y >= y_axis_at:
        return 1
    if x < x_axis_at and y < y_axis_at:
        return 2
    return 3


def xy_clusters_plot(
    x,
    y,
    x_axis_at,
    y_axis_at,
    labels,
    node_sizes,
    color_scheme,
    xlabel,
    ylabel,
    figsize,
):

    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    ## quadrants
    quadrants = [_get_quadrant(x_, y_, x_axis_at, y_axis_at) for x_, y_ in zip(x, y)]

    ## Select node colors
    node_colors = None
    if color_scheme == "4 Quadrants":
        node_colors = [COLORS[q] for q in quadrants]

    if color_scheme == "Clusters":
        node_colors = [COLORS[i] for i, _ in enumerate(x)]

    if node_colors is None:
        cmap = pyplot.cm.get_cmap(color_scheme)
        max_node_sizes = max(node_sizes)
        min_node_sizes = min(node_sizes)
        if max_node_sizes == min_node_sizes:
            node_colors = [cmap(0.8) for i in node_sizes]
        else:
            node_colors = [
                cmap(
                    0.4
                    + 0.60 * (i - min_node_sizes) / (max_node_sizes - min_node_sizes)
                )
                for i in node_sizes
            ]

    ## plot bubbles
    ax.scatter(
        x,
        y,
        marker="o",
        s=node_sizes,
        c=node_colors,
        alpha=0.4,
        linewidths=3,
    )

    ## plot centers as black dots
    ax.scatter(
        x,
        y,
        marker="o",
        s=50,
        c="k",
        alpha=1.0,
    )

    ## plot node labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    factor = 0.1

    texts = []

    for x_, y_, label, quadrant in zip(x, y, labels, quadrants):

        ha = {
            0: "left",
            1: "right",
            2: "right",
            3: "left",
        }[quadrant]

        va = {
            0: "center",
            1: "center",
            2: "center",
            3: "center",
        }[quadrant]

        delta = factor * (xlim[1] - xlim[0])
        angle = math.atan(math.fabs(y_ / x_))
        radious = math.sqrt(x_ ** 2 + y_ ** 2) + delta
        x_label = math.copysign(radious * math.cos(angle), x_)
        y_label = math.copysign(radious * math.sin(angle), y_)

        texts.append(
            ax.text(
                x_label,
                y_label,
                s=" ".join(label.split(" ")[:-1]),
                fontsize=9,
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    boxstyle="round,pad=0.5",
                    edgecolor="lightgray",
                ),
                horizontalalignment=ha,
                verticalalignment=va,
            )
        )

        ax.plot(
            [x_, x_label],
            [y_, y_label],
            lw=1,
            ls="-",
            c="k",
            zorder=-1,
        )

    texts

    ## limits
    ax.axis("equal")

    ## labels
    ax.text(
        ax.get_xlim()[1],
        y_axis_at,
        s=xlabel,
        fontsize=9,
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    ax.text(
        0.02 + y_axis_at,
        ax.get_ylim()[1],
        s=ylabel,
        fontsize=9,
        horizontalalignment="left",
        verticalalignment="top",
    )

    ## generic

    ax.axhline(
        y=y_axis_at,
        color="gray",
        linestyle="--",
        linewidth=1,
        zorder=-1,
    )
    ax.axvline(
        x=x_axis_at,
        color="gray",
        linestyle="--",
        linewidth=1,
        zorder=-1,
    )

    ax.axis("off")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.set_tight_layout(True)

    return fig
