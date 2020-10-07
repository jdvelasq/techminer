import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from techminer.core.sort_axis import sort_axis
from techminer.plots import expand_ax_limits
from techminer.plots.set_spines_invisible import set_spines_invisible

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


def _get_quadrant(x, y, x_axis_at, y_axis_at):
    if x >= x_axis_at and y >= y_axis_at:
        return 0
    if x < x_axis_at and y >= y_axis_at:
        return 1
    if x < x_axis_at and y < y_axis_at:
        return 2
    return 3


def conceptual_structure_map(coordinates, cluster_labels, top_n, figsize):
    #
    def encircle(x, y, ax, **kw):
        p = np.c_[x, y]
        hull = ConvexHull(p, qhull_options="QJ")
        poly = pyplot.Polygon(p[hull.vertices, :], **kw)
        ax.add_patch(poly)

    ##
    ##  Creates the plot in memory
    ##
    matplotlib.rc("font", size=11)
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()

    ##
    ##  Plot the points of each cluster
    ##
    factor = 0.005
    n_clusters = len(set(cluster_labels))
    for i_cluster in range(n_clusters):

        X = coordinates[cluster_labels == i_cluster]
        X = sort_axis(
            data=X,
            num_documents=True,
            axis=0,
            ascending=False,
        )

        x = X[X.columns[0]]
        y = X[X.columns[1]]
        ax.scatter(
            x,
            y,
            marker="o",
            s=10,
            alpha=0.9,
            c=COLORS[i_cluster],
            # c="k",
        )

        if len(X) > 2:
            encircle(x, y, ax=ax, ec="k", fc=COLORS[i_cluster], alpha=0.2)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        for x_, y_, t in zip(
            x.head(top_n).tolist(), y.head(top_n).tolist(), X.head(top_n).index
        ):

            quadrant = _get_quadrant(x=x_, y=y_, x_axis_at=0, y_axis_at=0)

            delta_x = {
                0: +factor * (xlim[1] - xlim[0]),
                1: -factor * (xlim[1] - xlim[0]),
                2: -factor * (xlim[1] - xlim[0]),
                3: +factor * (xlim[1] - xlim[0]),
            }[quadrant]

            delta_y = {
                0: +factor * (ylim[1] - ylim[0]),
                1: -factor * (ylim[1] - ylim[0]),
                2: -factor * (ylim[1] - ylim[0]),
                3: +factor * (ylim[1] - ylim[0]),
            }[quadrant]

            ha = {
                0: "left",
                1: "right",
                2: "right",
                3: "left",
            }[quadrant]

            va = {
                0: "bottom",
                1: "bottom",
                2: "top",
                3: "top",
            }[quadrant]

            ax.text(
                x_ + delta_x,
                y_ + delta_y,
                s=" ".join(t.split(" ")[:-1]),
                fontsize=10,
                color=COLORS[i_cluster],
                horizontalalignment=ha,
                verticalalignment=va,
            )

    #
    # 3.-- Generic
    #
    ax.axhline(
        y=coordinates[coordinates.columns[1]].mean(),
        color="gray",
        linestyle="--",
        linewidth=0.5,
        zorder=-1,
    )
    ax.axvline(
        x=coordinates[coordinates.columns[1]].mean(),
        color="gray",
        linestyle="--",
        linewidth=0.5,
        zorder=-1,
    )
    ax.axis("off")
    ax.set_aspect("equal")
    set_spines_invisible(ax)
    ax.grid(axis="both", color="lightgray", linestyle="--", linewidth=0.5)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(
        x=xlim[1],
        y=0.01 * (ylim[1] - ylim[0]),
        s="Dim-0",
        fontsize=9,
        color="dimgray",
        horizontalalignment="right",
        verticalalignment="bottom",
    )
    ax.text(
        x=0.01 * (xlim[1] - xlim[0]),
        y=ylim[1],
        s="Dim-1",
        fontsize=9,
        color="dimgray",
        horizontalalignment="left",
        verticalalignment="top",
    )

    fig.set_tight_layout(True)
    return fig
