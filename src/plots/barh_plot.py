import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from techminer.plots.shorten_ticklabels import shorten_ticklabels

TEXTLEN = 40


def barh_plot(
    width,
    darkness=None,
    cmap="Greys",
    figsize=(6, 6),
    fontsize=11,
    edgecolor="k",
    linewidth=0.5,
    xlabel=None,
    zorder=10,
    **kwargs,
):
    """Make a horizontal bar plot.

    See https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.barh.html

    Args:
        width (pandas.Series): The widths of the bars.
        darkness (pandas.Series, optional): The color darkness of the bars. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "Greys".
        figsize (tuple, optional): Figure size passed to matplotlib. Defaults to (6, 6).
        fontsize (int, optional): Font size. Defaults to 11.
        edgecolor (str, optional): The colors of the bar edges. Defaults to "k".
        linewidth (float, optional): Width of the bar edges. If 0, don't draw edges. Defaults to 0.5.
        zorder (int, optional): order of drawing. Defaults to 10.

    Returns:
        container: Container with all the bars and optionally errorbars.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "Num_Documents": [3, 2, 2, 1],
    ...         "Global_Citations": [1, 2, 3, 4],
    ...     },
    ...     index="author 3,author 1,author 0,author 2".split(","),
    ... )
    >>> df
              Num_Documents  Global_Citations
    author 3              3            1
    author 1              2            2
    author 0              2            3
    author 2              1            4
    >>> fig = barh(width=df['Num_Documents'], darkness=df['Global_Citations'])
    >>> fig.savefig('/workspaces/techminer/sphinx/images/barhplot.png')

    .. image:: images/barhplot.png
        :width: 400px
        :align: center

    """
    darkness = width if darkness is None else darkness
    cmap = plt.cm.get_cmap(cmap)
    if max(darkness) == min(darkness):
        kwargs["color"] = [cmap(0.1) for _ in darkness]
    else:
        kwargs["color"] = [
            cmap(0.1 + 0.90 * (d - min(darkness)) / (max(darkness) - min(darkness)))
            for d in darkness
        ]

    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.barh(
        y=range(len(width)),
        width=width,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    yticklabels = width.index
    if yticklabels.dtype != "int64":

        yticklabels = shorten_ticklabels(yticklabels)

    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(width)))
    ax.set_yticklabels(yticklabels)

    for x in ["top", "right", "bottom"]:
        ax.spines[x].set_visible(False)

    ax.grid(axis="x", color="gray", linestyle=":")

    fig.set_tight_layout(True)

    return fig
