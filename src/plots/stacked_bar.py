import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEXTLEN = 40


def stacked_bar(
    X,
    cmap="Greys",
    figsize=(6, 6),
    fontsize=11,
    edgecolor="k",
    linewidth=0.5,
    ylabel=None,
    **kwargs,
):
    """Stacked vertical bar plot.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "col 0": [6, 5, 2, 3, 4, 1],
    ...         "col 1": [0, 1, 2, 3, 4, 5],
    ...         "col 2": [3, 2, 3, 1, 0, 1],
    ...     },
    ...     index = "author 0,author 1,author 2,author 3,author 4,author 5".split(","),
    ... )
    >>> df
              col 0  col 1  col 2
    author 0      6      0      3
    author 1      5      1      2
    author 2      2      2      3
    author 3      3      3      1
    author 4      4      4      0
    author 5      1      5      1

    >>> fig = stacked_bar(df, cmap='Blues')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/stkbar0.png')

    .. image:: images/stkbar0.png
        :width: 400px
        :align: center

    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    bottom = X[X.columns[0]].map(lambda w: 0.0)

    for icol, col in enumerate(X.columns):

        kwargs["color"] = cmap((0.3 + 0.50 * icol / (len(X.columns) - 1)))
        ax.bar(
            x=range(len(X)), height=X[col], bottom=bottom, label=col, **kwargs,
        )
        bottom = bottom + X[col]

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.legend()

    ax.grid(axis="y", color="gray", linestyle=":")

    ax.set_xticks(np.arange(len(X)))
    ax.set_xticklabels(X.index)
    ax.tick_params(axis="x", labelrotation=90)

    for x in ["top", "right", "left"]:
        ax.spines[x].set_visible(False)

    fig.set_tight_layout(True)

    return fig
