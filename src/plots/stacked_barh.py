import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEXTLEN = 40


def stacked_barh(
    X, figsize=(6, 6), height=0.8, fontsize=11, cmap="Greys", xlabel=None, **kwargs,
):
    """Stacked horzontal bar plot.

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
    >>> fig = stacked_barh(df, cmap='Blues')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/stkbarh1.png')

    .. image:: images/stkbarh1.png
        :width: 400px
        :align: center

    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    left = X[X.columns[0]].map(lambda w: 0.0)

    for icol, col in enumerate(X.columns):

        kwargs["color"] = cmap((0.3 + 0.50 * icol / (len(X.columns) - 1)))
        ax.barh(
            y=range(len(X)),
            width=X[col],
            height=height,
            left=left,
            label=col,
            **kwargs,
        )
        left = left + X[col]

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.legend()

    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(X[X.columns[0]])))
    ax.set_yticklabels(X.index)

    for x in ["top", "right", "bottom"]:
        ax.spines[x].set_visible(False)

    ax.grid(axis="x", color="gray", linestyle=":")

    return fig

