import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEXTLEN = 40


def gant0_plot(
    x,
    figsize=(8, 8),
    fontsize=12,
    grid_lw=1.0,
    grid_c="gray",
    grid_ls=":",
    *args,
    **kwargs,
):
    """Creates a gant activity plot from a dataframe.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "author 0": [1, 1, 0, 0, 0, 0, 0],
    ...         "author 1": [0, 1, 1, 0, 0, 0, 0],
    ...         "author 2": [1, 0, 0, 0, 0, 0, 0],
    ...         "author 3": [0, 0, 1, 1, 1, 0, 0],
    ...         "author 4": [0, 0, 0, 0, 0, 0, 1],
    ...     },
    ...     index =[2010, 2011, 2012, 2013, 2014, 2015, 2016]
    ... )
    >>> df
          author 0  author 1  author 2  author 3  author 4
    2010         1         0         1         0         0
    2011         1         1         0         0         0
    2012         0         1         0         1         0
    2013         0         0         0         1         0
    2014         0         0         0         1         0
    2015         0         0         0         0         0
    2016         0         0         0         0         1

    >>> fig = gant(df)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/gantplot.png')

    .. image:: images/gantplot.png
        :width: 400px
        :align: center

    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    x = x.copy()
    if "linewidth" not in kwargs.keys() and "lw" not in kwargs.keys():
        kwargs["linewidth"] = 4
    if "marker" not in kwargs.keys():
        kwargs["marker"] = "o"
    if "markersize" not in kwargs.keys() and "ms" not in kwargs.keys():
        kwargs["markersize"] = 8
    if "color" not in kwargs.keys() and "c" not in kwargs.keys():
        kwargs["color"] = "k"
    for idx, col in enumerate(x.columns):
        w = x[col]
        w = w[w > 0]
        ax.plot(w.index, [idx] * len(w.index), **kwargs)

    ax.grid(axis="both", color=grid_c, linestyle=grid_ls, linewidth=grid_lw)

    ax.set_yticks(np.arange(len(x.columns)))
    ax.set_yticklabels(x.columns)
    ax.invert_yaxis()

    years = list(range(min(x.index), max(x.index) + 1))

    ax.set_xticks(years)
    ax.set_xticklabels(years)
    ax.tick_params(axis="x", labelrotation=90)

    for x in ["top", "right", "left"]:
        ax.spines[x].set_visible(False)

    ax.set_aspect("equal")

    return fig
