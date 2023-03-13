import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

TEXTLEN = 35


def collapse_text(w, width=TEXTLEN):
    if not isinstance(w, str):
        return w
    text_begining = " ".join(w.split(" ")[:-1])
    text_ending = w.split(" ")[-1]
    return textwrap.shorten(text=text_begining, width=TEXTLEN) + " " + text_ending


def bubble_plot(
    X,
    darkness=None,
    figsize=(6, 6),
    cmap="Greys",
    grid_lw=1.0,
    grid_c="gray",
    grid_ls=":",
    fontsize=11,
    **kwargs,
):

    """Creates a gant activity plot from a dataframe.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "author 0": [ 1, 2, 3, 4, 5, 6, 7],
    ...         "author 1": [14, 13, 12, 11, 10, 9, 8],
    ...         "author 2": [1, 5, 8, 9, 0, 0, 0],
    ...         "author 3": [0, 0, 1, 1, 1, 0, 0],
    ...         "author 4": [0, 10, 0, 4, 2, 0, 1],
    ...     },
    ...     index =[2010, 2011, 2012, 2013, 2014, 2015, 2016]
    ... )
    >>> df
          author 0  author 1  author 2  author 3  author 4
    2010         1        14         1         0         0
    2011         2        13         5         0        10
    2012         3        12         8         1         0
    2013         4        11         9         1         4
    2014         5        10         0         1         2
    2015         6         9         0         0         0
    2016         7         8         0         0         1

    >>> fig = bubble(df, axis=0, alpha=0.5, rmax=150)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/bubbleplot0.png')

    .. image:: images/bubbleplot0.png
        :width: 400px
        :align: center

    >>> fig = bubble(df, axis=1, alpha=0.5, rmax=150)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/bubbleplot1.png')

    .. image:: images/bubbleplot1.png
        :width: 400px
        :align: center


    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    x = X.copy()

    ##
    ## Text wrap
    ##
    x.columns = [collapse_text(w) for w in x.columns.tolist()]
    x.index = [collapse_text(w) for w in x.index.tolist()]

    size_max = x.max().max()
    size_min = x.min().min()

    if darkness is None:
        darkness = x
    darkness = darkness.loc[:, x.columns]

    color_max = darkness.max().max()
    color_min = darkness.min().min()

    for idx, row in enumerate(x.index.tolist()):

        sizes = [
            150 + 1000 * (w - size_min) / (size_max - size_min) if w != 0 else 0
            for w in x.loc[row, :]
        ]

        colors = [
            cmap(0.2 + 0.8 * (w - color_min) / (color_max - color_min))
            for w in darkness.loc[row, :]
        ]

        #  return range(len(x.columns)), [idx] * len(x.columns)

        ax.scatter(
            list(range(len(x.columns))),
            [idx] * len(x.columns),
            marker="o",
            s=sizes,
            alpha=1.0,
            c=colors,
            edgecolors="k",
            zorder=11,
            #  **kwargs,
        )

    for idx, row in enumerate(x.iterrows()):
        ax.hlines(
            idx, -1, len(x.columns), linewidth=grid_lw, color=grid_c, linestyle=grid_ls,
        )

    for idx, col in enumerate(x.columns):
        ax.vlines(
            idx, -1, len(x.index), linewidth=grid_lw, color=grid_c, linestyle=grid_ls,
        )

    mean_color = 0.5 * (color_min + color_max)
    for idx_col, col in enumerate(x.columns):
        for idx_row, row in enumerate(x.index):

            if x[col][row] != 0:
                if darkness[col][row] >= 0.8 * mean_color:
                    text_color = "w"
                else:
                    text_color = "k"

                ax.text(
                    idx_col,
                    idx_row,
                    "{}".format(x[col][row])
                    if x[col][row].dtype == "int64"
                    else "{:.2f}".format(x[col][row]),
                    va="center",
                    ha="center",
                    zorder=12,
                    color=text_color,
                )

    ax.set_aspect("equal")

    ax.set_xlim(-1, len(x.columns))
    ax.set_ylim(-1, len(x.index) + 1)

    ax.set_xticks(np.arange(len(x.columns)))
    ax.set_xticklabels(x.columns)
    ax.tick_params(axis="x", labelrotation=90)
    ax.xaxis.tick_top()

    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(x.index)))
    ax.set_yticklabels(x.index)

    for x in ["top", "right", "left", "bottom"]:
        ax.spines[x].set_visible(False)

    fig.set_tight_layout(True)

    return fig
