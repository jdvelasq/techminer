import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEXTLEN = 40


def gant_plot(
    X, cmap="Greys", figsize=(6, 6), fontsize=11, linewsidth=0.5, zorder=10, **kwargs,
):
    """

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

    data = X.copy()
    years = [year for year in range(data.index.min(), data.index.max() + 1)]
    data = data.applymap(lambda w: 1 if w > 0 else 0)
    data = data.applymap(lambda w: int(w))
    matrix1 = data.copy()
    matrix1 = matrix1.cumsum()
    matrix1 = matrix1.applymap(lambda X: True if X > 0 else False)
    matrix2 = data.copy()
    matrix2 = matrix2.sort_index(ascending=False)
    matrix2 = matrix2.cumsum()
    matrix2 = matrix2.applymap(lambda X: True if X > 0 else False)
    matrix2 = matrix2.sort_index(ascending=True)
    result = matrix1.eq(matrix2)
    result = result.applymap(lambda X: 1 if X is True else 0)
    gant_width = result.sum()
    #  gant_width = gant_width.map(lambda w: w - 0.5)
    gant_left = matrix1.applymap(lambda w: 1 - w)
    gant_left = gant_left.sum()
    gant_left = gant_left.map(lambda w: w - 0.5)

    w = pd.DataFrame({"terms": result.columns, "values": gant_width,})

    cmap = plt.cm.get_cmap(cmap)

    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    kwargs["color"] = [
        cmap(0.1 + 0.90 * (v - min(gant_width)) / (max(gant_width) - min(gant_width)))
        for v in gant_width
    ]

    ax.barh(
        y=range(len(data.columns)),
        width=gant_width,
        left=gant_left,
        edgecolor="k",
        linewidth=0.5,
        zorder=10,
        **kwargs,
    )

    xlim = ax.get_xlim()
    ax.set_xlim(left=xlim[0] - 0.5, right=xlim[0] + len(data) + 0.5)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data.index)
    ax.tick_params(axis="x", labelrotation=90)

    ax.invert_yaxis()
    yticklabels = [textwrap.shorten(text=text, width=TEXTLEN) for text in data.columns]
    ax.set_yticks(np.arange(len(data.columns)))
    ax.set_yticklabels(yticklabels)

    for x in ["top", "right", "left", "bottom"]:
        ax.spines[x].set_visible(False)

    ax.grid(axis="both", color="gray", linestyle=":")

    ax.set_aspect("equal")

    fig.set_tight_layout(True)

    return fig
