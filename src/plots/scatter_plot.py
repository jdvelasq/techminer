import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(
    data, cmap="Greys", figsize=(6, 6), fontsize=11, **kwargs,
):
    """Creates a plot from a dataframe.

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
    >>> fig = plot(df)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotplot.png')

    .. image:: images/plotplot.png
        :width: 400px
        :align: center


    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    x = data.copy()
    if "ID" in x.columns:
        x.pop("ID")
        ax.plot(
            range(len(x)), x[x.columns[1]], **kwargs,
        )
        plt.xticks(
            np.arange(len(x[x.columns[0]])), x[x.columns[0]], rotation="vertical"
        )
        ax.set_xlabel(x.columns[0])
        ax.set_ylabel(x.columns[1])
    else:
        colors = [cmap(0.2 + i / (len(x.columns) - 1)) for i in range(len(x.columns))]

        for i, col in enumerate(x.columns):
            kwargs["color"] = colors[i]
            ax.plot(x.index, x[col], label=col, **kwargs)
        ax.legend()

    ax.set_xticks(x.index)
    ax.set_xticklabels(x.index)
    ax.tick_params(axis="x", labelrotation=90)
    # ax.xaxis.tick_top()

    for x in ["top", "right", "left"]:
        ax.spines[x].set_visible(False)

    fig.set_tight_layout(True)

    return fig

