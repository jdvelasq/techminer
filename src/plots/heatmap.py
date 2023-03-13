import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEXTLEN = 40


def heatmap(X, cmap="Greys", figsize=(6, 6), fontsize=11, **kwargs):
    """Plots a heatmap from a matrix.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         'word 0': [1.00, 0.80, 0.70, 0.00,-0.30],
    ...         'word 1': [0.80, 1.00, 0.70, 0.50, 0.00],
    ...         'word 2': [0.70, 0.70, 1.00, 0.00, 0.00],
    ...         'word 3': [0.00, 0.50, 0.00, 1.00, 0.30],
    ...         'word 4': [-0.30, 0.00, 0.00, 0.30, 1.00],
    ...     },
    ...     index=['word {:d}'.format(i) for i in range(5)]
    ... )
    >>> df
            word 0  word 1  word 2  word 3  word 4
    word 0     1.0     0.8     0.7     0.0    -0.3
    word 1     0.8     1.0     0.7     0.5     0.0
    word 2     0.7     0.7     1.0     0.0     0.0
    word 3     0.0     0.5     0.0     1.0     0.3
    word 4    -0.3     0.0     0.0     0.3     1.0
    >>> fig = heatmap(df)
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotheatmap1.png')

    .. image:: images/plotheatmap1.png
        :width: 400px
        :align: center

    >>> fig = heatmap(df, cmap='Blues')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotheatmap2.png')

    .. image:: images/plotheatmap2.png
        :width: 400px
        :align: center


    >>> df = pd.DataFrame(
    ...     {
    ...         'word 0': [100, 80, 70, 0,30],
    ...         'word 1': [80, 100, 70, 50, 0],
    ...         'word 2': [70, 70, 100, 0, 0],
    ...         'word 3': [0, 50, 0, 100, 3],
    ...         'word 4': [30, 0, 0, 30, 100],
    ...     },
    ...     index=['word {:d}'.format(i) for i in range(5)]
    ... )
    >>> df
            word 0  word 1  word 2  word 3  word 4
    word 0     100      80      70       0      30
    word 1      80     100      70      50       0
    word 2      70      70     100       0       0
    word 3       0      50       0     100      30
    word 4      30       0       0       3     100
    >>> fig = heatmap(df, cmap='Greys')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotheatmap3.png')

    .. image:: images/plotheatmap3.png
        :width: 400px
        :align: center


    >>> df = pd.DataFrame(
    ...     {
    ...         'word 0': [100, 80, 70, 0,30, 1],
    ...         'word 1': [80, 100, 70, 50, 0, 2],
    ...         'word 2': [70, 70, 100, 0, 0, 3],
    ...         'word 3': [0, 50, 0, 100, 3, 4],
    ...         'word 4': [30, 0, 0, 30, 100, 5],
    ...     },
    ...     index=['word {:d}'.format(i) for i in range(6)]
    ... )
    >>> df
            word 0  word 1  word 2  word 3  word 4
    word 0     100      80      70       0      30
    word 1      80     100      70      50       0
    word 2      70      70     100       0       0
    word 3       0      50       0     100      30
    word 4      30       0       0       3     100
    word 5       1       2       3       4       5

    >>> fig = heatmap(df, cmap='Greys')
    >>> fig.savefig('/workspaces/techminer/sphinx/images/plotheatmap3.png')

    .. image:: images/plotheatmap3.png
        :width: 400px
        :align: center

    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    result = ax.pcolor(X.values, cmap=cmap, **kwargs,)
    X.columns = [
        textwrap.shorten(text=w, width=TEXTLEN) if isinstance(w, str) else w
        for w in X.columns
    ]
    X.index = [
        textwrap.shorten(text=w, width=TEXTLEN) if isinstance(w, str) else w
        for w in X.index
    ]

    ax.set_aspect("equal")

    ax.set_xticks(np.arange(len(X.columns)) + 0.5)
    ax.set_xticklabels(X.columns)
    ax.tick_params(axis="x", labelrotation=90)

    ax.set_yticks(np.arange(len(X.index)) + 0.5)
    ax.set_yticklabels(X.index)
    ax.invert_yaxis()

    cmap = plt.cm.get_cmap(cmap)

    if all(X.dtypes == "int64"):
        fmt = "{:3.0f}"
    else:
        fmt = "{:3.2f}"
    for idx_row, row in enumerate(X.columns):
        for idx_col, col in enumerate(X.index):
            if abs(X.loc[col, row]) > X.values.max().max() / 2.0:
                color = cmap(0.0)
            else:
                color = cmap(1.0)
            ax.text(
                idx_row + 0.5,
                idx_col + 0.5,
                fmt.format(X.loc[col, row]),
                ha="center",
                va="center",
                color=color,
            )
    ax.xaxis.tick_top()

    fig.set_tight_layout(True)

    return fig
