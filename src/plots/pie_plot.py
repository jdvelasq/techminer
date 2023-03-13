import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEXTLEN = 40


def pie_plot(
    x,
    darkness=None,
    cmap="Greys",
    figsize=(6, 6),
    fontsize=11,
    wedgeprops={
        "width": 0.6,
        "edgecolor": "k",
        "linewidth": 0.5,
        "linestyle": "-",
        "antialiased": True,
    },
    **kwargs,
):
    """Plot a pie chart.

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
    >>> fig = pie(x=df['Num_Documents'], darkness=df['Global_Citations'], cmap="Blues")
    >>> fig.savefig('/workspaces/techminer/sphinx/images/pieplot.png')

    .. image:: images/pieplot.png
        :width: 400px
        :align: center


    """
    darkness = x if darkness is None else darkness

    cmap = plt.cm.get_cmap(cmap)
    colors = [
        cmap(0.1 + 0.90 * (d - min(darkness)) / (max(darkness) - min(darkness)))
        for d in darkness
    ]

    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.pie(
        x=x, labels=x.index, colors=colors, wedgeprops=wedgeprops, **kwargs,
    )

    fig.set_tight_layout(True)

    return fig

