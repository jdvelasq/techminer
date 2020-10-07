import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

TEXTLEN = 40


def bar_plot(
    height,
    darkness=None,
    cmap="Greys",
    figsize=(6, 6),
    fontsize=11,
    edgecolor="k",
    linewidth=0.5,
    zorder=10,
    ylabel=None,
    **kwargs,
):
    """Make a bar plot.

    See https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.axes.Axes.bar.html.

    Args:
        height (pandas.Series): The height(s) of the bars.
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
    >>> fig = bar(height=df['Num_Documents'], darkness=df['Global_Citations'])
    >>> fig.savefig('/workspaces/techminer/sphinx/images/barplot1.png')

    .. image:: images/barplot1.png
        :width: 400px
        :align: center

    """
    darkness = height if darkness is None else darkness

    cmap = plt.cm.get_cmap(cmap)
    kwargs["color"] = [
        cmap(0.1 + 0.90 * (d - min(darkness)) / (max(darkness) - min(darkness)))
        for d in darkness
    ]

    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()

    ax.bar(
        x=range(len(height)),
        height=height,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        **kwargs,
    )

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    xticklabels = height.index
    if xticklabels.dtype != "int64":
        xticklabels = [
            textwrap.shorten(text=text, width=TEXTLEN) for text in xticklabels
        ]

    ax.set_xticks(np.arange(len(height)))
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis="x", labelrotation=90)

    for x in ["top", "right", "left"]:
        ax.spines[x].set_visible(False)

    ax.grid(axis="y", color="gray", linestyle=":")

    fig.set_tight_layout(True)

    return fig
