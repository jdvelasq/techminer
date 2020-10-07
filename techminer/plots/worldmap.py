import json
from os.path import dirname, join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEXTLEN = 40


def worldmap(
    x, cmap="Pastel2", figsize=(6, 6), legend=True, fontsize=11, *args, **kwargs,
):

    """Worldmap plot with the number of documents per country.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> x = pd.Series(
    ...    data = [1000, 900, 800, 700, 600, 1000],
    ...    index = ["China", "Taiwan", "United States", "United Kingdom", "India", "Colombia"],
    ... )
    >>> x
    China             1000
    Taiwan             900
    United States      800
    United Kingdom     700
    India              600
    Colombia          1000
    dtype: int64

    >>> fig = worldmap(x, figsize=(15, 6))
    >>> fig.savefig('/workspaces/techminer/sphinx/images/worldmap.png')

    .. image:: images/worldmap.png
        :width: 2000px
        :align: center


    """
    matplotlib.rc("font", size=fontsize)
    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    df = x.to_frame()

    df["color"] = x.map(lambda w: 0.1 + 0.9 * (w - x.min()) / (x.max() - x.min()))

    module_path = dirname(__file__)
    with open(join(module_path, "../data/worldmap.data"), "r") as f:
        countries = json.load(f)

    #
    #  country_names_in_table = df.index.tolist()
    # country_names_in_map = [c.lower() for c in countries.keys()]

    #  for country in country_names_in_table:
    #      if country not in country_names_in_map:
    #          print(country)
    # print(sorted(country_names))
    #

    for country in countries.keys():
        data = countries[country]
        for item in data:
            ax.plot(item[0], item[1], "-k", linewidth=0.5)
            if country.lower() in df.index.tolist():
                ax.fill(item[0], item[1], color=cmap(df.color[country.lower()]))
    #
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xleft = xmax - 0.02 * (xmax - xmin)
    xright = xmax
    xbar = np.linspace(xleft, xright, 10)
    ybar = np.linspace(ymin, ymin + (ymax - ymin), 100)
    xv, yv = np.meshgrid(xbar, ybar)
    z = yv / (ymax - ymin) - ymin
    ax.pcolormesh(xv, yv, z, cmap=cmap)
    #
    pos = np.linspace(ymin, ymin + (ymax - ymin), 11)
    value = [round(x.min() + (x.max() - x.min()) * i / 10, 0) for i in range(11)]
    for i in range(11):
        ax.text(
            xright + 0.4 * (xright - xleft),
            pos[i],
            str(int(value[i])),
            ha="left",
            va="center",
        )

    ax.plot(
        [xleft - 0.1 * (xright - xleft), xleft - 0.1 * (xright - xleft)],
        [ymin, ymax],
        color="gray",
        linewidth=1,
    )
    for i in range(11):
        ax.plot(
            [xleft - 0.0 * (xright - xleft), xright],
            [pos[i], pos[i]],
            linewidth=2.0,
            color=cmap((11 - i) / 11),
        )

    ax.set_aspect("equal")
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines["bottom"].set_color("gray")
    ax.spines["top"].set_color("gray")
    ax.spines["right"].set_color("gray")
    ax.spines["left"].set_color("gray")

    fig.set_tight_layout(True)

    return fig

