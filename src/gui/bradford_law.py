import matplotlib
import matplotlib.pyplot as plt
from techminer.core import explode
import pandas as pd
import numpy as np

import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display
from techminer.core import Dashboard
import techminer.core.dashboard as dash

from techminer.core.filter_records import filter_records

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(self, data):
        self.data = data

    def bradford_law(self):

        x = self.data.copy()

        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    "Source_title",
                    "Num_Documents",
                    "Global_Citations",
                    "ID",
                ]
            ],
            self.column,
        )
        result = x.groupby("Source_title", as_index=False).agg(
            {
                "Num_Documents": np.sum,
                "Global_Citations": np.sum,
            }
        )
        result["Global_Citations"] = result["Global_Citations"].map(lambda w: int(w))
        result = result.sort_values(
            by=["Num_Documents", "Global_Citations"], ascending=False
        )
        result = result.reset_index()
        result["Cum_Num_Documents"] = result.Num_Documents.cumsum()

        matplotlib.rc("font", size=11)
        fig = plt.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = plt.cm.get_cmap(self.colormap)
        color = cmap(0.6)

        ax.plot(
            list(range(1, len(result) + 1)),
            result.Num_Documents.tolist(),
            linestyle="-",
            linewidth=3,
            color="k",
        )
        ax.fill_between(
            list(range(1, len(result) + 1)),
            result.Num_Documents.tolist(),
            color=color,
            alpha=0.6,
        )

        ##
        ## Compute core sources
        ##
        core_documents = int(len(self.data) / 3)
        core_sources = result[
            result.Cum_Num_Documents <= core_documents
        ].Source_title.head(self.max_items)
        core_sources = core_sources.map(
            lambda w: w[0:27] + " [...]" if len(w) > 30 else w
        )

        ax.fill_between(
            [1, len(core_sources) + 1],
            [ax.get_ylim()[1], ax.get_ylim()[1]],
            color=color,
            alpha=0.2,
        )

        for x in ["top", "right", "left", "bottom"]:
            ax.spines[x].set_visible(False)

        ax.set_xscale("log")

        ax.grid(axis="y", color="gray", linestyle=":")
        ax.grid(axis="x", color="gray", linestyle=":")
        ax.set_ylabel("Num Documnets")

        ax.tick_params(axis="x", labelrotation=90)

        ax.set_xticks(list(range(1, len(core_sources) + 1)))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticklabels(core_sources.tolist())

        fig.set_tight_layout(True)

        return fig


class App(Dashboard, Model):
    def __init__(self):

        data = filter_records(pd.read_csv("corpus.csv"))

        Model.__init__(self, data=data)

        self.column = "Source_title"
        self.menu = "bradford_law"

        self.command_panel = [
            dash.HTML("Max items:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.IntSlider(value=5, min=5, max=25, step=1),
            dash.HTML("Colormap:"),
            dash.cmap(description=None),
            dash.HTML("Figure size:"),
            dash.fig_width(slider=True),
            dash.fig_height(slider=True),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                # Max items:
                "max_items": self.command_panel[1],
                #  Colormap
                "colormap": self.command_panel[3],
                # Figure size
                "width": self.command_panel[5],
                "height": self.command_panel[6],
            },
        )

        Dashboard.__init__(self)

        self.interactive_output(
            **{
                # Visualization:
                "max_items": self.command_panel[1].value,
                "colormap": self.command_panel[3].value,
                "width": self.command_panel[5].value,
                "height": self.command_panel[6].value,
            }
        )

    def interactive_output(self, **kwargs):

        Dashboard.interactive_output(self, **kwargs)
