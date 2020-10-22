import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import Dashboard, explode

import pandas as pd
import numpy as np

from techminer.plots import worldmap
from techminer.core.filter_records import filter_records


class Model:
    def __init__(self, data):
        self.data = data
        self.column = "Countries"
        self.menu = "worldmap_plot"

    def worldmap_plot(self):
        x = self.data.copy()
        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    self.column,
                    "Num_Documents",
                    "Global_Citations",
                    "ID",
                ]
            ],
            self.column,
        )
        result = x.groupby(self.column, as_index=True).agg(
            {
                "Num_Documents": np.sum,
                "Global_Citations": np.sum,
            }
        )
        top_by = self.top_by.replace(" ", "_")
        return worldmap(
            x=result[top_by],
            figsize=(self.width, self.height),
            cmap=self.colormap,
        )


class App(Dashboard, Model):
    def __init__(self):

        data = filter_records(pd.read_csv("corpus.csv"))

        Model.__init__(self, data=data)

        self.command_panel = [
            dash.HTML("Top by:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.RadioButtons(
                options=[
                    "Num Documents",
                    "Global Citations",
                ],
                description="",
            ),
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
                # Visualization:
                "top_by": self.command_panel[1],
                "colormap": self.command_panel[3],
                "width": self.command_panel[5],
                "height": self.command_panel[6],
            },
        )

        Dashboard.__init__(self)

    def interactive_output(self, **kwargs):
        Dashboard.interactive_output(self, **kwargs)
