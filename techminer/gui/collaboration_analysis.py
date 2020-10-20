import numpy as np
import pandas as pd

import techminer.core.dashboard as dash
import ipywidgets as widgets
from techminer.core import DASH
from techminer.core import explode
from techminer.core import exclude_terms, add_counters_to_axis, sort_by_axis

from techminer.plots import (
    stacked_bar,
    stacked_barh,
)

from techminer.core.filter_records import filter_records


class Model:
    def __init__(self, data):
        self.data = data
        self.limit_to = None
        self.exclude = None
        self.top_by = "Num_Documents"

    def compute(self):

        x = self.data.copy()
        x["SD"] = x[self.column].map(
            lambda w: 1 if isinstance(w, str) and len(w.split(";")) == 1 else 0
        )
        x["MD"] = x[self.column].map(
            lambda w: 1 if isinstance(w, str) and len(w.split(";")) > 1 else 0
        )
        x = explode(
            x[
                [
                    self.column,
                    "SD",
                    "MD",
                    "ID",
                ]
            ],
            self.column,
        )
        result = x.groupby(self.column, as_index=False).agg(
            {
                "SD": np.sum,
                "MD": np.sum,
            }
        )
        result["SMR"] = [
            round(MD / max(SD, 1), 2) for SD, MD in zip(result.SD, result.MD)
        ]
        result = result.set_index(self.column)

        ## limit to / exclude options
        result = exclude_terms(data=result, axis=0)

        ## counters in axis names
        result = add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )

        ## Top by / Top N
        result = sort_by_axis(data=result, sort_by=self.top_by, ascending=False, axis=0)
        result = result.head(self.max_items)

        ## Sort by
        if self.sort_by in result.columns:
            result = result.sort_values(self.sort_by, ascending=self.ascending)
        else:
            result = sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )

        if self.view == "Table":
            return result

        if self.view == "Bar plot":
            return stacked_bar(
                X=result[["SD", "MD"]],
                cmap=self.colormap,
                ylabel="Num Documents",
                figsize=(self.width, self.height),
            )

        if self.view == "Horizontal bar plot":
            return stacked_barh(
                X=result[["SD", "MD"]],
                cmap=self.colormap,
                xlabel="Num Documents",
                figsize=(self.width, self.height),
            )


COLUMNS = [
    "Authors",
    "Countries",
    "Institutions",
]


class DASHapp(DASH, Model):
    def __init__(
        self,
    ):
        data = filter_records(pd.read_csv("corpus.csv"))

        Model.__init__(
            self,
            data=data,
        )

        self.menu = "compute"

        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.RadioButtons(
                options=[
                    "Table",
                    "Bar plot",
                    "Horizontal bar plot",
                ],
                description="",
            ),
            dash.HTML("Parameters:"),
            dash.Dropdown(
                description="Column:",
                options=[z for z in COLUMNS if z in self.data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.HTML("Visualization:"),
            dash.Dropdown(
                description="Sort by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Global Citations",
                    "SD",
                    "MR",
                    "SMR",
                ],
            ),
            dash.ascending(),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                # Display:
                "view": self.command_panel[1],
                # Parameters:
                "column": self.command_panel[3],
                "min_occ": self.command_panel[4],
                "max_items": self.command_panel[5],
                # Visualization:
                "sort_by": self.command_panel[7],
                "ascending": self.command_panel[8],
                "colormap": self.command_panel[9],
                "width": self.command_panel[10],
                "height": self.command_panel[11],
            },
        )

        DASH.__init__(self)

        self.interactive_output(
            **{
                # Display:
                "view": self.command_panel[1].value,
                # Parameters:
                "column": self.command_panel[3].value,
                "min_occ": self.command_panel[4].value,
                "max_items": self.command_panel[5].value,
                # Visualization:
                "sort_by": self.command_panel[7].value,
                "ascending": self.command_panel[8].value,
                "colormap": self.command_panel[9].value,
                "width": self.command_panel[10].value,
                "height": self.command_panel[11].value,
            }
        )

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.view == "Table":
            self.set_disabled("Colormap:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")
        else:
            self.set_enabled("Colormap:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")
