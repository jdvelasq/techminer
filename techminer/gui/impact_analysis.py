import numpy as np
import pandas as pd

import techminer.core.dashboard as dash
import ipywidgets as widgets
from techminer.core import DASH
from techminer.core import explode
from techminer.core import (
    limit_to_exclude,
    add_counters_to_axis,
    sort_by_axis,
    sort_axis,
)

from techminer.plots import (
    bar_plot,
    barh_plot,
)


class Model:
    def __init__(self, data):
        self.data = data
        self.limit_to = None
        self.exclude = None

    def compute(self):

        x = self.data.copy()

        x["Frac_Num_Documents"] = x.Frac_Num_Documents.map(
            lambda w: round(w, 2), na_action="ignore"
        )

        last_year = x.Year.max()
        x["Num_Documents"] = 1
        x["First_Year"] = x.Year
        if self.column == "Authors":
            x = explode(
                x[
                    [
                        self.column,
                        "Frac_Num_Documents",
                        "Num_Documents",
                        "Global_Citations",
                        "First_Year",
                        "ID",
                    ]
                ],
                self.column,
            )
            result = x.groupby(self.column, as_index=False).agg(
                {
                    "Frac_Num_Documents": np.sum,
                    "Num_Documents": np.sum,
                    "Global_Citations": np.sum,
                    "First_Year": np.min,
                }
            )
        else:
            x = explode(
                x[
                    [
                        self.column,
                        "Num_Documents",
                        "Global_Citations",
                        "First_Year",
                        "ID",
                    ]
                ],
                self.column,
            )
            result = x.groupby(self.column, as_index=False).agg(
                {
                    "Num_Documents": np.sum,
                    "Global_Citations": np.sum,
                    "First_Year": np.min,
                }
            )

        result["Last_Year"] = last_year
        result = result.assign(Years=result.Last_Year - result.First_Year + 1)
        result = result.assign(
            Global_Citations_per_Year=result.Global_Citations / result.Years
        )
        result["Global_Citations_per_Year"] = result["Global_Citations_per_Year"].map(
            lambda w: round(w, 2)
        )
        result = result.assign(
            Avg_Global_Citations=result.Global_Citations / result.Num_Documents
        )
        result["Avg_Global_Citations"] = result["Avg_Global_Citations"].map(
            lambda w: round(w, 2)
        )

        result["Global_Citations"] = result["Global_Citations"].map(lambda x: int(x))

        #
        # Indice H
        #
        z = x[[self.column, "Global_Citations", "ID"]].copy()
        z = (
            x.assign(
                rn=x.sort_values("Global_Citations", ascending=False)
                .groupby(self.column)
                .cumcount()
                + 1
            )
        ).sort_values(
            [self.column, "Global_Citations", "rn"], ascending=[False, False, True]
        )
        z["rn2"] = z.rn.map(lambda w: w * w)

        q = z.query("Global_Citations >= rn")
        q = q.groupby(self.column, as_index=False).agg({"rn": np.max})
        h_dict = {key: value for key, value in zip(q[self.column], q.rn)}

        result["H_index"] = result[self.column].map(
            lambda w: h_dict[w] if w in h_dict.keys() else 0
        )

        #
        # indice M
        #
        result = result.assign(M_index=result.H_index / result.Years)
        result["M_index"] = result["M_index"].map(lambda w: round(w, 2))

        #
        # indice G
        #
        q = z.query("Global_Citations >= rn2")
        q = q.groupby(self.column, as_index=False).agg({"rn": np.max})
        h_dict = {key: value for key, value in zip(q[self.column], q.rn)}
        result["G_index"] = result[self.column].map(
            lambda w: h_dict[w] if w in h_dict.keys() else 0
        )

        ## counters in axis names
        result.index = result[self.column]

        ## limit to / exclude options
        result = limit_to_exclude(
            data=result,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        result = add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )

        ## Top by / Top N
        top_by = self.top_by.replace(" ", "_").replace("-", "_").replace("/", "_")
        if top_by in ["Num_Documents", "Global_Citations"]:
            result = sort_axis(
                data=result,
                num_documents=(top_by == "Num_Documents"),
                axis=0,
                ascending=False,
            )
        else:
            result = result.sort_values(top_by, ascending=False)
        result = result.head(self.max_items)

        ## Sort by
        sort_by = self.sort_by.replace(" ", "_").replace("-", "_").replace("/", "_")
        if sort_by in ["Alphabetic", "Num_Documents", "Global_Citations"]:
            result = sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )
        else:
            result = result.sort_values(sort_by, ascending=self.ascending)

        if self.view == "Table":
            result.pop(self.column)
            result.pop("Num_Documents")
            result.pop("Global_Citations")
            result.pop("First_Year")
            result.pop("Last_Year")
            result.pop("Years")
            return result

        if self.view == "Bar plot":
            top_by = self.top_by.replace(" ", "_")
            return bar_plot(
                height=result[top_by],
                cmap=self.colormap,
                ylabel=self.top_by,
                figsize=(self.width, self.height),
            )

        if self.view == "Horizontal bar plot":
            top_by = self.top_by.replace(" ", "_")
            return barh_plot(
                width=result[top_by],
                cmap=self.colormap,
                xlabel=self.top_by,
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
        data = pd.read_csv("corpus.csv")

        Model.__init__(self, data=data)

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
                description="Top by:",
                options=[
                    "Num Documents",
                    "Global Citations",
                    "Global Citations per Year",
                    "Avg Global Citations",
                    "H index",
                    "M index",
                    "G index",
                ],
            ),
            dash.Dropdown(
                description="Sort by:",
                options=[
                    "Num Documents",
                    "Global Citations",
                    "Global Citations per Year",
                    "Avg Global Citations",
                    "H index",
                    "M index",
                    "G index",
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
                "top_by": self.command_panel[7],
                "sort_by": self.command_panel[8],
                "ascending": self.command_panel[9],
                "colormap": self.command_panel[10],
                "width": self.command_panel[11],
                "height": self.command_panel[12],
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
                "top_by": self.command_panel[7].value,
                "sort_by": self.command_panel[8].value,
                "ascending": self.command_panel[9].value,
                "colormap": self.command_panel[10].value,
                "width": self.command_panel[11].value,
                "height": self.command_panel[12].value,
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
