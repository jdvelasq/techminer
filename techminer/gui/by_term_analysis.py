"""
Analysis by Term
==========================================================================

"""
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from nltk.corpus.reader.ieer import documents

import techminer.core.dashboard as dash
from techminer.core import (
    Dashboard,
    add_counters_to_axis,
    corpus_filter,
    explode,
    exclude_terms,
    sort_axis,
    sort_by_axis,
)
from techminer.plots import (
    bar_plot,
    barh_plot,
    pie_plot,
    stacked_bar,
    stacked_barh,
    treemap,
    wordcloud_,
    worldmap,
)

#  from techminer.core.dashboard import max_items, min_occurrence

from techminer.core.filter_records import filter_records

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(
        self,
        data,
        limit_to,
        exclude,
        years_range,
    ):

        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

        self.column = None
        self.top_by = None
        self.sort_by = None
        self.ascending = None
        self.colormap = None
        self.height = None
        self.width = None
        self.view = None

    def compute_general_table(self):

        x = self.data.copy()

        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    self.column,
                    "Num_Documents",
                    "Global_Citations",
                    "Local_Citations",
                    "ID",
                ]
            ],
            self.column,
        )
        result = x.groupby(self.column, as_index=True).agg(
            {
                "Num_Documents": np.sum,
                "Global_Citations": np.sum,
                "Local_Citations": np.sum,
            }
        )
        result["Global_Citations"] = result["Global_Citations"].map(lambda w: int(w))
        result["Local_Citations"] = result["Local_Citations"].map(lambda w: int(w))

        ##
        ## Exclude items
        ##
        result = exclude_terms(data=result, axis=0)

        ##
        ## Minimal occurrence
        ##
        result = result[result.Num_Documents >= self.min_occ]

        ##
        ## Counters
        ##
        result = add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )

        ##
        ## Top by
        ##
        columns = {
            "Num Documents": ["Num_Documents", "Global_Citations", "Local_Citations"],
            "Global Citations": [
                "Global_Citations",
                "Num_Documents",
                "Local_Citations",
            ],
            "Local Citations": ["Local_Citations", "Global_Citations", "Num_Documents"],
        }[self.top_by]
        result = result.sort_values(by=columns, ascending=False)

        # top_by = self.top_by.replace(" ", "_").replace("-", "_").replace("/", "_")
        # result = sort_axis(
        #     data=result,
        #     num_documents=(top_by == "Num_Documents"),
        #     axis=0,
        #     ascending=False,
        # )

        result = result.head(self.max_items)

        ##
        ## Sort by
        ##
        # result = sort_axis(
        #      data=result,
        #      num_documents=(self.sort_by == "Num Documents"),
        #      axis=0,
        #      ascending=self.ascending,
        #  )
        result = result.sort_values(by=columns, ascending=self.ascending)

        return result

    def compute(self):

        result = self.compute_general_table()

        if self.view == "Table":
            return result

        if self.top_by == "Num Documents":
            values = result.Num_Documents
            darkness = result.Global_Citations
        elif self.top_by == "Global Citations":
            values = result.Global_Citations
            darkness = result.Num_Documents

        elif self.top_by == "Local Citations":
            values = result.Local_Citations
            darkness = result.Num_Documents
        else:
            values = None
            darkness = None

        if self.view == "Bar plot":
            return bar_plot(
                height=values,
                darkness=darkness,
                cmap=self.colormap,
                ylabel=self.top_by,
                figsize=(self.width, self.height),
            )

        if self.view == "Horizontal bar plot":
            return barh_plot(
                width=values,
                darkness=darkness,
                cmap=self.colormap,
                xlabel=self.top_by,
                figsize=(self.width, self.height),
            )
        if self.view == "Pie plot":
            return pie_plot(
                x=values,
                darkness=darkness,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )

        if self.view == "Wordcloud":
            ## remueve num_documents:global_citations from terms
            values.index = [" ".join(term.split(" ")[:-1]) for term in values.index]
            darkness.index = [" ".join(term.split(" ")[:-1]) for term in darkness.index]
            return wordcloud_(
                x=values,
                darkness=darkness,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )

        if self.view == "Treemap":
            return treemap(
                x=values,
                darkness=darkness,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )

    def list_of_core_source_titles(self):

        x = self.data.copy()
        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    "Source_title",
                    "Num_Documents",
                    "ID",
                ]
            ],
            "Source_title",
        )
        m = x.groupby("Source_title", as_index=True).agg(
            {
                "Num_Documents": np.sum,
            }
        )
        m = m[["Num_Documents"]]
        m = m.sort_values(by="Num_Documents", ascending=False)
        m["Cum_Num_Documents"] = m.Num_Documents.cumsum()
        m = m[m.Cum_Num_Documents <= int(len(self.data) / 3)]
        HTML = "1 st. Bradford' Group<br>"
        for value in m.Num_Documents.unique():
            n = m[m.Num_Documents == value]
            HTML += "======================================================<br>"
            HTML += "Num Documents Published:" + str(value) + "<br>"
            HTML += "<br>"
            for source in n.index:
                HTML += "    " + source + "<br>"
            HTML += "<br><br>"
        return widgets.HTML("<pre>" + HTML + "</pre>")

    def limit_to_python_code(self):

        result = self.compute_general_table()
        items = result.index.tolist()
        items = [" ".join(item.split(" ")[:-1]) for item in items]
        HTML = "LIMIT_TO = {<br>"
        HTML += '    "' + self.column + '": [<br>'
        for item in sorted(items):
            HTML += '        "' + item + '",<br>'
        HTML += "    ]<br>"
        HTML += "}<br>"
        return widgets.HTML("<pre>" + HTML + "</pre>")


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class App(Dashboard, Model):
    def __init__(
        self,
        limit_to=None,
        exclude=None,
        years_range=None,
    ):
        """Dashboard app"""

        data = filter_records(pd.read_csv("corpus.csv"))
        self.menu = "compute"

        Model.__init__(
            self,
            data=data,
            limit_to=limit_to,
            exclude=exclude,
            years_range=years_range,
        )

        COLUMNS = sorted([column for column in data.columns])

        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.RadioButtons(
                options=[
                    "Table",
                    "Bar plot",
                    "Horizontal bar plot",
                    "Pie plot",
                    "Wordcloud",
                    "Treemap",
                ],
                description="",
            ),
            dash.HTML("Parameters:"),
            dash.Dropdown(
                description="Column:",
                options=sorted(data.columns),
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.HTML("Visualization:"),
            dash.Dropdown(
                description="Top by:",
                options=[
                    "Num Documents",
                    "Global Citations",
                    "Local Citations",
                ],
            ),
            dash.Dropdown(
                description="Sort by:",
                options=[
                    "Num Documents",
                    "Global Citations",
                    "Local Citations",
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

        Dashboard.__init__(self)
