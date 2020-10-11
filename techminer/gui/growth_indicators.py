import numpy as np
import pandas as pd
import ipywidgets as widgets


import techminer.core.dashboard as dash

#  from techminer.by_term_per_year_analysis import by_year_analysis
from techminer.core import (
    DASH,
    add_counters_to_axis,
    corpus_filter,
    explode,
    limit_to_exclude,
    sort_axis,
    sort_by_axis,
)
from techminer.core.dashboard import max_items, min_occurrence
from techminer.plots import bar_plot, barh_plot, stacked_bar, stacked_barh

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
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

    def apply(self):
        def average_growth_rate():
            #
            #         sum_{i=Y_start}^Y_end  Num_Documents[i] - Num_Documents[i-1]
            #  AGR = --------------------------------------------------------------
            #                          Y_end - Y_start + 1
            #
            #
            result = self.data.copy()
            result = explode(result[["Year", self.column, "ID"]], self.column)
            result["Num_Documents_per_Year"] = 1
            result = result.groupby([self.column, "Year"], as_index=False).agg(
                {"Num_Documents_per_Year": np.size}
            )

            years_AGR = sorted(set(result.Year))[-(self.time_window + 1) :]
            years_AGR = [years_AGR[0], years_AGR[-1]]
            result = result[result.Year.map(lambda w: w in years_AGR)]
            result = pd.pivot_table(
                result,
                columns="Year",
                index=self.column,
                values="Num_Documents_per_Year",
                fill_value=0,
            )
            result["AGR"] = 0.0
            result = result.assign(
                AGR=(result[years_AGR[1]] - result[years_AGR[0]]) / self.time_window
            )
            result.pop(years_AGR[0])
            result.pop(years_AGR[1])
            result.columns = list(result.columns)
            return result

        def average_documents_per_year():
            #
            #         sum_{i=Y_start}^Y_end  Num_Documents[i]
            #  ADY = -----------------------------------------
            #                  Y_end - Y_start + 1
            #
            result = self.data.copy()
            result = explode(result[["Year", self.column, "ID"]], self.column)
            result["Num_Documents_per_Year"] = 1
            result = result.groupby([self.column, "Year"], as_index=False).agg(
                {"Num_Documents_per_Year": np.size}
            )

            years_ADY = sorted(set(result.Year))[-self.time_window :]
            result = result[result.Year.map(lambda w: w in years_ADY)]
            result = result.groupby([self.column], as_index=False).agg(
                {"Num_Documents_per_Year": np.sum}
            )
            result = result.rename(columns={"Num_Documents_per_Year": "ADY"})
            result["ADY"] = result.ADY.map(lambda w: w / self.time_window)
            return result

        def compute_num_documents():

            result = self.data.copy()
            result = explode(result[["Year", self.column, "ID"]], self.column)
            result["Num_Documents_per_Year"] = 1
            result = result.groupby([self.column, "Year"], as_index=False).agg(
                {"Num_Documents_per_Year": np.size}
            )

            years_between = sorted(set(result.Year))[-self.time_window :]
            years_before = sorted(set(result.Year))[0 : -self.time_window]
            between = result[result.Year.map(lambda w: w in years_between)]
            before = result[result.Year.map(lambda w: w in years_before)]
            between = between.groupby([self.column], as_index=False).agg(
                {"Num_Documents_per_Year": np.sum}
            )
            between = between.rename(
                columns={
                    "Num_Documents_per_Year": "Between {}-{}".format(
                        years_between[0], years_between[-1]
                    )
                }
            )
            before = before.groupby([self.column], as_index=False).agg(
                {"Num_Documents_per_Year": np.sum}
            )
            before = before.rename(
                columns={"Num_Documents_per_Year": "Before {}".format(years_between[0])}
            )
            result = pd.merge(before, between, on=self.column)
            return result

        result = average_growth_rate()
        ady = average_documents_per_year()
        result = pd.merge(result, ady, on=self.column)
        result = result.assign(PDLY=round(result.ADY / len(self.data) * 100, 2))
        num_docs = compute_num_documents()
        result = pd.merge(result, num_docs, on=self.column)
        result = result.reset_index(drop=True)
        result = result.set_index(self.column)

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
        result = sort_axis(data=result, num_documents=True, axis=0, ascending=False)
        index = [
            index
            for index in result.index
            if int(index.split(" ")[-1].split(":")[0]) >= self.min_occ
        ]
        result = result.loc[index, :]

        if self.top_by in ["Alphabetic", "Num Documents", "Global Citations"]:
            result = sort_by_axis(
                data=result, sort_by=self.top_by, ascending=False, axis=0
            )
        elif self.top_by == "Average Growth Rate":
            result = result.sort_values(["AGR", "ADY", "PDLY"], ascending=False)
        elif self.top_by == "Average Documents per Year":
            result = result.sort_values(["ADY", "AGR", "PDLY"], ascending=False)
        elif self.top_by == "Perc of Documents in Last Years":
            result = result.sort_values(["PDLY", "ADY", "AGR"], ascending=False)
        elif self.top_by == "Before":
            result = result.sort_values(
                [result.columns[-2], result.columns[-1]], ascending=False
            )
        elif self.top_by == "Between":
            result = result.sort_values(
                [result.columns[-1], result.columns[-2]], ascending=False
            )
        else:
            pass

        result = result.head(self.max_items)

        if self.sort_by in ["Alphabetic", "Num Documents", "Global Citations"]:
            result = sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )
        else:

            if isinstance(self.sort_by, str):
                sort_by = self.sort_by.replace(" ", "_")
                sort_by = {
                    "Average_Growth_Rate": 3,
                    "Average_Documents_per_Year": 4,
                    "Percentage_of_Documents_in_Last_Years": 5,
                    "Before": 6,
                    "Between": 7,
                }[sort_by]

            if sort_by == 3:
                result = result.sort_values(
                    ["AGR", "ADY", "PDLY"], ascending=self.ascending
                )

            if sort_by == 4:
                result = result.sort_values(
                    ["ADY", "AGR", "PDLY"], ascending=self.ascending
                )

            if sort_by == 5:
                result = result.sort_values(
                    ["PDLY", "ADY", "AGR"], ascending=self.ascending
                )

            if sort_by == 6:
                result = result.sort_values(
                    [result.columns[-2], result.columns[-1]], ascending=self.ascending
                )

            if sort_by == 7:
                result = result.sort_values(
                    [result.columns[-1], result.columns[-2]], ascending=self.ascending
                )

        self.X_ = result

    def table(self):
        self.apply()
        return self.X_

    def average_growth_rate(self):
        self.apply()
        if self.plot == "bar":
            return bar_plot(
                height=self.X_.AGR,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )
        if self.plot == "barh":
            return barh_plot(
                width=self.X_.AGR, cmap=self.colormap, figsize=(self.width, self.height)
            )

    def average_documents_per_year(self):
        self.apply()
        if self.plot == "bar":
            return bar_plot(
                height=self.X_.ADY,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )
        if self.plot == "barh":
            return barh_plot(
                width=self.X_.ADY, cmap=self.colormap, figsize=(self.width, self.height)
            )

    def perc_of_documents_in_last_years(self):
        self.apply()
        if self.plot == "bar":
            return bar_plot(
                height=self.X_.PDLY,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )
        if self.plot == "barh":
            return barh_plot(
                width=self.X_.PDLY,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )

    def num_documents(self):
        self.apply()
        if self.plot == "bar":
            return stacked_bar(
                self.X_[[self.X_.columns[-2], self.X_.columns[-1]]],
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )
        if self.plot == "barh":
            return stacked_barh(
                self.X_[[self.X_.columns[-2], self.X_.columns[-1]]],
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )


###############################################################################
##
##  DASHBOARD
##
###############################################################################

COLUMNS = sorted(
    [
        "Abstract_words_CL",
        "Abstract_words",
        "Affiliations",
        "Author_Keywords_CL",
        "Author_Keywords",
        "Authors",
        "Countries",
        "Index_Keywords_CL",
        "Index_Keywords",
        "Institutions",
        "Keywords_CL",
        "Title_words_CL",
        "Title_words",
    ]
)


class DASHapp(DASH, Model):
    def __init__(
        self,
        limit_to=None,
        exclude=None,
        years_range=None,
    ):
        data = pd.read_csv("corpus.csv")

        Model.__init__(
            self,
            data=data,
            limit_to=limit_to,
            exclude=exclude,
            years_range=years_range,
        )

        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.RadioButtons(
                options=[
                    "Table",
                    "Average Growth Rate",
                    "Average Documents per Year",
                    "Perc of Documents in Last Years",
                    "Num Documents",
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
            dash.Dropdown(
                description="Time window:",
                options=[2, 3, 4, 5],
            ),
            dash.HTML("Visualization:"),
            dash.Dropdown(
                description="Top by:",
                options=[
                    "Num Documents",
                    "Global Citations",
                    "Average Growth Rate",
                    "Average Documents per Year",
                    "Perc of Documents in Last Years",
                    "Number of Document Published",
                    "Before",
                    "Between",
                ],
            ),
            dash.Dropdown(
                description="Sort by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Global Citations",
                    "Average Growth Rate",
                    "Average Documents per Year",
                    "Perc of Documents in Last Years",
                    "Before",
                    "Between",
                ],
            ),
            dash.ascending(),
            dash.Dropdown(
                description="Plot:",
                options=["bar", "barh"],
            ),
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
                "menu": self.command_panel[1],
                # Parameters:
                "column": self.command_panel[3],
                "min_occ": self.command_panel[4],
                "max_items": self.command_panel[5],
                "time_window": self.command_panel[6],
                # Visualization:
                "top_by": self.command_panel[8],
                "sort_by": self.command_panel[9],
                "ascending": self.command_panel[10],
                "plot": self.command_panel[11],
                "colormap": self.command_panel[12],
                "width": self.command_panel[13],
                "height": self.command_panel[14],
            },
        )

        DASH.__init__(self)

        self.interactive_output(
            **{
                "menu": self.command_panel[1].value,
                # Parameters:
                "column": self.command_panel[3].value,
                "min_occ": self.command_panel[4].value,
                "max_items": self.command_panel[5].value,
                "time_window": self.command_panel[6].value,
                # Visualization:
                "top_by": self.command_panel[8].value,
                "sort_by": self.command_panel[9].value,
                "ascending": self.command_panel[10].value,
                "plot": self.command_panel[11].value,
                "colormap": self.command_panel[12].value,
                "width": self.command_panel[13].value,
                "height": self.command_panel[14].value,
            }
        )

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == "Table":
            self.set_disabled("Plot:")
            self.set_disabled("Colormap:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")
        else:
            self.set_enabled("Plot:")
            self.set_enabled("Colormap:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")
