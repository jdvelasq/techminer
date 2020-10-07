import numpy as np
import pandas as pd
from pandas.core.base import NoNewAttributesMixin

import techminer.core.dashboard as dash
from techminer.core import DASH, corpus_filter
from techminer.plots import bar_plot, barh_plot

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(
        self,
        data,
        years_range,
    ):
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        #
        self.ascending = True
        self.colormap = None
        self.height = None
        self.plot = None
        self.sort_by = None
        self.width = None

    def apply(self):
        ##
        data = self.data[["Year", "Global_Citations", "ID"]].explode("Year")
        data["Num_Documents"] = 1
        result = data.groupby("Year", as_index=False).agg(
            {"Global_Citations": np.sum, "Num_Documents": np.size}
        )
        result = result.assign(
            ID=data.groupby("Year").agg({"ID": list}).reset_index()["ID"]
        )
        result["Global_Citations"] = result["Global_Citations"].map(lambda w: int(w))
        years = [year for year in range(result.Year.min(), result.Year.max() + 1)]
        result = result.set_index("Year")
        result = result.reindex(years, fill_value=0)
        result["ID"] = result["ID"].map(lambda x: [] if x == 0 else x)
        result.sort_values(
            "Year",
            ascending=True,
            inplace=True,
        )
        result["Cum_Num_Documents"] = result["Num_Documents"].cumsum()
        result["Cum_Global_Citations"] = result["Global_Citations"].cumsum()
        result["Avg_Global_Citations"] = (
            result["Global_Citations"] / result["Num_Documents"]
        )
        result["Avg_Global_Citations"] = result["Avg_Global_Citations"].map(
            lambda x: 0 if pd.isna(x) else round(x, 2)
        )
        result.pop("ID")
        self.X_ = result

    def table(self):
        self.apply()
        if self.sort_by == "Year":
            return self.X_.sort_index(axis=0, ascending=self.ascending)
        return self.X_.sort_values(by=self.sort_by, ascending=self.ascending)

    def plot_(self, values, darkness, label, figsize):
        self.apply()
        if self.plot == "Bar plot":
            return bar_plot(
                height=values,
                darkness=darkness,
                ylabel=label,
                figsize=figsize,
                cmap=self.colormap,
            )

        if self.plot == "Horizontal bar plot":
            return barh_plot(
                width=values,
                darkness=darkness,
                xlabel=label,
                figsize=figsize,
                cmap=self.colormap,
            )

    def num_documents_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Num_Documents"],
            darkness=self.X_["Global_Citations"],
            label="Num Documents by Year",
            figsize=(self.width, self.height),
        )

    def global_citations_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Global_Citations"],
            darkness=self.X_["Num_Documents"],
            label="Global Citations by Year",
            figsize=(self.width, self.height),
        )

    def cum_num_documents_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Cum_Num_Documents"],
            darkness=self.X_["Cum_Global_Citations"],
            label="Cum Num Documents by Year",
            figsize=(self.width, self.height),
        )

    def cum_global_citations_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Cum_Global_Citations"],
            darkness=self.X_["Cum_Num_Documents"],
            label="Cum Global Citations by Year",
            figsize=(self.width, self.height),
        )

    def avg_global_citations_by_year(self):
        self.apply()
        return self.plot_(
            values=self.X_["Avg_Global_Citations"],
            darkness=None,
            label="Avg Global Citations by Year",
            figsize=(self.width, self.height),
        )


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class DASHapp(DASH, Model):
    def __init__(self, years_range=None):
        #
        # Generic code
        #
        data = pd.read_csv("corpus.csv")

        Model.__init__(self, data, years_range=years_range)
        DASH.__init__(self)

        self.command_panel = [
            dash.dropdown(
                description="MENU:",
                options=[
                    "Table",
                    "Num Documents by Year",
                    "Global Citations by Year",
                    "Cum Num Documents by Year",
                    "Cum Global Citations by Year",
                    "Avg Global Citations by Year",
                ],
            ),
            dash.separator(text="Visualization"),
            dash.dropdown(
                description="Sort by:",
                options=[
                    "Year",
                    "Global_Citations",
                    "Num_Documents",
                    "Cum_Num_Documents",
                    "Cum_Global_Citations",
                    "Avg_Global_Citations",
                ],
            ),
            dash.ascending(),
            dash.dropdown(
                description="Plot:",
                options=["Bar plot", "Horizontal bar plot"],
            ),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]

    def interactive_output(self, **kwargs):

        super().interactive_output(**kwargs)

        if self.menu == self.menu_options[0]:

            self.set_enabled("Sort by:")
            self.set_enabled("Ascending:")
            self.set_disabled("Plot:")
            self.set_disabled("Colormap:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        else:

            self.set_disabled("Sort by:")
            self.set_disabled("Ascending:")
            self.set_enabled("Plot:")
            self.set_enabled("Colormap:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def by_year_analysis(
    years_range=None,
):

    return DASHapp(
        years_range=years_range,
    ).run()
