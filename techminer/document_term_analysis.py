import ipywidgets as widgets
import pandas as pd

import techminer.core.dashboard as dash
from techminer.core import (
    DASH,
    TF_matrix,
    TFIDF_matrix,
    add_counters_to_axis,
    limit_to_exclude,
    sort_by_axis,
)
from techminer.plots import bar_plot, barh_plot

###############################################################################
##
##  Model
##
###############################################################################


class Model:
    def __init__(self, data, limit_to, exclude, years_range):
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

    def apply(self):

        ##
        ## Builds TF matrix
        ##
        matrix = TF_matrix(
            data=self.data,
            column=self.column,
            scheme="raw",
            min_occurrence=self.min_occ,
        )

        ##
        ## Limit to / Exclude
        ##
        matrix = limit_to_exclude(
            data=matrix,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ##
        ## TF*IDF matrix
        ##
        TF_matrix_ = add_counters_to_axis(
            X=matrix, axis=1, data=self.data, column=self.column
        )

        if self.norm is not None:
            self.norm = self.norm.lower()

        TFIDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )

        ##
        ## Selects max_items
        ##
        vector = (
            TFIDF_matrix_.sum(axis=0).sort_values(ascending=False).head(self.max_items)
        )
        TFIDF_matrix_ = vector.to_frame()
        TFIDF_matrix_.columns = ["TF*IDF"]

        if self.sort_by == "TF*IDF":
            TFIDF_matrix_["TEXT"] = TFIDF_matrix_.index.map(lambda w: w.split(" ")[-1])
            TFIDF_matrix_ = TFIDF_matrix_.sort_values(
                ["TF*IDF", "TEXT"], ascending=self.ascending
            )
            TFIDF_matrix_.pop("TEXT")
        else:
            TFIDF_matrix_ = sort_by_axis(
                data=TFIDF_matrix_,
                sort_by=self.sort_by,
                ascending=self.ascending,
                axis=0,
            )

        if self.use_idf is False:
            TFIDF_matrix_["TF*IDF"] = TFIDF_matrix_["TF*IDF"].map(int)

        self.X_ = TFIDF_matrix_

    def table(self):
        self.apply()
        return self.X_.style.set_precision(2).background_gradient(cmap=self.colormap)

    def limit_to_python_code(self):
        self.apply()
        items = self.X_.index.tolist()
        items = [" ".join(item.split(" ")[:-1]) for item in items]
        HTML = "LIMIT_TO = {<br>"
        HTML += '    "' + self.column + '": [<br>'
        for item in items:
            HTML += '        "' + item + '",<br>'
        HTML += "    ]<br>"
        HTML += "}<br>"
        return widgets.HTML("<pre>" + HTML + "</pre>")

    def bar_plot(self):
        self.apply()
        return bar_plot(
            height=self.X_["TF*IDF"],
            darkness=None,
            cmap=self.colormap,
            figsize=(self.width, self.height),
            ylabel="TF*IDF",
        )

    def horizontal_bar_plot(self):
        self.apply()
        return barh_plot(
            width=self.X_["TF*IDF"],
            darkness=None,
            cmap=self.colormap,
            figsize=(self.width, self.height),
            xlabel="TF*IDF",
        )


###############################################################################
##
##  DASH
##
###############################################################################

COLUMNS = sorted(
    [
        "Abstract_words_CL",
        "Abstract_words",
        "Author_Keywords_CL",
        "Author_Keywords",
        "Index_Keywords_CL",
        "Index_Keywords",
        "Title_words_CL",
        "Title_words",
        "Keywords_CL",
    ]
)


class DASHapp(DASH, Model):
    def __init__(self, limit_to=None, exclude=None, years_range=None):
        data = pd.read_csv("corpus.csv")

        Model.__init__(
            self, data=data, limit_to=limit_to, exclude=exclude, years_range=years_range
        )
        DASH.__init__(self)

        self.pandas_max_rows = 300

        self.command_panel = [
            dash.dropdown(
                description="MENU:",
                options=[
                    "Table",
                    "Bar plot",
                    "Horizontal bar plot",
                    "LIMIT TO python code",
                ],
            ),
            dash.dropdown(
                description="Column:",
                options=[t for t in sorted(data.columns) if t in COLUMNS],
            ),
            dash.dropdown(
                description="Norm:",
                options=[None, "L1", "L2"],
            ),
            dash.dropdown(
                description="Use IDF:",
                options=[
                    True,
                    False,
                ],
            ),
            dash.dropdown(
                description="Smooth IDF:",
                options=[
                    True,
                    False,
                ],
            ),
            dash.dropdown(
                description="Sublinear TF:",
                options=[
                    True,
                    False,
                ],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.separator(text="Visualization"),
            dash.dropdown(
                description="Sort by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Global Citations",
                    "TF*IDF",
                ],
            ),
            dash.ascending(),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == self.menu_options[0]:
            self.set_disabled("Width:")
            self.set_disabled("Height:")
        else:
            self.set_enabled("Width:")
            self.set_enabled("Height:")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def document_term_analysis(limit_to=None, exclude=None, years_range=None):
    return DASHapp(
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
    ).run()
