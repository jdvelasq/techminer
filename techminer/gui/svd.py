import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

import techminer.common as cmn
import techminer.dashboard as dash
import techminer.plots as plt
from techminer.core import limit_to_exclude
from techminer.dashboard import DASH
from techminer.diagram_plot import diagram_plot
from techminer.document_term import TF_matrix, TFIDF_matrix
from techminer.params import EXCLUDE_COLS

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(self, data, limit_to, exclude):
        #
        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude
        #
        self.analysis_type = None
        self.ascending = None
        self.colormap = None
        self.column = None
        self.height = None
        self.max_items = None
        self.max_terms = None
        self.min_occ = None
        self.n_components = None
        self.n_iter = None
        self.norm = None
        self.random_state = None
        self.smooth_idf = None
        self.sort_by = None
        self.sublinear_tf = None
        self.top_by = None
        self.top_n = None
        self.use_idf = None
        self.width = None
        self.x_axis = None
        self.y_axis = None

    def apply(self):

        ##
        ## SVD for documents x terms matrix & co-occurrence matrix
        ##   from https://tlab.it/en/allegati/help_en_online/msvd.htm
        ##

        #
        # 1.-- Computes TF matrix for terms in min_occurrence
        #
        TF_matrix_ = TF_matrix(
            data=self.data,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occ,
        )

        #
        # 2.-- Computtes TFIDF matrix and select max_term frequent terms
        #
        #      tf-idf = tf * (log(N / df) + 1)
        #
        TFIDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm=None,
            use_idf=True,
            smooth_idf=False,
            sublinear_tf=False,
            max_items=self.max_items,
        )

        TFIDF_matrix_ = cmn.add_counters_to_axis(
            X=TFIDF_matrix_, axis=1, data=self.data, column=self.column
        )

        #
        # 3.-- Data to analyze
        #
        X = None
        if self.analysis_type == "Co-occurrence":
            X = np.matmul(TFIDF_matrix_.transpose().values, TFIDF_matrix_.values)
            X = pd.DataFrame(
                X, columns=TFIDF_matrix_.columns, index=TFIDF_matrix_.columns
            )
        if self.analysis_type == "TF*IDF":
            X = TFIDF_matrix_

        #
        # 4.-- SVD for a maximum of 20 dimensions
        #
        TruncatedSVD_ = TruncatedSVD(
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=int(self.random_state),
        ).fit(X)

        #
        # 5.-- Results
        #
        axis_names = ["Dim-{:>02d}".format(i) for i in range(self.n_components)]
        self.components_ = pd.DataFrame(
            np.transpose(TruncatedSVD_.components_),
            columns=axis_names,
            index=X.columns,
        )
        self.statistics_ = pd.DataFrame(
            TruncatedSVD_.explained_variance_,
            columns=["Explained Variance"],
            index=axis_names,
        )
        self.statistics_["Explained Variance"] = TruncatedSVD_.explained_variance_
        self.statistics_[
            "Explained Variance Ratio"
        ] = TruncatedSVD_.explained_variance_ratio_
        self.statistics_["Singular Values"] = TruncatedSVD_.singular_values_

    def table(self):
        self.apply()
        X = self.components_
        X = limit_to_exclude(
            data=X,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )
        X = cmn.sort_by_axis(data=X, sort_by=self.top_by, ascending=False, axis=0)
        X = X.head(self.top_n)
        X = cmn.sort_by_axis(
            data=X, sort_by=self.sort_by, ascending=self.ascending, axis=0
        )
        return X

    def statistics(self):
        self.apply()
        return self.statistics_

    def plot_singular_values(self):

        self.apply()

        return plt.barh(
            width=self.statistics_["Singular Values"],
            cmap=self.colormap,
            figsize=(self.width, self.height),
        )

    def plot_relationships(self):

        self.apply()
        X = self.table()

        return diagram_plot(
            x=X[X.columns[self.x_axis]],
            y=X[X.columns[self.y_axis]],
            labels=X.index,
            x_axis_at=0,
            y_axis_at=0,
            cmap=self.colormap,
            width=self.width,
            height=self.height,
        )


###############################################################################
##
##  DASHBOARD
##
###############################################################################


COLUMNS = [
    "Author_Keywords",
    "Index_Keywords",
    "Abstract_words_CL",
    "Abstract_words",
    "Title_words_CL",
    "Title_words",
    "Author_Keywords_CL",
    "Index_Keywords_CL",
]


class DASHapp(DASH, Model):
    def __init__(
        self,
        limit_to=None,
        exclude=None,
    ):
        data = pd.read_csv("corpus.csv")

        Model.__init__(self, data, limit_to, exclude)
        DASH.__init__(self)

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

        COLUMNS = sorted(
            [column for column in data.columns if column not in EXCLUDE_COLS]
        )

        self.command_panel = [
            dash.dropdown(
                description="MENU:",
                options=[
                    "Table",
                    "Statistics",
                    "Plot singular values",
                    "Plot relationships",
                ],
            ),
            dash.dropdown(
                desc="Column:",
                options=[t for t in data if t in COLUMNS],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.separator(text="SVD"),
            dash.dropdown(
                desc="Analysis type:",
                options=[
                    "Co-occurrence",
                    "TF*IDF",
                ],
            ),
            dash.n_components(),
            dash.random_state(),
            dash.n_iter(),
            dash.separator(text="Visualization"),
            dash.dropdown(
                desc="Top by:",
                options=[
                    "Num Documents",
                    "Global Citations",
                ],
            ),
            dash.top_n(),
            dash.dropdown(
                desc="Sort by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Global Citations",
                ],
            ),
            dash.ascending(),
            dash.cmap(),
            dash.x_axis(),
            dash.y_axis(),
            dash.fig_width(),
            dash.fig_height(),
        ]

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu in ["Table"]:
            self.set_enabled("Top by:")
            self.set_enabled("Top N:")
            self.set_enabled("sort by:")
            self.set_enabled("ascending:")
            self.set_disabled("Colormap:")
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        if self.menu in ["Statistics"]:
            self.set_disabled("Top by:")
            self.set_disabled("Top N:")
            self.set_disabled("sort by:")
            self.set_disabled("ascending:")
            self.set_disabled("Colormap:")
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        if self.menu in ["Plot singular values"]:
            self.set_disabled("Top by:")
            self.set_disabled("Top N:")
            self.set_disabled("sort by:")
            self.set_disabled("ascending:")
            self.set_enabled("Colormap:")
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        if self.menu in ["Plot relationships"]:
            self.set_enabled("Top by:")
            self.set_enabled("Top N:")
            self.set_disabled("sort by:")
            self.set_disabled("ascending:")
            self.set_enabled("Colormap:")
            self.set_enabled("X-axis:")
            self.set_enabled("Y-axis:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        self.set_options("X-axis:", list(range(self.n_components)))
        self.set_options("Y-axis:", list(range(self.n_components)))


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def app(
    limit_to=None,
    exclude=None,
):
    return DASHapp(
        limit_to=limit_to,
        exclude=exclude,
    ).run()
