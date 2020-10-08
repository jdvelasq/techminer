import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.manifold import MDS

import techminer.core.dashboard as dash
from techminer.core import (
    DASH,
    TF_matrix,
    TFIDF_matrix,
    add_counters_to_axis,
    clustering,
    corpus_filter,
    limit_to_exclude,
)
from techminer.core.dashboard import fig_height
from techminer.plots import counters_to_node_sizes, xy_clusters_plot
import ipywidgets as widgets

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

        ## TLAB suggestion
        self.n_components = 10

    def apply(self):
        #
        X = self.data.copy()

        #
        # 1.-- TF matrix
        #
        M = TF_matrix(
            data=X,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occ,
        )

        #
        # 2.-- Limit to / Exclude
        #
        M = limit_to_exclude(
            data=M,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        #
        # 3.-- Computtes TFIDF matrix and select max_term frequent terms
        #
        #      tf-idf = tf * (log(N / df) + 1)
        #
        if self.apply_to == "TF*IDF matrix":
            M = TFIDF_matrix(
                TF_matrix=M,
                norm=None,
                use_idf=False,
                smooth_idf=False,
                sublinear_tf=False,
                max_items=self.max_items,
            )
        else:
            if len(M.columns) > self.max_items:
                top_items = M.sum(axis=0)
                top_items = top_items.sort_values(ascending=False)
                top_items = top_items.head(self.max_items)
                M = M.loc[:, top_items.index]
                rows = M.sum(axis=1)
                rows = rows[rows > 0]
                M = M.loc[rows.index, :]

        #
        # 4.-- Add counters to axes
        #
        M = add_counters_to_axis(X=M, axis=1, data=self.data, column=self.column)

        #
        # 5.-- Transpose
        #
        M = M.transpose()

        #
        # 6.-- Factor decomposition
        #
        model = {
            "Factor Analysis": FactorAnalysis,
            "PCA": PCA,
            "Fast ICA": FastICA,
            "SVD": TruncatedSVD,
            "MDS": MDS,
        }[self.decomposition_method](
            n_components=self.n_components, random_state=int(self.random_state)
        )

        if self.decomposition_method == "MDS":
            R = model.fit_transform(X=M.values)
        else:
            R = model.fit_transform(X=M.values)

        R = pd.DataFrame(
            R,
            columns=["Dim-{:>02d}".format(i) for i in range(self.n_components)],
            index=M.index,
        )

        #
        # 7.-- Clustering
        #
        (
            self.n_clusters,
            self.labels_,
            self.cluster_members_,
            self.cluster_centers_,
            self.cluster_names_,
        ) = clustering(
            X=R,
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            random_state=self.random_state,
            top_n=self.top_n,
            name_prefix="Cluster {}",
        )

        ##
        R["Cluster"] = self.labels_

        #
        # 8.-- Results
        #
        self.X_ = R

    def cluster_members(self):
        self.apply()
        return self.cluster_members_

    def cluster_plot(self):

        ## clustering
        try:
            self.apply()
        except:
            return "Clustering algorithm did not converge"

        ## plot
        return xy_clusters_plot(
            x=self.cluster_centers_[self.cluster_centers_.columns[self.x_axis]],
            y=self.cluster_centers_[self.cluster_centers_.columns[self.y_axis]],
            x_axis_at=0,
            y_axis_at=0,
            labels=self.cluster_names_,
            node_sizes=counters_to_node_sizes(self.cluster_names_),
            color_scheme="4 Quadrants",
            xlabel="Dim-{}".format(self.x_axis),
            ylabel="Dim-{}".format(self.y_axis),
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
        "Author_Keywords_CL",
        "Author_Keywords",
        "Index_Keywords_CL",
        "Index_Keywords",
        "Keywords_CL",
        "Title_words_CL",
        "Title_words",
    ]
)

###############################################################################
##
##  DASHBOARD
##
###############################################################################


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

        #
        self.command_panel = [
            dash.HTML("Display:", margin="0px 0px 0px 5px", hr=False),
            dash.Dropdown(
                options=[
                    "Cluster members",
                    "Cluster plot",
                ],
            ),
            dash.HTML("Parameters:"),
            dash.Dropdown(
                description="Column:",
                options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.random_state(),
            dash.HTML("Decomposition:"),
            dash.Dropdown(
                description="Apply to:",
                options=[
                    "TF matrix",
                    "TF*IDF matrix",
                ],
            ),
            dash.decomposition_method("Method:"),
            dash.HTML("Clustering:"),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=51, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.HTML("Visualization:"),
            dash.Dropdown(
                options=["Num Documents", "Global Citations"],
                description="Top by:",
            ),
            dash.top_n(10, 51, 1),
            dash.x_axis(n=self.n_components),
            dash.y_axis(n=self.n_components),
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
                "random_state": self.command_panel[6],
                # Decomposition:
                "apply_to": self.command_panel[8],
                "decomposition_method": self.command_panel[9],
                # Clustering:
                "clustering_method": self.command_panel[11],
                "n_clusters": self.command_panel[12],
                "affinity": self.command_panel[13],
                "linkage": self.command_panel[14],
                # Visualization
                "top_by": self.command_panel[16],
                "top_n": self.command_panel[17],
                "x_axis": self.command_panel[18],
                "y_axis": self.command_panel[19],
                "width": self.command_panel[20],
                "height": self.command_panel[21],
            },
        )

        DASH.__init__(self)

        self.interactive_output(
            **{
                # Display:
                "menu": self.command_panel[1].value,
                # Parameters:
                "column": self.command_panel[3].value,
                "min_occ": self.command_panel[4].value,
                "max_items": self.command_panel[5].value,
                "random_state": self.command_panel[6].value,
                # Decomposition:
                "apply_to": self.command_panel[8].value,
                "decomposition_method": self.command_panel[9].value,
                # Clustering:
                "clustering_method": self.command_panel[11].value,
                "n_clusters": self.command_panel[12].value,
                "affinity": self.command_panel[13].value,
                "linkage": self.command_panel[14].value,
                # Visualization
                "top_by": self.command_panel[16].value,
                "top_n": self.command_panel[17].value,
                "x_axis": self.command_panel[18].value,
                "y_axis": self.command_panel[19].value,
                "width": self.command_panel[20].value,
                "height": self.command_panel[21].value,
            }
        )

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == "Cluster members":
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        if self.menu == "Cluster plot":
            self.set_enabled("X-axis:")
            self.set_enabled("Y-axis:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        self.enable_disable_clustering_options()

        #  self.set_options(name="X-axis:", options=list(range(self.n_components)))
        #  self.set_options(name="Y-axis:", options=list(range(self.n_components)))


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def latent_semantic_analysis(
    limit_to=None,
    exclude=None,
    years_range=None,
):
    return DASHapp(
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
    ).run()
