from collections import Counter

import pandas as pd
import ipywidgets as widgets

import techminer.core.dashboard as dash
from techminer.core import (
    CA,
    Dashboard,
    TF_matrix,
    TFIDF_matrix,
    add_counters_to_axis,
    clustering,
    corpus_filter,
    exclude_terms,
)

#  from techminer.core.params import EXCLUDE_COLS
from techminer.plots import counters_to_node_sizes, xy_clusters_plot
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
        clusters=None,
        cluster=None,
    ):
        ##
        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        #
        # Filter for cluster members
        #
        if clusters is not None and cluster is not None:
            data = corpus_filter(data=data, clusters=clusters, cluster=cluster)

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

        self.column = None
        self.min_occ = None
        self.max_items = None
        self.clustering_method = None
        self.n_clusters = None
        self.affinity = None
        self.linkage = None
        self.random_state = None
        self.top_n = None
        self.color_scheme = None
        self.x_axis = None
        self.y_axis = None
        self.width = None
        self.height = None

    def apply(self):

        ##
        ## Comparative analysis
        ##   from https://tlab.it/en/allegati/help_en_online/mcluster.htm
        ##

        ##
        ##  Computes TF matrix for terms in min_occurrence
        ##
        TF_matrix_ = TF_matrix(
            data=self.data,
            column=self.column,
            scheme="binary",
            min_occurrence=self.min_occ,
        )

        ##
        ##  Exclude Terms
        ##
        TF_matrix_ = exclude_terms(data=TF_matrix_, axis=1)

        ##
        ##  Adds counter to axies
        ##
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        ##
        ##  Computtes TFIDF matrix and select max_term frequent terms
        ##
        ##      tf-idf = tf * (log(N / df) + 1)
        ##
        TFIDF_matrix_ = TFIDF_matrix(
            TF_matrix=TF_matrix_,
            norm=None,
            use_idf=True,
            smooth_idf=False,
            sublinear_tf=False,
            max_items=self.max_items,
        )

        ##
        ##  Correspondence Analysis
        ##      10 first factors for ploting
        ##
        ca = CA()

        ca.fit(TFIDF_matrix_)

        self.eigenvalues_ = ca.eigenvalues_[0:10]
        self.explained_variance_ = ca.explained_variance_[0:10]

        z = ca.principal_coordinates_rows_
        z = z[z.columns[:10]]
        self.principal_coordinates_rows_ = z

        z = ca.principal_coordinates_cols_
        z = z[z.columns[:10]]
        self.principal_coordinates_cols_ = z

        self.TF_matrix_ = TF_matrix_
        self.TFIDF_matrix_ = TFIDF_matrix_

    def ca_plot_of_keywords(self):

        self.apply()

        ##
        ##  Selects the first n_factors to cluster
        ##
        X = pd.DataFrame(
            self.principal_coordinates_cols_,
            columns=["Dim-{}".format(i) for i in range(10)],
            index=self.TFIDF_matrix_.columns,
        )

        return xy_clusters_plot(
            x=X["Dim-{}".format(self.x_axis)],
            y=X["Dim-{}".format(self.y_axis)],
            x_axis_at=0,
            y_axis_at=0,
            labels=self.TFIDF_matrix_.columns,
            node_sizes=counters_to_node_sizes(self.TFIDF_matrix_.columns),
            color_scheme=self.color_scheme,
            xlabel="Dim-{}".format(self.x_axis),
            ylabel="Dim-{}".format(self.y_axis),
            figsize=(self.width, self.height),
        )

    def cluster_plot_of_keywords(self):

        self.apply()

        X = pd.DataFrame(
            self.principal_coordinates_cols_,
            columns=["Dim-{}".format(i) for i in range(10)],
            index=self.TFIDF_matrix_.columns,
        )

        (
            self.n_clusters,
            self.labels_,
            self.cluster_members_,
            self.cluster_centers_,
            self.cluster_names_,
        ) = clustering(
            X=X,
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            random_state=self.random_state,
            top_n=self.top_n,
            name_prefix="Cluster {}",
        )

        ##
        ## Cluster filters
        ##
        self.generate_cluster_filters(
            terms=X.index,
            labels=self.labels_,
        )

        y = self.cluster_members_.copy()
        y = y.applymap(lambda w: pd.NA if w == "" else w)
        node_sizes = [500 + 2500 * len(y[m].dropna()) for m in y.columns]

        return xy_clusters_plot(
            x=self.cluster_centers_["Dim-{}".format(self.x_axis)],
            y=self.cluster_centers_["Dim-{}".format(self.y_axis)],
            x_axis_at=0,
            y_axis_at=0,
            labels=["CLUST_{} xxx".format(i) for i in range(self.n_clusters)],
            node_sizes=node_sizes,
            color_scheme=self.color_scheme,
            xlabel="Dim-{}".format(self.x_axis),
            ylabel="Dim-{}".format(self.y_axis),
            figsize=(self.width, self.height),
        )

    def cluster_plot_of_documents(self):

        self.apply()

        X = pd.DataFrame(
            self.principal_coordinates_rows_,
            columns=["Dim-{}".format(i) for i in range(10)],
            index=[
                "{} {}".format(i, i)
                for i in range(len(self.principal_coordinates_rows_))
            ],
        )

        (
            self.n_clusters,
            self.labels_,
            self.cluster_members_,
            self.cluster_centers_,
            self.cluster_names_,
        ) = clustering(
            X=X,
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            random_state=self.random_state,
            top_n=self.top_n,
            name_prefix="Cluster {}",
        )

        ##
        ## Cluster filters
        ##
        self.generate_cluster_filters(
            terms=X.index,
            labels=self.labels_,
        )

        y = self.cluster_members_.copy()
        y = y.applymap(lambda w: pd.NA if w == "" else w)
        node_sizes = [500 + 2500 * len(y[m].dropna()) for m in y.columns]

        return xy_clusters_plot(
            x=self.cluster_centers_["Dim-{}".format(self.x_axis)],
            y=self.cluster_centers_["Dim-{}".format(self.y_axis)],
            x_axis_at=0,
            y_axis_at=0,
            labels=["CLUST_{} xxx".format(i) for i in range(self.n_clusters)],
            node_sizes=node_sizes,
            color_scheme=self.color_scheme,
            xlabel="Dim-{}".format(self.x_axis),
            ylabel="Dim-{}".format(self.y_axis),
            figsize=(self.width, self.height),
        )


###############################################################################
##
##  DASHBOARD
##
###############################################################################


COLUMNS = [
    "Author_Keywords_CL",
    "Author_Keywords",
    "Index_Keywords_CL",
    "Index_Keywords",
    "Keywords_CL",
]


class App(Dashboard, Model):
    def __init__(
        self,
        limit_to=None,
        exclude=None,
        years_range=None,
    ):

        data = filter_records(pd.read_csv("corpus.csv"))

        Model.__init__(
            self,
            data=data,
            limit_to=limit_to,
            exclude=exclude,
            years_range=years_range,
        )

        #  COLUMNS = sorted(
        #      [column for column in sorted(data.columns) if column not in EXCLUDE_COLS]
        #  )

        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.Dropdown(
                options=[
                    "CA plot of keywords",
                    "Cluster plot of keywords",
                    "Cluster plot of documents",
                ],
            ),
            dash.HTML("Parameters:"),
            dash.Dropdown(
                description="Column:",
                options=[t for t in sorted(data.columns) if t in COLUMNS],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.HTML("Clustering:"),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=50, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.random_state(),
            dash.HTML("Visualization:"),
            dash.top_n(m=10, n=51, i=5),
            dash.color_scheme(),
            dash.x_axis(),
            dash.y_axis(),
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
                # Clustering
                "clustering_method": self.command_panel[7],
                "n_clusters": self.command_panel[8],
                "affinity": self.command_panel[9],
                "linkage": self.command_panel[10],
                "random_state": self.command_panel[11],
                # Visualization
                "top_n": self.command_panel[13],
                "colors": self.command_panel[14],
                "x_axis": self.command_panel[15],
                "y_axis": self.command_panel[16],
                "width": self.command_panel[17],
                "height": self.command_panel[18],
            },
        )

        Dashboard.__init__(self)

    def interactive_output(self, **kwargs):

        Dashboard.interactive_output(self, **kwargs)

        def visualization_disabled():

            self.set_disabled("Color Scheme:")
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        def visualization_enabled():

            self.set_enabled("Color Scheme:")
            self.set_enabled("X-axis:")
            self.set_enabled("Y-axis:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        def clustering_disabled():

            self.set_disabled("N Factors:")
            self.set_disabled("Clustering Method:")
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            self.set_disabled("Random State:")

        def clustering_enabled():

            self.set_enabled("N Factors:")
            self.set_enabled("Clustering Method:")
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            self.set_enabled("Random State:")

            self.enable_disable_clustering_options(include_random_state=True)

        if self.menu == "Correspondence analysis plot":

            clustering_disabled()
            visualization_enabled()

        if self.menu == "Cluster members":

            clustering_enabled()
            visualization_disabled()

        if self.menu == "Cluster plot":

            clustering_enabled()
            visualization_enabled()


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def comparative_analysis(
    limit_to=None,
    exclude=None,
    years_range=None,
):
    return App(
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
    ).run()
