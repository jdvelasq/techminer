import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.manifold import MDS

import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import (
    Dashboard,
    TF_matrix,
    TFIDF_matrix,
    add_counters_to_axis,
    clustering,
    corpus_filter,
    exclude_terms,
    normalize_network,
    sort_axis,
)
from techminer.plots import (
    conceptual_structure_map,
    counters_to_node_sizes,
    xy_clusters_plot,
)
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

    def apply(self):
        #
        X = self.data.copy()

        #
        # 1.-- TF matrix
        #
        TF_matrix_ = TF_matrix(
            data=X,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occ,
        )

        #
        # 2.-- Limit to / Exclude
        #
        TF_matrix_ = exclude_terms(data=TF_matrix_, axis=1)

        #
        # 3.-- Add counters to axes
        #
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        #
        # 4.-- Select top terms
        #
        TF_matrix_ = sort_axis(
            data=TF_matrix_, num_documents=True, axis=1, ascending=False
        )
        if len(TF_matrix_.columns) > self.max_items:
            TF_matrix_ = TF_matrix_.loc[:, TF_matrix_.columns[0 : self.max_items]]
            rows = TF_matrix_.sum(axis=1)
            rows = rows[rows > 0]
            TF_matrix_ = TF_matrix_.loc[rows.index, :]

        #
        # 5.-- Co-occurrence matrix and normalization
        #
        M = np.matmul(TF_matrix_.transpose().values, TF_matrix_.values)
        M = pd.DataFrame(M, columns=TF_matrix_.columns, index=TF_matrix_.columns)
        M = normalize_network(M, normalization=self.normalization)

        #
        # 6.-- Dissimilarity matrix
        #
        M = 1 - M
        for i in M.columns.tolist():
            M.at[i, i] = 0.0

        #
        # 5.-- Number of factors
        #
        # self.n_components = 2 if self.decomposition_method == "MDS" else 10

        #
        # 6.-- Factor decomposition
        #
        model = {
            "Factor Analysis": FactorAnalysis,
            "PCA": PCA,
            "Fast ICA": FastICA,
            "SVD": TruncatedSVD,
            "MDS": MDS,
        }[self.decomposition_method]

        model = (
            model(
                n_components=self.n_components,
                random_state=int(self.random_state),
                dissimilarity="precomputed",
            )
            if self.decomposition_method == "MDS"
            else model(
                n_components=self.n_components, random_state=int(self.random_state)
            )
        )

        R = model.fit_transform(X=M.values)
        R = pd.DataFrame(
            R,
            columns=["Dim-{}".format(i) for i in range(self.n_components)],
            index=M.columns,
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
        ## Cluster filters
        ##
        self.generate_cluster_filters(terms=R.index, labels=self.labels_)

        R["Cluster"] = self.labels_

        self.coordinates_ = R

        #
        # 8.-- Results
        #
        self.X_ = R

    def cluster_members(self):
        self.apply()
        return self.cluster_members_

    def conceptual_structure_map(self):
        self.apply()
        X = self.X_
        cluster_labels = X.Cluster
        X.pop("Cluster")
        return conceptual_structure_map(
            coordinates=X,
            cluster_labels=cluster_labels,
            top_n=self.top_n,
            figsize=(self.width, self.height),
        )

    def conceptual_structure_members(self):
        self.apply()
        return self.cluster_members_

    def cluster_plot(self):
        self.apply()
        return xy_clusters_plot(
            x=self.cluster_centers_["Dim-{}".format(self.x_axis)],
            y=self.cluster_centers_["Dim-{}".format(self.y_axis)],
            x_axis_at=0,
            y_axis_at=0,
            labels=self.cluster_names_,
            node_sizes=counters_to_node_sizes(self.cluster_names_),
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

COLUMNS = sorted(
    [
        "Abstract_phrase_words",
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

        self.command_panel = [
            dash.HTML("Display:", margin="0px 0px 0px 5px", hr=False),
            dash.Dropdown(
                options=[
                    "Cluster members",
                    "Cluster plot",
                    "Conceptual Structure Map",
                    "Conceptual Structure Members",
                ],
            ),
            dash.HTML("Parameters:"),
            dash.Dropdown(
                description="Column:",
                options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.normalization(include_none=False),
            dash.random_state(),
            dash.HTML("Clustering"),
            dash.decomposition_method(),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=21, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.HTML("Visualization"),
            dash.top_n(),
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
                "normalization": self.command_panel[6],
                "random_state": self.command_panel[7],
                # Clustering:
                "decomposition_method": self.command_panel[9],
                "clustering_method": self.command_panel[10],
                "n_clusters": self.command_panel[11],
                "affinity": self.command_panel[12],
                "linkage": self.command_panel[13],
                # Visualization
                "top_n": self.command_panel[15],
                "color_scheme": self.command_panel[16],
                "x_axis": self.command_panel[17],
                "y_axis": self.command_panel[18],
                "width": self.command_panel[19],
                "height": self.command_panel[20],
            },
        )

        Dashboard.__init__(self)

        self.interactive_output(
            **{
                # Display:
                "menu": self.command_panel[1].value,
                # Parameters:
                "column": self.command_panel[3].value,
                "min_occ": self.command_panel[4].value,
                "max_items": self.command_panel[5].value,
                "normalization": self.command_panel[6].value,
                "random_state": self.command_panel[7].value,
                # Clustering:
                "decomposition_method": self.command_panel[9].value,
                "clustering_method": self.command_panel[10].value,
                "n_clusters": self.command_panel[11].value,
                "affinity": self.command_panel[12].value,
                "linkage": self.command_panel[13].value,
                # Visualization
                "top_n": self.command_panel[15].value,
                "color_scheme": self.command_panel[16].value,
                "x_axis": self.command_panel[17].value,
                "y_axis": self.command_panel[18].value,
                "width": self.command_panel[19].value,
                "height": self.command_panel[20].value,
            }
        )

    def interactive_output(self, **kwargs):

        Dashboard.interactive_output(self, **kwargs)

        if self.menu == "Cluster members":
            #
            self.n_components = 10
            #
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")
            self.set_disabled("Color Scheme:")

        if self.menu == "Cluster plot":
            #
            self.n_components = 10
            #
            self.set_enabled("X-axis:")
            self.set_enabled("Y-axis:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")
            self.set_enabled("Color Scheme:")

        if self.menu in ["Conceptual Structure Map", "Conceptual Structure Members"]:
            #
            self.n_components = 2
            #
            self.set_disabled("X-axis:")
            self.set_disabled("Y-axis:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")
            self.set_disabled("Color Scheme:")

        self.enable_disable_clustering_options()
