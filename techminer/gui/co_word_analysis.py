import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import (
    CA,
    DASH,
    TF_matrix,
    TFIDF_matrix,
    add_counters_to_axis,
    clustering,
    corpus_filter,
    limit_to_exclude,
    normalize_network,
    sort_by_axis,
    cluster_table_to_list,
    cluster_table_to_python_code,
    keywords_coverage,
)
from techminer.plots import (
    ax_text_node_labels,
    counters_to_node_sizes,
    expand_ax_limits,
    set_spines_invisible,
    xy_clusters_plot,
    xy_cluster_members_plot,
)

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

        self.column = None
        self.min_occ = None
        self.max_items = None
        self.normalization = None
        self.clustering_method = None
        self.n_clusters = None
        self.affinity = None
        self.linkage = None
        self.random_state = None
        self.x_axis = None
        self.y_axis = None
        self.top_n = None
        self.colors = None
        self.width = None
        self.height = None

    def apply(self):

        ##
        ## Concept mapping
        ## https://tlab.it/en/allegati/help_en_online/mmappe2.htm
        ##

        ##
        ##  Co-occurrence matrix
        ##
        TF_matrix_ = TF_matrix(
            data=self.data,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occ,
        )

        ##
        ##  Limit to/Exclude
        ##
        TF_matrix_ = limit_to_exclude(
            data=TF_matrix_,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ##
        ##  Select max items
        ##
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )
        TF_matrix_ = sort_by_axis(
            data=TF_matrix_, sort_by="Num Documents", ascending=False, axis=1
        )
        TF_matrix_ = TF_matrix_[TF_matrix_.columns[: self.max_items]]
        if len(TF_matrix_.columns) > self.max_items:
            top_items = TF_matrix_.sum(axis=0)
            top_items = top_items.sort_values(ascending=False)
            top_items = top_items.head(self.max_items)
            TF_matrix_ = TF_matrix_.loc[:, top_items.index]
            rows = TF_matrix_.sum(axis=1)
            rows = rows[rows > 0]
            TF_matrix_ = TF_matrix_.loc[rows.index, :]

        ##
        ##  Co-occurrence matrix and association index
        ##
        X = np.matmul(TF_matrix_.transpose().values, TF_matrix_.values)
        X = pd.DataFrame(X, columns=TF_matrix_.columns, index=TF_matrix_.columns)
        X = normalize_network(X=X, normalization=self.normalization)

        ##
        ##  Clustering of the dissimilarity matrix
        ##
        (
            self.n_clusters,
            self.labels_,
            self.cluster_members_,
            self.cluster_centers_,
            self.cluster_names_,
        ) = clustering(
            X=(1 - X),
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            random_state=self.random_state,
            top_n=self.top_n,
            name_prefix="Cluster {}",
        )
        self.X_ = X

        ##
        ## Cluster co-occurrence
        ##
        M = X.copy()
        M["CLUSTER"] = self.labels_
        M = M.groupby("CLUSTER").sum()
        #
        M = M.transpose()
        M["CLUSTER"] = self.labels_
        M = M.groupby("CLUSTER").sum()
        #
        M.columns = ["Cluster {}".format(i) for i in range(self.n_clusters)]
        M.index = M.columns
        #
        self.cluster_co_occurrence_ = M

        ##
        ##  Strategic Map
        ##

        ## clusters name
        strategic_map = pd.DataFrame(
            self.cluster_names_, columns=["Cluster name"], index=M.columns
        )
        strategic_map["Density"] = 0.0
        strategic_map["Centrality"] = 0.0

        ## Density -- internal conections
        for cluster in M.columns:
            strategic_map.at[cluster, "Density"] = M[cluster][cluster]

        ## Centrality -- outside conections
        strategic_map["Centrality"] = M.sum()
        strategic_map["Centrality"] = (
            strategic_map["Centrality"] - strategic_map["Density"]
        )

        self.strategic_map_ = strategic_map

    def mds_keywords_map(self):

        ##
        ## Compute co-occurrence matrix
        ##
        self.apply()
        X = self.X_.copy()

        ##
        ## MDS
        ##
        embedding = MDS(n_components=2)
        X_transformed = embedding.fit_transform(
            1 - X,
        )
        x_axis = X_transformed[:, 0]
        y_axis = X_transformed[:, 1]

        ##
        ## Plot
        ##
        return xy_cluster_members_plot(
            x=x_axis,
            y=y_axis,
            x_axis_at=0,
            y_axis_at=0,
            labels=self.labels_,
            keywords=X.index,
            color_scheme=self.colors,
            xlabel="Dim-0",
            ylabel="Dim-1",
            figsize=(self.width, self.height),
        )

    def mds_cluster_map(self):

        ##
        ## Compute co-occurrence matrix
        ##
        self.apply()
        X = self.X_.copy()

        ##
        ## MDS
        ##
        embedding = MDS(n_components=2)
        X_transformed = embedding.fit_transform(
            1 - X,
        )

        X_transformed = pd.DataFrame(X_transformed, columns=["x_axis", "y_axis"])
        X_transformed["CLUSTER"] = self.labels_
        X_transformed = X_transformed.groupby(["CLUSTER"], as_index=True).mean()
        X_transformed = X_transformed.sort_index(axis=0)

        ##
        ## Cluster coordinates
        ##
        x_axis = X_transformed.x_axis.tolist()
        y_axis = X_transformed.y_axis.tolist()

        ##
        ## Cluster names
        ##
        labels = [
            "CLUST_{} {}".format(index, label)
            for index, label in enumerate(self.cluster_names_)
        ]

        return xy_clusters_plot(
            x=x_axis,
            y=y_axis,
            x_axis_at=0,
            y_axis_at=0,
            labels=labels,
            node_sizes=counters_to_node_sizes(labels),
            color_scheme=self.colors,
            xlabel="Dim-{}".format(self.x_axis),
            ylabel="Dim-{}".format(self.y_axis),
            figsize=(self.width, self.height),
        )

    def mds_keywords_by_cluster_table(self):
        self.apply()
        return self.cluster_members_

    def mds_keywords_by_cluster_list(self):
        self.apply()
        return cluster_table_to_list(self.cluster_members_)

    def mds_keywords_by_cluster_python_code(self):
        self.apply()
        return cluster_table_to_python_code(self.column, self.cluster_members_)

    def ca_keywords_map(self):

        ##
        ## Compute co-occurrence matrix
        ##
        self.apply()
        X = self.X_.copy()

        ##
        ## CA
        ##
        ca = CA()
        ca.fit(1 - X)
        X_transformed = ca.principal_coordinates_cols_
        x_axis = X_transformed.loc[:, X_transformed.columns[self.x_axis]]
        y_axis = X_transformed.loc[:, X_transformed.columns[self.y_axis]]

        ##
        ## Plot
        ##
        return xy_cluster_members_plot(
            x=x_axis,
            y=y_axis,
            x_axis_at=0,
            y_axis_at=0,
            labels=self.labels_,
            keywords=X.index,
            color_scheme=self.colors,
            xlabel="Dim-0",
            ylabel="Dim-1",
            figsize=(self.width, self.height),
        )

    def ca_cluster_map(self):

        ##
        ## Compute co-occurrence matrix
        ##
        self.apply()
        X = self.X_.copy()

        ##
        ## CA
        ##
        ca = CA()
        ca.fit(1 - X)
        X_transformed = ca.principal_coordinates_cols_
        x_axis = X_transformed.loc[:, X_transformed.columns[self.x_axis]]
        y_axis = X_transformed.loc[:, X_transformed.columns[self.y_axis]]

        X_transformed = pd.DataFrame(
            {"x_axis": x_axis, "y_axis": y_axis, "CLUSTER": self.labels_}
        )
        X_transformed = X_transformed.groupby(["CLUSTER"], as_index=True).mean()
        X_transformed = X_transformed.sort_index(axis=0)

        ##
        ## Cluster coordinates
        ##
        x_axis = X_transformed.x_axis.tolist()
        y_axis = X_transformed.y_axis.tolist()

        ##
        ## Cluster names
        ##
        labels = [
            "CLUST_{} {}".format(index, label)
            for index, label in enumerate(self.cluster_names_)
        ]

        return xy_clusters_plot(
            x=x_axis,
            y=y_axis,
            x_axis_at=0,
            y_axis_at=0,
            labels=labels,
            node_sizes=counters_to_node_sizes(labels),
            color_scheme=self.colors,
            xlabel="Dim-{}".format(self.x_axis),
            ylabel="Dim-{}".format(self.y_axis),
            figsize=(self.width, self.height),
        )

    def ca_keywords_by_cluster_table(self):
        self.apply()
        return self.cluster_members_

    def ca_keywords_by_cluster_list(self):
        self.apply()
        return cluster_table_to_list(self.cluster_members_)

    def ca_keywords_by_cluster_python_code(self):
        self.apply()
        return cluster_table_to_python_code(self.column, self.cluster_members_)

    ######

    def strategic_map(self):

        self.apply()

        strategic_map = self.strategic_map_.copy()

        strategic_map["node_sizes"] = strategic_map["Cluster name"].map(
            lambda w: w.split(" ")[-1]
        )
        strategic_map["node_sizes"] = strategic_map.node_sizes.map(
            lambda w: w.split(":")[0]
        )
        strategic_map["node_sizes"] = strategic_map.node_sizes.map(int)
        max_node_size = strategic_map.node_sizes.max()
        min_node_size = strategic_map.node_sizes.min()
        strategic_map["node_sizes"] = strategic_map.node_sizes.map(
            lambda w: 200 + 2800 * (w - min_node_size) / (max_node_size - min_node_size)
        )

        return xy_clusters_plot(
            x=strategic_map.Centrality,
            y=strategic_map.Density,
            x_axis_at=strategic_map.Centrality.median(),
            y_axis_at=strategic_map.Density.median(),
            labels=strategic_map["Cluster name"]
            .map(lambda w: " ".join(w.split(" ")[:-1]))
            .tolist(),
            node_sizes=strategic_map.node_sizes,
            color_scheme=self.colors,
            xlabel="Centrality",
            ylabel="Density",
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
            dash.Dropdown(
                options=[
                    "MDS Keywords Map",
                    "MDS Cluster Map",
                    "MDS Keywords by Cluster (table)",
                    "MDS Keywords by Cluster (list)",
                    "MDS Keywords by Cluster (Python code)",
                    "CA Keywords Map",
                    "CA Cluster Map",
                    "CA Keywords by Cluster (table)",
                    "CA Keywords by Cluster (list)",
                    "CA Keywords by Cluster (Python code)",
                    "Strategic Map",
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
            dash.HTML("Clustering"),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=50, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.random_state(),
            dash.HTML("CA diagram"),
            dash.x_axis(),
            dash.y_axis(),
            dash.HTML("Visualization"),
            dash.top_n(),
            dash.color_scheme(),
            dash.fig_width(),
            dash.fig_height(),
        ]

        self.n_components = 10

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
                # Clustering
                "clustering_method": self.command_panel[8],
                "n_clusters": self.command_panel[9],
                "affinity": self.command_panel[10],
                "linkage": self.command_panel[11],
                "random_state": self.command_panel[12],
                #  CA Diagram
                "x_axis": self.command_panel[14],
                "y_axis": self.command_panel[15],
                # Visualization
                "top_n": self.command_panel[17],
                "colors": self.command_panel[18],
                "width": self.command_panel[19],
                "height": self.command_panel[20],
            },
        )

        DASH.__init__(self)

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        with self.output:

            if self.menu in ["MDS Keywords Map", "MDS Cluster Map", "Strategic Map"]:

                self.set_disabled("X-axis:")
                self.set_disabled("Y-axis:")
                self.set_enabled("Colors:")
                self.set_enabled("Width:")
                self.set_enabled("Height:")

            if self.menu in ["CA Keywords Map", "CA Cluster Map"]:

                self.set_enabled("X-axis:")
                self.set_enabled("Y-axis:")
                self.set_enabled("Colors:")
                self.set_enabled("Width:")
                self.set_enabled("Height:")

            if self.menu in [
                "MDS Keywords by Cluster (table)",
                "MDS Keywords by Cluster (list)",
                "MDS Keywords by Cluster (Python code)",
                "CA Keywords by Cluster (table)",
                "CA Keywords by Cluster (list)",
                "CA Keywords by Cluster (Python code)",
            ]:
                self.set_disabled("X-axis:")
                self.set_disabled("Y-axis:")
                self.set_disabled("Colors:")
                self.set_disabled("Width:")
                self.set_disabled("Height:")


