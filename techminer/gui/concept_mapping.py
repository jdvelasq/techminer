import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

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
)
from techminer.plots import (
    ax_text_node_labels,
    counters_to_node_sizes,
    expand_ax_limits,
    set_spines_invisible,
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
            X=1 - X,
            method=self.clustering_method,
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            random_state=self.random_state,
            top_n=self.top_n,
            name_prefix="Cluster {}",
        )

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

    def cluster_members(self):
        self.apply()
        return self.cluster_members_

    def cluster_co_occurrence_matrix(self):
        self.apply()
        return self.cluster_co_occurrence_

    def cluster_plot(self, method):

        if method == "MDS":
            x_axis = 0
            y_axis = 1
        else:
            x_axis = self.x_axis
            y_axis = self.y_axis

        X = self.cluster_co_occurrence_

        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()

        ##
        ## Both methods use the dissimilitud matrix
        ##
        if method == "MDS":
            embedding = MDS(n_components=self.n_components)
            X_transformed = embedding.fit_transform(
                1 - X,
            )
            x_axis = X_transformed[:, x_axis]
            y_axis = X_transformed[:, y_axis]
        if method == "CA":
            ca = CA()
            ca.fit(1 - X)
            X_transformed = ca.principal_coordinates_cols_
            x_axis = X_transformed.loc[:, X_transformed.columns[x_axis]]
            y_axis = X_transformed.loc[:, X_transformed.columns[y_axis]]

        colors = []
        for cmap_name in ["tab20", "tab20b", "tab20c"]:
            cmap = pyplot.cm.get_cmap(cmap_name)
            colors += [cmap(0.025 + 0.05 * i) for i in range(20)]

        node_sizes = counters_to_node_sizes(X.columns)

        node_colors = [
            cmap(0.2 + 0.80 * t / (self.n_clusters - 1)) for t in range(self.n_clusters)
        ]

        ax.scatter(x_axis, y_axis, s=node_sizes, c=node_colors, alpha=0.5)

        expand_ax_limits(ax)

        pos = {term: (x_axis[idx], y_axis[idx]) for idx, term in enumerate(X.columns)}
        ax_text_node_labels(
            ax=ax, labels=X.columns, dict_pos=pos, node_sizes=node_sizes
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5, zorder=-1)

        ax.set_aspect("equal")
        ax.axis("off")
        set_spines_invisible(ax)

        fig.set_tight_layout(True)

        return fig

    def mds_cluster_map(self):
        self.apply()
        return self.cluster_plot(method="MDS")

    def ca_cluster_map(self):
        self.apply()
        return self.cluster_plot(method="CA")

    def centratlity_density_table(self):
        self.apply()
        return self.strategic_map_

    def strategic_map(self):
        ##
        self.apply()
        ##

        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()

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

        median_density = self.strategic_map_.Density.median()
        median_centrality = self.strategic_map_.Centrality.median()

        node_colors = ["tomato"] * len(self.strategic_map_)
        if self.colors == "4 Quadrants":

            strategic_map["node_colors"] = "red"

            strategic_map.at[
                (strategic_map.Centrality < median_centrality)
                & ((strategic_map.Density < median_density)),
                "node_colors",
            ] = "gold"

            strategic_map.at[
                (strategic_map.Centrality < median_centrality)
                & ((strategic_map.Density > median_density)),
                "node_colors",
            ] = "darkseagreen"

            strategic_map.at[
                (strategic_map.Centrality > median_centrality)
                & ((strategic_map.Density < median_density)),
                "node_colors",
            ] = "cornflowerblue"

        elif self.colors == "Groups":

            colors = []
            for cmap_name in ["tab20", "tab20b", "tab20c", "tab20", "tab20b", "tab20c"]:
                cmap = pyplot.cm.get_cmap(cmap_name)
                colors += [cmap(0.025 + 0.05 * i) for i in range(20)]
            strategic_map["node_colors"] = colors[0 : self.n_clusters]

        else:

            cmap = pyplot.cm.get_cmap(self.colors)

            strategic_map["node_colors"] = strategic_map["Cluster name"].map(
                lambda w: w.split(" ")[-1]
            )
            strategic_map["node_colors"] = strategic_map.node_colors.map(
                lambda w: w.split(":")[1]
            )
            strategic_map["node_colors"] = strategic_map.node_colors.map(int)

            max_node_color = strategic_map.node_colors.max()
            min_node_color = strategic_map.node_colors.min()
            strategic_map["node_colors"] = strategic_map.node_colors.map(
                lambda w: 0.2
                + 0.8 * (w - min_node_color) / (max_node_color - min_node_color)
            )
            strategic_map["node_colors"] = strategic_map.node_colors.map(cmap)

        ax.scatter(
            self.strategic_map_.Centrality,
            self.strategic_map_.Density,
            s=strategic_map.node_sizes.tolist(),
            c=strategic_map.node_colors.tolist(),
            alpha=0.4,
        )

        expand_ax_limits(ax)
        ax_text_node_labels(
            ax,
            labels=self.strategic_map_["Cluster name"],
            dict_pos={
                key: (c, d)
                for key, c, d in zip(
                    self.strategic_map_["Cluster name"],
                    self.strategic_map_.Centrality,
                    self.strategic_map_.Density,
                )
            },
            node_sizes=strategic_map.node_sizes.tolist(),
        )

        ax.axhline(
            y=median_density,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )
        ax.axvline(
            x=median_centrality,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )

        #  ax.set_aspect("equal")
        ax.axis("off")
        set_spines_invisible(ax)
        fig.set_tight_layout(True)

        return fig


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
        DASH.__init__(self)

        self.command_panel = [
            dash.dropdown(
                description="MENU:",
                options=[
                    "Cluster members",
                    "Cluster co-occurrence matrix",
                    "Centratlity-Density table",
                    "MDS cluster map",
                    "CA cluster map",
                    "Strategic map",
                ],
            ),
            dash.dropdown(
                description="Column:",
                options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.normalization(include_none=False),
            dash.separator(text="Clustering"),
            dash.clustering_method(),
            dash.n_clusters(m=3, n=50, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.random_state(),
            dash.separator(text="CA diagram"),
            dash.x_axis(),
            dash.y_axis(),
            dash.separator(text="Visualization"),
            dash.top_n(),
            dash.dropdown(
                description="Colors:",
                options=[
                    "4 Quadrants",
                    "Groups",
                    "Greys",
                    "Purples",
                    "Blues",
                    "Greens",
                    "Oranges",
                    "Reds",
                ],
            ),
            dash.fig_width(),
            dash.fig_height(),
        ]

        self.n_components = 10

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        with self.output:

            if self.menu in [
                "Cluster memberships",
                "Cluster co-occurrence matrix",
                "Centratlity-Density table",
            ]:
                self.set_disabled("Width:")
                self.set_disabled("Height:")
            else:
                self.set_enabled("Width:")
                self.set_enabled("Height:")

            if self.menu in [
                "CA cluster map",
            ]:
                self.set_enabled("X-axis:")
                self.set_enabled("Y-axis:")
            else:
                self.set_disabled("X-axis:")
                self.set_disabled("Y-axis:")

            if self.menu == "MDS cluster map":
                self.set_disabled("X-axis:")
                self.set_disabled("Y-axis:")

            if self.clustering_method in ["Affinity Propagation"]:
                self.set_disabled("N Clusters:")
                self.set_disabled("Affinity:")
                self.set_disabled("Linkage:")
                self.set_enabled("Random State:")

            if self.clustering_method in ["Agglomerative Clustering"]:
                self.set_enabled("N Clusters:")
                self.set_enabled("Affinity:")
                self.set_enabled("Linkage:")
                self.set_disabled("Random State:")

            if self.clustering_method in ["Birch"]:
                self.set_enabled("N Clusters:")
                self.set_disabled("Affinity:")
                self.set_disabled("Linkage:")
                self.set_disabled("Random State:")

            if self.clustering_method in ["DBSCAN"]:
                self.set_disabled("N Clusters:")
                self.set_disabled("Affinity:")
                self.set_disabled("Linkage:")
                self.set_disabled("Random State:")

            if self.clustering_method in ["Feature Agglomeration"]:
                self.set_enabled("N Clusters:")
                self.set_enabled("Affinity:")
                self.set_enabled("Linkage:")
                self.set_disabled("Random State:")

            if self.clustering_method in ["KMeans"]:
                self.set_enabled("N Clusters:")
                self.set_disabled("Affinity:")
                self.set_disabled("Linkage:")
                self.set_disabled("Random State:")

            if self.clustering_method in ["Mean Shift"]:
                self.set_disabled("N Clusters:")
                self.set_disabled("Affinity:")
                self.set_disabled("Linkage:")
                self.set_disabled("Random State:")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################
