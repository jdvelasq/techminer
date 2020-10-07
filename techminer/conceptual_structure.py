import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.manifold import MDS
import techminer.core.dashboard as dash
from techminer.core import (
    DASH,
    CA,
    TF_matrix,
    TFIDF_matrix,
    add_counters_to_axis,
    clustering,
    corpus_filter,
    limit_to_exclude,
    normalize_network,
    sort_axis,
    cluster_table_to_list,
    cluster_table_to_python_code,
    keywords_coverage,
)
from techminer.plots import (
    conceptual_structure_map,
    counters_to_node_sizes,
    xy_clusters_plot,
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

        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

        self.affinity = None
        self.clustering_method = None
        self.column = None
        self.height = None
        self.linkage = None
        self.max_items = None
        self.method = None
        self.min_occ = None
        self.normalization = None
        self.random_state = None
        self.top_n = None
        self.width = None

    def apply_other_methods(self):

        ##
        ## Conceptual Structure Map
        ##

        X = self.data.copy()
        
        ##
        ## Compute TF matrix
        ##
        TF_matrix_ = TF_matrix(
            data=X,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occ,
        )

        ##
        ## Limit to / Exclude
        ##
        TF_matrix_ = limit_to_exclude(
            data=TF_matrix_,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ##
        ## Add counters to axes
        ##
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        ##
        ## Select top terms
        ##
        TF_matrix_ = sort_axis(
            data=TF_matrix_, num_documents=True, axis=1, ascending=False
        )
        if len(TF_matrix_.columns) > self.max_items:
            TF_matrix_ = TF_matrix_.loc[:, TF_matrix_.columns[0 : self.max_items]]
            rows = TF_matrix_.sum(axis=1)
            rows = rows[rows > 0]
            TF_matrix_ = TF_matrix_.loc[rows.index, :]

        ##
        ## Co-occurrence matrix and normalization using Association Index
        ##
        M = np.matmul(TF_matrix_.transpose().values, TF_matrix_.values)
        M = pd.DataFrame(M, columns=TF_matrix_.columns, index=TF_matrix_.columns)
        M = normalize_network(M, normalization=self.normalization)

        ##
        ## Dissimilarity matrix
        ##
        M = 1 - M
        for i in M.columns.tolist():
            M.at[i, i] = 0.0

        ##
        ## Decomposition
        ##
        model = {
            "Multidimensional Scaling": MDS(
                n_components=2,
                random_state=int(self.random_state),
                dissimilarity="precomputed",
            ),
            "Truncated SVD": TruncatedSVD(
                n_components=2, random_state=int(self.random_state)
            ),
            "Factor Analysis": FactorAnalysis(
                n_components=2, random_state=int(self.random_state)
            ),
            "PCA": PCA(n_components=2, random_state=int(self.random_state)),
        }[self.method]

        R = model.fit_transform(X=M.values)
        R = pd.DataFrame(
            R,
            columns=["Dim-{}".format(i) for i in range(2)],
            index=M.columns,
        )

        ##
        ## Clustering
        ##
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

        R["Cluster"] = self.labels_

        self.coordinates_ = R

        ##
        ## Results
        ##
        self.X_ = R

    def keywords_map(self):
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

    def keywords_by_cluster_table(self):
        self.apply()
        return self.cluster_members_

    def keywords_by_cluster_list(self):
        self.apply()
        return cluster_table_to_list(self.cluster_members_)

    def keywords_by_cluster_python_code(self):
        self.apply()
        return cluster_table_to_python_code(self.column, self.cluster_members_)

    def keywords_coverage(self):

        X = self.data.copy()

        ##
        ## Compute TF matrix
        ##
        TF_matrix_ = TF_matrix(
            data=X,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occ,
        )

        ##
        ## Limit to / Exclude
        ##
        TF_matrix_ = limit_to_exclude(
            data=TF_matrix_,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ##
        ## Add counters to axes
        ##
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        ##
        ## Select top terms
        ##
        TF_matrix_ = sort_axis(
            data=TF_matrix_, num_documents=True, axis=1, ascending=False
        )
        if len(TF_matrix_.columns) > self.max_items:
            TF_matrix_ = TF_matrix_.loc[:, TF_matrix_.columns[0 : self.max_items]]

        ##
        ## Keywords list
        ##
        keywords_list = TF_matrix_.columns

        return keywords_coverage(
            data=self.data, column=self.column, keywords_list=keywords_list
        )

    def apply(self):

        if self.method in [
            "Multidimensional Scaling",
            "Truncated SVD",
            "Factor Analysis",
            "PCA",
        ]:
            self.apply_other_methods()

        if self.method == "Correspondence Analysis":
            self.apply_ca()

    def apply_ca(self):

        ##
        ## Based on comparative analysis methodology
        ##   from https://tlab.it/en/allegati/help_en_online/mcluster.htm
        ##

        ##
        ## Computes TF matrix for terms in min_occurrence
        ##
        TF_matrix_ = TF_matrix(
            data=self.data,
            column=self.column,
            scheme="binary",
            min_occurrence=self.min_occ,
        )

        ##
        ## Limit to / Exclude
        ##
        TF_matrix_ = limit_to_exclude(
            data=TF_matrix_,
            axis=1,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ##
        ## Computtes TFIDF matrix and select max_term frequent terms
        ##
        ##   tf-idf = tf * (log(N / df) + 1)
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
        ## Adds counter to axies
        ##
        TFIDF_matrix_ = add_counters_to_axis(
            X=TFIDF_matrix_, axis=1, data=self.data, column=self.column
        )

        ##
        ## Correspondence Analysis
        ## 2 first factors for ploting
        ##
        ca = CA()
        ca.fit(TFIDF_matrix_)

        R = ca.principal_coordinates_cols_
        R = R[R.columns[:2]]
        R = pd.DataFrame(
            R,
            columns=["Dim-{}".format(i) for i in range(2)],
            index=TFIDF_matrix_.columns,
        )

        ##
        ## Clustering
        ##
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

        R["Cluster"] = self.labels_

        self.coordinates_ = R

        ##
        ## Results
        ##
        self.X_ = R


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
                    "Keywords Map",
                    "Keywords by Cluster (table)",
                    "Keywords by Cluster (list)",
                    "Keywords by Cluster (Python code)",
                    "Keywords coverage",
                ],
            ),
            dash.dropdown(
                description="Method:",
                options=[
                    "Multidimensional Scaling",
                    "PCA",
                    "Truncated SVD",
                    "Factor Analysis",
                    "Correspondence Analysis",
                ],
            ),
            dash.normalization(include_none=False),
            dash.dropdown(
                description="Column:",
                options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.random_state(),
            dash.separator(text="Clustering"),
            dash.clustering_method(),
            dash.n_clusters(m=2, n=11, i=1),
            dash.affinity(),
            dash.linkage(),
            dash.separator(text="Visualization"),
            dash.top_n(),
            dash.fig_width(),
            dash.fig_height(),
        ]
        

    def interactive_output(self, **kwargs):

        with self.output:
            DASH.interactive_output(self, **kwargs)

        if self.menu in ["Keywords Map"] and self.method in [
            "Multidimensional Scaling",
            "PCA",
            "Truncated SVD",
            "Factor Analysis",
        ]:

            self.set_enabled("Method:")
            self.set_enabled("Normalization:")
            self.set_enabled("Min occurrence:") 
            self.set_enabled("Max items:") 
            self.set_enabled("Random State:")
            self.set_enabled("Clustering Method:")
            self.set_enabled("Top N:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

            self.enable_disable_clustering_options()


        if self.menu in ["Keywords Map"] and self.method in ["Correspondence Analysis"]:

            self.set_enabled("Method:")
            self.set_disabled("Normalization:")
            self.set_enabled("Min occurrence:") 
            self.set_enabled("Max items:") 
            self.set_enabled("Random State:")
            self.set_enabled("Clustering Method:")
            self.set_enabled("Top N:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

            self.enable_disable_clustering_options()

        if self.menu in [ 
            "Keywords by Cluster (table)",
            "Keywords by Cluster (list)",
            "Keywords by Cluster (Python code)"
            ] and self.method in [
            "Multidimensional Scaling",
            "PCA",
            "Truncated SVD",
            "Factor Analysis",
            ]:

                self.set_enabled("Method:")
                self.set_enabled("Normalization:")
                self.set_enabled("Min occurrence:") 
                self.set_enabled("Max items:") 
                self.set_enabled("Random State:")
                self.set_enabled("Clustering Method:")
                self.set_disabled("Top N:")
                self.set_disabled("Width:")
                self.set_disabled("Height:")

                self.enable_disable_clustering_options()

        if self.menu in [
            "Keywords by Cluster (table)",
            "Keywords by Cluster (list)",
            "Keywords by Cluster (Python code)"] and self.method in ["Correspondence Analysis"]:

                self.set_enabled("Method:")
                self.set_disabled("Normalization:")
                self.set_enabled("Min occurrence:") 
                self.set_enabled("Max items:") 
                self.set_enabled("Random State:")
                self.set_enabled("Clustering Method:")
                self.set_disabled("Top N:")
                self.set_disabled("Width:")
                self.set_disabled("Height:")

                self.enable_disable_clustering_options()


        if self.menu in ["Keywords coverage"]:

            self.set_disabled("Method:")
            self.set_disabled("Normalization:")
            self.set_disabled("Min occurrence:") 
            self.set_disabled("Max items:") 
            self.set_disabled("Random State:")
            self.set_disabled("Clustering Method:")
            self.set_disabled("Top N:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def conceptual_structure(
    limit_to=None,
    exclude=None,
    years_range=None,
):
    return DASHapp(
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
    ).run()
