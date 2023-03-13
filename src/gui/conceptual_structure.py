import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, TruncatedSVD
from sklearn.manifold import MDS
import techminer.core.dashboard as dash
from techminer.core import (
    Dashboard,
    CA,
    TF_matrix,
    TFIDF_matrix,
    add_counters_to_axis,
    clustering,
    corpus_filter,
    exclude_terms,
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
        TF_matrix_ = exclude_terms(
            data=TF_matrix_,
            axis=1
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

        ##
        ## Cluster filters
        ##
        self.generate_cluster_filters(terms=R.index, labels=self.labels_)

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
        TF_matrix_ = exclude_terms(
            data=TF_matrix_,
            axis=1
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
        TF_matrix_ = exclude_terms(            data=TF_matrix_,            axis=1        )

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

        ##
        ## Cluster filters
        ##
        self.generate_cluster_filters(terms=R.index, labels=self.labels_)

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


RANDOM_STATE=[
            "0012345",
            "0123456",
            "0234567",
            "0345678",
            "0456789",
            "0567890",
            "0678901",
            "0789012",
            "0890123",
            "0901234",
            "1012345",
            "1123456",
            "1234567",
            "1345678",
            "1456789",
            "1567890",
            "1678901",
            "1789012",
            "1890123",
            "1901234",
            "2012345",
            "2123456",
            "2234567",
            "2345678",
            "2456789",
            "2567890",
            "2678901",
            "2789012",
            "2890123",
            "2901234",
            "3012345",
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
        

        self.command_panel = [
            widgets.HTML("<b>Display:</b>"),
            widgets.Dropdown(
                options=[
                    "Keywords Map",
                    "Keywords by Cluster (table)",
                    "Keywords by Cluster (list)",
                    "Keywords by Cluster (Python code)",
                    "Keywords coverage",
                ],
                layout=Layout(width="auto"),
            ),
            widgets.HTML(
                "<hr><b>Parameters:</b>",
                layout=Layout(margin="20px 0px 0px 0px"),
            ),
            widgets.Dropdown(
                description="Method:",
                options=[
                    "Multidimensional Scaling",
                    "PCA",
                    "Truncated SVD",
                    "Factor Analysis",
                    "Correspondence Analysis",
                ],
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.Dropdown(
                description="Normalization:",
                options=[
                    "Association",
                    "Jaccard",
                    "Dice",
                    "Salton/Cosine",
                    "Equivalence",
                    "Inclusion",
                    "Mutual Information",
                ],
                layout=Layout(width="auto"),
                value="Association",
                style={"description_width": "130px"},
            ),
            widgets.Dropdown(
                description="Column:",
                options=sorted(data.columns),
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.Dropdown(
                description="Min OCC:",
                options=list(range(1, 21)),
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.Dropdown(
                options=list(range(5, 40, 1))
                + list(range(40, 100, 5))
                + list(range(100, 3001, 100)),
                description="Max items:",
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.Dropdown(
                description="Random State:",
                options=RANDOM_STATE,
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.HTML(
                "<hr><b>Clustering:</b>",
                layout=Layout(margin="20px 0px 0px 0px"),
            ),
            widgets.Dropdown(
                description="Clustering Method:",
                options=[
                    "Affinity Propagation",
                    "Agglomerative Clustering",
                    "Birch",
                    "DBSCAN",
                    "KMeans",
                    "Mean Shift",
                ],
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.Dropdown(
                description="N Clusters:",
                options=list(range(3, 11, 1)),
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.Dropdown(
                description="Affinity",
                options=["euclidean", "l1", "l2", "manhattan", "cosine"],
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.Dropdown(
                description="Linkage:",
                options=["ward", "complete", "average", "single"],
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.HTML(
                "<hr><b>Visualization:</b>",
                layout=Layout(margin="20px 0px 0px 0px"),
            ),
            widgets.Dropdown(
                description="Top N:",
                options=list(range(10, 51, 1)),
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
                
            ),
            widgets.Dropdown(
                description="Width:",
                options=range(5, 26, 1),
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
            widgets.Dropdown(
                description="Height:",
                options=range(5, 26, 1),
                layout=Layout(width="auto"),
                style={"description_width": "130px"},
            ),
        ]
        
        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                "menu": self.command_panel[1],
                "method": self.command_panel[3],
                "normalization": self.command_panel[4],
                "column": self.command_panel[5],
                "min_occ": self.command_panel[6],
                "max_items": self.command_panel[7],
                "random_state": self.command_panel[8],
                "clustering_method": self.command_panel[10],
                "n_clusters": self.command_panel[11],
                "affinity": self.command_panel[12],
                "linkage": self.command_panel[13],
                "top_n": self.command_panel[15],
                "width": self.command_panel[16],
                "height": self.command_panel[17],
            },
        )

        Dashboard.__init__(self)

        self.interactive_output(
            **{
                "menu": self.command_panel[1].value,
                "method": self.command_panel[3].value,
                "normalization": self.command_panel[4].value,
                "column": self.command_panel[5].value,
                "min_occ": self.command_panel[6].value,
                "max_items": self.command_panel[7].value,
                "random_state": self.command_panel[8].value,
                "clustering_method": self.command_panel[10].value,
                "n_clusters": self.command_panel[11].value,
                "affinity": self.command_panel[12].value,
                "linkage": self.command_panel[13].value,
                "top_n": self.command_panel[15].value,
                "width": self.command_panel[16].value,
                "height": self.command_panel[17].value,
            }
        )

    def interactive_output(self, **kwargs):

        with self.output:
            Dashboard.interactive_output(self, **kwargs)

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
    return App(
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
    ).run()
