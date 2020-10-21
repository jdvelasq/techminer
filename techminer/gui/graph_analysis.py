import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from pyvis.network import Network

import techminer.core.dashboard as dash
from techminer.core import (
    DASH,
    Network,
    TF_matrix,
    add_counters_to_axis,
    corpus_filter,
    exclude_terms,
    normalize_network,
    sort_by_axis,
)

from techminer.core import cluster_table_to_list
from techminer.plots import ChordDiagram
from techminer.plots import bubble_plot as bubble_plot_
from techminer.plots import counters_to_node_colors, counters_to_node_sizes
from techminer.plots import heatmap as heatmap_
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
        #
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

        self.X_ = None
        self.c_axis_ascending = None
        self.clustering = None
        self.colormap = None
        self.column = None
        self.height = None
        self.layout = None
        self.max_nodes = None
        self.normalization = None
        self.nx_iterations = None
        self.r_axis_ascending = None
        self.sort_c_axis_by = None
        self.sort_r_axis_by = None
        self.top_by = None
        self.width = None
        self.min_occ = None
        self.max_items = None
        self.n_labels = None

    def apply(self):

        ##
        ##  Computes TF_matrix with occurrence >= min_occurrence
        ##
        TF_matrix_ = TF_matrix(
            data=self.data,
            column=self.column,
            scheme=None,
            min_occurrence=self.min_occ,
        )

        ##
        ##  Exclude Terms
        ##
        TF_matrix_ = exclude_terms(data=TF_matrix_, axis=1)

        ##
        ##  Adds counters to axis
        ##
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )

        TF_matrix_ = sort_by_axis(
            data=TF_matrix_, sort_by=self.top_by, ascending=False, axis=1
        )

        ##
        ##  Select max_items
        ##
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
        X = normalize_network(X, self.normalization)

        ##
        ##  Sort by
        ##
        X = sort_by_axis(
            data=X, sort_by=self.sort_r_axis_by, ascending=self.r_axis_ascending, axis=0
        )
        X = sort_by_axis(
            data=X, sort_by=self.sort_c_axis_by, ascending=self.c_axis_ascending, axis=1
        )

        self.X_ = X

    def matrix(self):
        self.apply()
        if self.normalization == "None":
            return self.X_.style.background_gradient(cmap=self.colormap, axis=None)
        else:
            return self.X_.style.set_precision(2).background_gradient(
                cmap=self.colormap, axis=None
            )

    def heatmap(self):
        self.apply()
        return heatmap_(self.X_, cmap=self.colormap, figsize=(self.width, self.height))

    def bubble_plot(self):
        self.apply()
        return bubble_plot_(
            self.X_, axis=0, cmap=self.colormap, figsize=(self.width, self.height)
        )

    def to_cluster_filters(self, table):
        terms = []
        labels = []
        for i_cluster, cluster in enumerate(table.columns):
            x = table[cluster].tolist()
            x = [m for m in x if m.strip() != ""]
            x = [" ".join(m.split(" ")[:-1]) for m in x]
            terms += x
            labels += [i_cluster] * len(x)

        self.generate_cluster_filters(terms=terms, labels=labels)

    def network(self):
        self.apply()
        return Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).networkx_plot(
            layout=self.layout,
            iterations=self.nx_iterations,
            k=self.nx_k,
            scale=self.nx_scale,
            seed=int(self.random_state),
            figsize=(self.width, self.height),
        )

    def communities_table(self):
        self.apply()
        table = Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).cluster_members_
        self.to_cluster_filters(table)
        return table

    def communities_list(self):
        self.apply()
        table = Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).cluster_members_
        self.to_cluster_filters(table)
        return cluster_table_to_list(table=table)

    def communities_python_code(self):
        self.apply()
        members = Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).cluster_members_

        self.to_cluster_filters(members)

        dict_ = {}
        for i_cluster, cluster in enumerate(members.columns):

            x = members[cluster].tolist()
            x = [m for m in x if m.strip() != ""]
            x = [" ".join(m.split(" ")[:-1]) for m in x]

            dict_[i_cluster] = x

        HTML = "CLUSTERS = [<br>"
        HTML += '    "' + self.column + '",<br>'
        HTML += "    {<br>"
        for key in dict_.keys():
            HTML += "        " + str(key) + ": [<br>"
            for value in dict_[key]:
                HTML += "            " + repr(value) + ",<br>"
            HTML += "        ],<br>"
        HTML += "    }<br>"
        HTML += "]"
        return widgets.HTML("<pre>" + HTML + "</pre>")

    def network_interactive(self):

        self.apply()
        return Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).pyvis_plot()

    def chord_diagram(self):
        self.apply()
        x = self.X_.copy()
        terms = self.X_.columns.tolist()
        node_sizes = counters_to_node_sizes(x=terms)
        node_colors = counters_to_node_colors(x, cmap=pyplot.cm.get_cmap(self.colormap))

        cd = ChordDiagram()

        ## add nodes
        for idx, term in enumerate(x.columns):
            cd.add_node(term, color=node_colors[idx], s=node_sizes[idx])

        ## add links
        m = x.stack().to_frame().reset_index()
        m = m[m.level_0 < m.level_1]
        m.columns = ["from_", "to_", "link_"]
        m = m.reset_index(drop=True)

        for idx in range(len(m)):

            if m.link_[idx] > 0.001:
                d = {
                    "linestyle": "-",
                    "linewidth": 0.0
                    + 2
                    * (m.link_[idx] - m.link_.min())
                    / (m.link_.max() - m.link_.min()),
                    "color": "gray",
                }

                cd.add_edge(m.from_[idx], m.to_[idx], **d)

        return cd.plot(figsize=(self.width, self.height))


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class App(DASH, Model):
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

        COLUMNS = sorted(
            [
                column
                for column in data.columns
                if column
                not in [
                    "Abb_Source_Title",
                    "Abstract",
                    "Affiliations",
                    "Authors_ID",
                    "Bradford_Law_Zone",
                    "Document_Type",
                    "Frac_Num_Documents",
                    "Global_Citations",
                    "Global_References",
                    "Historiograph_ID",
                    "ID",
                    "Local_Citations",
                    "Local_References",
                    "Num_Authors",
                    "Source_Title",
                    "Title",
                    "Year",
                ]
            ]
        )

        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.Dropdown(
                options=[
                    "Matrix",
                    "Heatmap",
                    "Bubble plot",
                    "Network",
                    "Chord diagram",
                    "Communities (table)",
                    "Communities (list)",
                    "Communities (Python code)",
                ],
            ),
            dash.HTML("Parameters:"),
            dash.Dropdown(
                description="Column:",
                options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.normalization(),
            dash.network_clustering(),
            dash.HTML("Visualization:"),
            dash.Dropdown(
                description="Top by:",
                options=[
                    "Num Documents",
                    "Global Citations",
                ],
            ),
            dash.Dropdown(
                description="Sort C-axis by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Global Citations",
                    "Data",
                ],
            ),
            dash.c_axis_ascending(),
            dash.Dropdown(
                description="Sort R-axis by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Global Citations",
                    "Data",
                ],
            ),
            dash.r_axis_ascending(),
            dash.cmap(),
            dash.n_labels(),
            dash.nx_iterations(),
            dash.nx_k(),
            dash.nx_scale(),
            dash.random_state(),
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
                "clustering": self.command_panel[7],
                # Visualization
                "top_by": self.command_panel[9],
                "sort_c_axis_by": self.command_panel[10],
                "c_axis_ascending": self.command_panel[11],
                "sort_r_axis_by": self.command_panel[12],
                "r_axis_ascending": self.command_panel[13],
                "colormap": self.command_panel[14],
                "n_labels": self.command_panel[15],
                "nx_iterations": self.command_panel[16],
                "nx_k": self.command_panel[17],
                "nx_scale": self.command_panel[18],
                "random_state": self.command_panel[19],
                "width": self.command_panel[20],
                "height": self.command_panel[21],
            },
        )

        DASH.__init__(self)

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)

        if self.menu == "Matrix":

            self.set_disabled("Clustering:")

            self.set_enabled("Top by:")
            self.set_enabled("Sort C-axis by:")
            self.set_enabled("C-axis ascending:")
            self.set_enabled("Sort R-axis by:")
            self.set_enabled("R-axis ascending:")
            self.set_disabled("Colormap:")
            self.set_disabled("N labels:")
            self.set_disabled("NX iterations:")
            self.set_disabled("NX K:")
            self.set_disabled("NX scale:")
            self.set_disabled("Random State:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        if self.menu in ["Heatmap", "Bubble plot"]:

            self.set_disabled("Clustering:")

            self.set_enabled("Top by:")
            self.set_enabled("Sort C-axis by:")
            self.set_enabled("C-axis ascending:")
            self.set_enabled("Sort R-axis by:")
            self.set_enabled("R-axis ascending:")

            self.set_enabled("Colormap:")
            self.set_disabled("N labels:")
            self.set_disabled("NX iterations:")
            self.set_disabled("NX K:")
            self.set_disabled("NX scale:")
            self.set_enabled("Random State:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        if self.menu in ["Chord diagram"]:

            self.set_disabled("Clustering:")

            self.set_enabled("Top by:")
            self.set_disabled("Sort C-axis by:")
            self.set_disabled("C-axis ascending:")
            self.set_disabled("Sort R-axis by:")
            self.set_disabled("R-axis ascending:")

            self.set_enabled("Colormap:")
            self.set_disabled("N labels:")
            self.set_disabled("NX iterations:")
            self.set_disabled("NX K:")
            self.set_disabled("NX scale:")
            self.set_disabled("Random State:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        if self.menu in ["Network"]:

            self.set_enabled("Clustering:")

            self.set_disabled("Top by:")
            self.set_disabled("Sort C-axis by:")
            self.set_disabled("C-axis ascending:")
            self.set_disabled("Sort R-axis by:")
            self.set_disabled("R-axis ascending:")

            self.set_enabled("Colormap:")
            self.set_enabled("NX iterations:")
            self.set_enabled("NX K:")
            self.set_enabled("NX scale:")
            self.set_enabled("Random State:")
            self.set_enabled("N labels:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")

        if self.menu in [
            "Communities (list)",
            "Communities (table)",
            "Communities (Python code)",
        ]:

            self.set_enabled("Clustering:")

            self.set_disabled("Top by:")
            self.set_disabled("Sort C-axis by:")
            self.set_disabled("C-axis ascending:")
            self.set_disabled("Sort R-axis by:")
            self.set_disabled("R-axis ascending:")

            self.set_disabled("Colormap:")
            self.set_disabled("N labels:")
            self.set_disabled("NX iterations:")
            self.set_disabled("NX K:")
            self.set_disabled("NX scale:")
            self.set_disabled("Random State:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        if self.menu == "Network (interactive)":

            self.set_enabled("Clustering:")

            self.set_disabled("Top by:")
            self.set_disabled("Sort C-axis by:")
            self.set_disabled("C-axis ascending:")
            self.set_disabled("Sort R-axis by:")
            self.set_disabled("R-axis ascending:")

            self.set_disabled("Colormap:")
            self.set_enabled("N labels:")
            self.set_disabled("NX iterations:")
            self.set_disabled("NX K:")
            self.set_disabled("NX scale:")
            self.set_disabled("Random State:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")
            self.set_disabled("nx iterations:")


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def graph_analysis(
    limit_to=None,
    exclude=None,
):
    return App(
        limit_to=limit_to,
        exclude=exclude,
    ).run()
