import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import ipywidgets as widgets

import techminer.core.dashboard as dash
from techminer.core import (
    Dashboard,
    Network,
    TF_matrix,
    add_counters_to_axis,
    corpus_filter,
    exclude_terms,
    sort_by_axis,
)
from techminer.core.dashboard import min_occurrence
from techminer.plots import (
    ChordDiagram,
    bubble_plot,
    counters_to_node_colors,
    counters_to_node_sizes,
    heatmap,
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

        x = self.data.copy()

        if self.column == self.by:

            ##
            ##  Drop NA from column
            ##
            w = x[[self.column, "ID"]].dropna()

            ##
            ##  Computes TF_matrix with occurrence >= min_occurrence
            ##
            A = TF_matrix(
                data=w,
                column=self.column,
                scheme=None,
                min_occurrence=self.min_occ,
            )

            ##
            ##  Exclude Terms
            ##
            A = exclude_terms(data=A, axis=1)

            ##
            ##   Select max_items
            ##
            A = add_counters_to_axis(X=A, axis=1, data=self.data, column=self.column)
            A = sort_by_axis(data=A, sort_by=self.top_by, ascending=False, axis=1)
            A = A[A.columns[: self.max_items]]
            if len(A.columns) > self.max_items:
                top_items = A.sum(axis=0)
                top_items = top_items.sort_values(ascending=False)
                top_items = top_items.head(self.max_items)
                A = A.loc[:, top_items.index]
                rows = A.sum(axis=1)
                rows = rows[rows > 0]
                A = A.loc[rows.index, :]

            ##
            ##  Computes correlation
            ##
            matrix = A.corr(method=self.method)

        else:

            ##
            ##  Drop NA from column
            ##
            w = x[[self.column, self.by, "ID"]].dropna()

            ##
            ##  Computes TF_matrix with occurrence >= min_occurrence
            ##
            A = TF_matrix(data=w, column=self.column, scheme=None)

            ##
            ##  Exclude Terms
            ##
            A = exclude_terms(data=A, axis=1)

            ##
            ##  Minimal occurrence
            ##
            terms = A.sum(axis=0)
            terms = terms.sort_values(ascending=False)
            terms = terms[terms >= self.min_occ]
            A = A.loc[:, terms.index]

            ##
            ##  Select max_items
            ##
            A = add_counters_to_axis(X=A, axis=1, data=self.data, column=self.column)
            A = sort_by_axis(data=A, sort_by=self.top_by, ascending=False, axis=1)
            if len(A.columns) > self.max_items:
                A = A[A.columns[: self.max_items]]

            ##
            ##  Computes correlation
            ##
            B = TF_matrix(w, column=self.by, scheme=None)
            matrix = np.matmul(B.transpose().values, A.values)
            matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)
            matrix = matrix.corr(method=self.method)

        matrix = sort_by_axis(
            data=matrix,
            sort_by=self.sort_r_axis_by,
            ascending=self.r_axis_ascending,
            axis=0,
        )

        matrix = sort_by_axis(
            data=matrix,
            sort_by=self.sort_c_axis_by,
            ascending=self.c_axis_ascending,
            axis=1,
        )
        self.X_ = matrix

    def matrix(self):
        self.apply()
        return self.X_.style.format("{:+4.3f}").background_gradient(
            cmap=self.colormap, axis=None
        )

    def heatmap(self):
        self.apply()
        return heatmap(self.X_, cmap=self.colormap, figsize=(self.width, self.height))

    def bubble_plot(self):
        self.apply()
        return bubble_plot(
            self.X_,
            axis=0,
            cmap=self.colormap,
            figsize=(self.width, self.height),
        )

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

        d = {
            0: {"linestyle": "-", "linewidth": 4, "color": "black"},
            1: {"linestyle": "-", "linewidth": 2, "color": "black"},
            2: {"linestyle": "--", "linewidth": 1, "color": "gray"},
            3: {"linestyle": ":", "linewidth": 1, "color": "lightgray"},
        }

        for idx in range(len(m)):

            key = (
                0
                if m.link_[idx] > 0.75
                else (1 if m.link_[idx] > 0.50 else (2 if m.link_[idx] > 0.25 else 3))
            )

            cd.add_edge(m.from_[idx], m.to_[idx], **(d[key]))

        return cd.plot(figsize=(self.width, self.height))

    def correlation_map_nx(self):
        self.apply()
        return Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).networkx_plot(
            layout=self.layout,
            iterations=self.nx_iterations,
            figsize=(self.width, self.height),
        )

    def communities(self):
        self.fit()
        return Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).cluster_members_

    def correlation_map_interactive(self):
        self.apply()
        return Network(
            X=self.X_,
            top_by=self.top_by,
            n_labels=self.n_labels,
            clustering=self.clustering,
        ).pyvis_plot()


###############################################################################
##
##  DASHBOARD
##
###############################################################################


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
                    "Correlation map (nx)",
                    "Chord diagram",
                ],
            ),
            dash.HTML("Parameters:"),
            dash.Dropdown(description="Column:", options=COLUMNS),
            dash.Dropdown(description="By:", options=COLUMNS),
            dash.Dropdown(
                description="Method:", options=["pearson", "kendall", "spearman"]
            ),
            dash.min_occurrence(),
            dash.max_items(),
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
                ],
            ),
            dash.c_axis_ascending(),
            dash.Dropdown(
                description="Sort R-axis by:",
                options=[
                    "Alphabetic",
                    "Num Documents",
                    "Global Citations",
                ],
            ),
            dash.r_axis_ascending(),
            dash.cmap(),
            dash.nx_layout(),
            dash.n_labels(),
            dash.nx_iterations(),
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
                "by": self.command_panel[4],
                "method": self.command_panel[5],
                "min_occ": self.command_panel[6],
                "max_items": self.command_panel[7],
                "clustering": self.command_panel[8],
                # Visualization
                "top_by": self.command_panel[10],
                "sort_c_axis_by": self.command_panel[11],
                "c_axis_ascending": self.command_panel[12],
                "sort_r_axis_by": self.command_panel[13],
                "r_axis_ascending": self.command_panel[14],
                "colormap": self.command_panel[15],
                "layout": self.command_panel[16],
                "n_labels": self.command_panel[17],
                "nx_iterations": self.command_panel[18],
                "width": self.command_panel[19],
                "height": self.command_panel[20],
            },
        )

        Dashboard.__init__(self)

    def interactive_output(self, **kwargs):

        Dashboard.interactive_output(self, **kwargs)

        if self.menu in [
            "Matrix",
            "Heatmap",
            "Bubble plot",
        ]:
            self.set_enabled("Sort C-axis by:")
            self.set_enabled("C-axis ascending:")
            self.set_enabled("Sort R-axis by:")
            self.set_enabled("R-axis ascending:")
        else:
            self.set_disabled("Sort C-axis by:")
            self.set_disabled("C-axis ascending:")
            self.set_disabled("Sort R-axis by:")
            self.set_disabled("R-axis ascending:")

        if self.menu == "Correlation map (nx)":
            self.set_enabled("Layout:")
            self.set_enabled("N labels:")
        else:
            self.set_disabled("Layout:")
            self.set_disabled("N labels:")

        if self.menu == "Correlation map" and self.layout == "Spring":
            self.set_enabled("nx iterations:")
        else:
            self.set_disabled("nx iterations:")

        if self.menu in ["Matrix", "Correlation map (interactive)"]:
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


def correlation_analysis(
    limit_to=None,
    exclude=None,
    years_range=None,
):

    return App(
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
    ).run()
