import warnings

import matplotlib
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import (
    Dashboard,
    TF_matrix,
    add_counters_to_axis,
    corpus_filter,
    exclude_terms,
    sort_by_axis,
)

from techminer.plots import (
    ax_text_node_labels,
    bubble_plot,
    counters_to_node_colors,
    counters_to_node_sizes,
    expand_ax_limits,
)
from techminer.plots import heatmap as heatmap_
from techminer.plots import set_spines_invisible
from techminer.core.filter_records import filter_records

warnings.filterwarnings("ignore", category=UserWarning)

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

        self.clusters = clusters
        self.cluster = cluster
        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

        self.by = None
        self.c_axis_ascending = None
        self.colormap_col = None
        self.colormap_by = None
        self.column = None
        self.height = None
        self.layout = None
        self.max_items_by = None
        self.max_items_col = None
        self.min_occ_by = None
        self.min_occ_col = None
        self.r_axis_ascending = None
        self.sort_c_axis_by = None
        self.sort_r_axis_by = None
        self.top_by = None
        self.width = None
        self.nx_iterations = None

    def apply(self):

        if self.column == self.by:
            self.X_ = None
            return

        W = self.data[[self.column, self.by, "ID"]].dropna()
        A = TF_matrix(data=W, column=self.column)
        B = TF_matrix(data=W, column=self.by)

        if self.top_by == "Data":

            A = exclude_terms(data=A, axis=1)
            B = exclude_terms(data=B, axis=1)
            matrix = np.matmul(B.transpose().values, A.values)
            matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)

            # sort max values per column
            max_columns = matrix.sum(axis=0)
            max_columns = max_columns.sort_values(ascending=False)
            max_columns = max_columns.head(self.max_items_col).index

            max_index = matrix.sum(axis=1)
            max_index = max_index.sort_values(ascending=False)
            max_index = max_index.head(self.max_items_by).index

            matrix = matrix.loc[
                [t for t in matrix.index if t in max_index],
                [t for t in matrix.columns if t in max_columns],
            ]

            matrix = add_counters_to_axis(
                X=matrix, axis=1, data=self.data, column=self.column
            )
            matrix = add_counters_to_axis(
                X=matrix, axis=0, data=self.data, column=self.by
            )

            columns = [
                column
                for column in matrix.columns
                if int(column.split(" ")[-1].split(":")[0]) >= self.min_occ_col
            ]
            index = [
                index
                for index in matrix.index
                if int(index.split(" ")[-1].split(":")[0]) >= self.min_occ_by
            ]
            matrix = matrix.loc[index, columns]

            self.X_ = matrix

        if self.top_by in ["Num Documents", "Global Citations"]:

            A = exclude_terms(data=A, axis=1)
            B = exclude_terms(data=B, axis=1)

            matrix = np.matmul(B.transpose().values, A.values)
            matrix = pd.DataFrame(matrix, columns=A.columns, index=B.columns)
            matrix = add_counters_to_axis(
                X=matrix, axis=1, data=self.data, column=self.column
            )
            matrix = add_counters_to_axis(
                X=matrix, axis=0, data=self.data, column=self.by
            )

            matrix = sort_by_axis(
                data=matrix, sort_by=self.top_by, ascending=False, axis=1
            )
            matrix = sort_by_axis(
                data=matrix, sort_by=self.top_by, ascending=False, axis=0
            )

            columns = matrix.columns[: self.max_items_col]
            index = matrix.index[: self.max_items_by]

            matrix = matrix.loc[index, columns]

            columns = [
                column
                for column in matrix.columns
                if int(column.split(" ")[-1].split(":")[0]) >= self.min_occ_col
            ]
            index = [
                index
                for index in matrix.index
                if int(index.split(" ")[-1].split(":")[0]) >= self.min_occ_by
            ]
            matrix = matrix.loc[index, columns]

            self.X_ = matrix

        self.sort()

    def sort(self):

        self.X_ = sort_by_axis(
            data=self.X_,
            sort_by=self.sort_r_axis_by,
            ascending=self.r_axis_ascending,
            axis=0,
        )

        self.X_ = sort_by_axis(
            data=self.X_,
            sort_by=self.sort_c_axis_by,
            ascending=self.c_axis_ascending,
            axis=1,
        )

    def matrix(self):
        self.apply()
        return self.X_.style.background_gradient(cmap=self.colormap_col, axis=None)

    def heatmap(self):
        self.apply()
        return heatmap_(
            self.X_,
            cmap=self.colormap_col,
            figsize=(self.width, self.height),
        )

    def bubble_plot(self):
        self.apply()
        return bubble_plot(
            self.X_,
            axis=0,
            cmap=self.colormap_col,
            figsize=(self.width, self.height),
        )

    def network(self):
        ##
        self.apply()
        ##

        X = self.X_

        ## Networkx
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        G = nx.Graph(ax=ax)
        G.clear()

        ## Data preparation
        terms = X.columns.tolist() + X.index.tolist()

        node_sizes = counters_to_node_sizes(x=terms)
        column_node_sizes = node_sizes[: len(X.index)]
        index_node_sizes = node_sizes[len(X.index) :]

        node_colors = counters_to_node_colors(x=terms, cmap=lambda w: w)
        column_node_colors = node_colors[: len(X.index)]
        index_node_colors = node_colors[len(X.index) :]

        cmap = pyplot.cm.get_cmap(self.colormap_col)
        cmap_by = pyplot.cm.get_cmap(self.colormap_by)

        index_node_colors = [cmap_by(t) for t in index_node_colors]
        column_node_colors = [cmap(t) for t in column_node_colors]

        ## Add nodes
        G.add_nodes_from(terms)

        ## node positions
        pos = nx.spring_layout(
            G,
            iterations=self.nx_iterations,
            k=self.nx_k,
            scale=self.nx_scale,
            seed=int(self.random_state),
        )

        # if self.layout == "Spring":
        #     pos = nx.spring_layout(G, iterations=self.nx_iterations)
        # else:
        #     pos = {
        #         "Circular": nx.circular_layout,
        #         "Kamada Kawai": nx.kamada_kawai_layout,
        #         "Planar": nx.planar_layout,
        #         "Random": nx.random_layout,
        #         "Spectral": nx.spectral_layout,
        #         "Spring": nx.spring_layout,
        #         "Shell": nx.shell_layout,
        #     }[self.layout](G)

        ## links
        m = X.stack().to_frame().reset_index()
        m.columns = ["from_", "to_", "link_"]
        m = m[m.link_ > 0.0]
        m = m.reset_index(drop=True)

        max_width = m.link_.max()

        for idx in range(len(m)):

            edge = [(m.from_[idx], m.to_[idx])]
            width = 0.1 + 2.9 * m.link_[idx] / max_width
            nx.draw_networkx_edges(
                G,
                pos=pos,
                ax=ax,
                node_size=1,
                #  with_labels=False,
                edge_color="k",
                edgelist=edge,
                width=width,
            )

        #
        # Draw column nodes
        #
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            #  edge_color="k",
            nodelist=X.columns.tolist(),
            node_size=column_node_sizes,
            node_color=column_node_colors,
            node_shape="o",
            edgecolors="k",
            linewidths=1,
        )

        #
        # Draw index nodes
        #
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            #  edge_color="k",
            nodelist=X.index.tolist(),
            node_size=index_node_sizes,
            node_color=index_node_colors,
            node_shape="o",
            edgecolors="k",
            linewidths=1,
        )

        node_sizes = column_node_sizes + index_node_sizes

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_points = [pos[label][0] for label in terms]
        y_points = [pos[label][1] for label in terms]
        x_mean = sum(x_points) / len(x_points)
        y_mean = sum(y_points) / len(y_points)

        for label, size in zip(terms, column_node_sizes + index_node_sizes):
            x_point, y_point = pos[label]
            delta_x = 0.05 * (xlim[1] - xlim[0]) + 0.001 * size / 300 * (
                xlim[1] - xlim[0]
            )
            delta_y = 0.05 * (ylim[1] - ylim[0]) + 0.001 * size / 300 * (
                ylim[1] - ylim[0]
            )
            ha = "left" if x_point > x_mean else "right"
            va = "top" if y_point > y_mean else "bottom"
            delta_x = delta_x if x_point > x_mean else -delta_x
            delta_y = delta_y if y_point > y_mean else -delta_y

            ax.text(
                x_point + delta_x,
                y_point + delta_y,
                s=label,
                fontsize=10,
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
                horizontalalignment=ha,
                verticalalignment=va,
            )

        #  ax_text_node_labels(ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes)
        expand_ax_limits(ax)
        #  ax.set_aspect("equal")
        ax.axis("off")
        set_spines_invisible(ax)
        return fig

    ##
    ##
    ## inferfaz con pyviz
    def network_interactive(self):
        ##
        self.apply()
        ##

        X = self.X_.copy()

        G = Network("700px", "870px", notebook=True)

        ## Data preparation
        terms = X.columns.tolist() + X.index.tolist()

        node_sizes = counters_to_node_sizes(x=terms)
        column_node_sizes = node_sizes[: len(X.index)]
        index_node_sizes = node_sizes[len(X.index) :]

        node_colors = counters_to_node_colors(x=terms, cmap=lambda w: w)
        column_node_colors = node_colors[: len(X.index)]
        index_node_colors = node_colors[len(X.index) :]

        cmap = pyplot.cm.get_cmap(self.colormap_col)
        cmap_by = pyplot.cm.get_cmap(self.colormap_by)
        column_node_colors = [cmap(t) for t in column_node_colors]
        column_node_colors = [
            matplotlib.colors.rgb2hex(t[:3]) for t in column_node_colors
        ]
        index_node_colors = [cmap_by(t) for t in index_node_colors]
        index_node_colors = [
            matplotlib.colors.rgb2hex(t[:3]) for t in index_node_colors
        ]

        for i_term, term in enumerate(X.columns.tolist()):
            G.add_node(
                term,
                size=column_node_sizes[i_term] / 100,
                color=column_node_colors[i_term],
            )

        for i_term, term in enumerate(X.index.tolist()):
            G.add_node(
                term,
                size=index_node_sizes[i_term] / 100,
                color=index_node_colors[i_term],
            )

        # links
        m = X.stack().to_frame().reset_index()
        m.columns = ["from_", "to_", "link_"]
        m = m[m.link_ > 0.0]
        m = m.reset_index(drop=True)

        max_width = m.link_.max()
        for idx in range(len(m)):
            value = 0.5 + 2.5 * m.link_[idx] / max_width
            G.add_edge(
                m.from_[idx], m.to_[idx], width=value, color="lightgray", physics=False
            )

        return G.show("net.html")

    def slope_chart(self):
        ##
        self.apply()
        ##

        matrix = self.X_
        ##
        matplotlib.rc("font", size=12)

        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = pyplot.cm.get_cmap(self.colormap_col)
        cmap_by = pyplot.cm.get_cmap(self.colormap_by)

        m = len(matrix.index)
        n = len(matrix.columns)
        maxmn = max(m, n)
        yleft = (maxmn - m) / 2.0 + np.linspace(0, m, m)
        yright = (maxmn - n) / 2.0 + np.linspace(0, n, n)

        ax.vlines(
            x=1,
            ymin=-1,
            ymax=maxmn + 1,
            color="gray",
            alpha=0.7,
            linewidth=1,
            linestyles="dotted",
        )

        ax.vlines(
            x=3,
            ymin=-1,
            ymax=maxmn + 1,
            color="gray",
            alpha=0.7,
            linewidth=1,
            linestyles="dotted",
        )

        #
        # Dibuja los ejes para las conexiones
        #
        ax.scatter(x=[1] * m, y=yleft, s=1)
        ax.scatter(x=[3] * n, y=yright, s=1)

        #
        # Dibuja las conexiones
        #
        maxlink = matrix.max().max()
        minlink = matrix.values.ravel()
        minlink = min([v for v in minlink if v > 0])
        for idx, index in enumerate(matrix.index):
            for icol, col in enumerate(matrix.columns):
                link = matrix.loc[index, col]
                if link > 0:
                    ax.plot(
                        [1, 3],
                        [yleft[idx], yright[icol]],
                        c="k",
                        linewidth=0.5 + 4 * (link - minlink) / (maxlink - minlink),
                        alpha=0.5 + 0.5 * (link - minlink) / (maxlink - minlink),
                    )

        #
        # Sizes
        #
        left_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in matrix.index]
        right_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in matrix.columns]

        min_size = min(left_sizes + right_sizes)
        max_size = max(left_sizes + right_sizes)

        left_sizes = [
            150 + 2000 * (t - min_size) / (max_size - min_size) for t in left_sizes
        ]
        right_sizes = [
            150 + 2000 * (t - min_size) / (max_size - min_size) for t in right_sizes
        ]

        #
        # Colors
        #
        left_colors = [int(t.split(" ")[-1].split(":")[1]) for t in matrix.index]
        right_colors = [int(t.split(" ")[-1].split(":")[1]) for t in matrix.columns]

        min_color = min(left_colors + right_colors)
        max_color = max(left_colors + right_colors)

        left_colors = [
            cmap_by(0.1 + 0.9 * (t - min_color) / (max_color - min_color))
            for t in left_colors
        ]
        right_colors = [
            cmap(0.1 + 0.9 * (t - min_color) / (max_color - min_color))
            for t in right_colors
        ]

        ax.scatter(
            x=[1] * m,
            y=yleft,
            s=left_sizes,
            c=left_colors,
            zorder=10,
            linewidths=1,
            edgecolors="k",
        )

        for idx, text in enumerate(matrix.index):
            ax.plot([0.7, 1.0], [yleft[idx], yleft[idx]], "-", c="grey")

        for idx, text in enumerate(matrix.index):
            ax.text(
                0.7,
                yleft[idx],
                text,
                fontsize=10,
                ha="right",
                va="center",
                zorder=10,
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
            )

        #
        # right y-axis
        #

        ax.scatter(
            x=[3] * n,
            y=yright,
            s=right_sizes,
            c=right_colors,
            zorder=10,
            linewidths=1,
            edgecolors="k",
        )

        for idx, text in enumerate(matrix.columns):
            ax.plot([3.0, 3.3], [yright[idx], yright[idx]], "-", c="grey")

        for idx, text in enumerate(matrix.columns):
            ax.text(
                3.3,
                yright[idx],
                text,
                fontsize=10,
                ha="left",
                va="center",
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
                zorder=11,
            )

        #
        # Figure size
        #
        expand_ax_limits(ax)
        ax.invert_yaxis()
        ax.axis("off")

        return fig


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
        """Dashboard app"""

        data = filter_records(pd.read_csv("corpus.csv"))

        Model.__init__(
            self,
            data=data,
            limit_to=limit_to,
            exclude=exclude,
            years_range=years_range,
        )

        COLUMNS = sorted([column for column in data.columns])

        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.Dropdown(
                options=[
                    "Matrix",
                    "Heatmap",
                    "Bubble plot",
                    "Network",
                    "Slope chart",
                ],
            ),
            dash.HTML("Column parameters:"),
            dash.Dropdown(
                description="Column:",
                options=sorted(data.columns),
            ),
            dash.min_occurrence(description="Min OCC COL"),
            dash.max_items(description="Max items COL"),
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
            dash.HTML("Index parameters:"),
            dash.Dropdown(
                description="By:",
                options=sorted(data.columns),
            ),
            dash.min_occurrence(description="Min OCC by:"),
            dash.max_items(description="Max items by:"),
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
            dash.HTML("Visualization:"),
            dash.Dropdown(
                description="Top by:",
                options=[
                    "Num Documents",
                    "Global Citations",
                    "Data",
                ],
            ),
            dash.cmap(description="Colormap Col:"),
            dash.cmap(description="Colormap By:"),
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
                # Column Parameters:
                "column": self.command_panel[3],
                "min_occ_col": self.command_panel[4],
                "max_items_col": self.command_panel[5],
                "sort_c_axis_by": self.command_panel[6],
                "c_axis_ascending": self.command_panel[7],
                # Index Parameters:
                "by": self.command_panel[9],
                "min_occ_by": self.command_panel[10],
                "max_items_by": self.command_panel[11],
                "sort_r_axis_by": self.command_panel[12],
                "r_axis_ascending": self.command_panel[13],
                # Visualization
                "top_by": self.command_panel[15],
                "colormap_col": self.command_panel[16],
                "colormap_by": self.command_panel[17],
                "nx_iterations": self.command_panel[18],
                "nx_k": self.command_panel[19],
                "nx_scale": self.command_panel[20],
                "random_state": self.command_panel[21],
                "width": self.command_panel[22],
                "height": self.command_panel[23],
            },
        )

        Dashboard.__init__(self)

        self.interactive_output(
            **{
                # Display:
                "menu": self.command_panel[1].value,
                # Column Parameters:
                "column": self.command_panel[3].value,
                "min_occ_col": self.command_panel[4].value,
                "max_items_col": self.command_panel[5].value,
                "sort_c_axis_by": self.command_panel[6].value,
                "c_axis_ascending": self.command_panel[7].value,
                # Index Parameters:
                "by": self.command_panel[9].value,
                "min_occ_by": self.command_panel[10].value,
                "max_items_by": self.command_panel[11].value,
                "sort_r_axis_by": self.command_panel[12].value,
                "r_axis_ascending": self.command_panel[13].value,
                # Visualization
                "top_by": self.command_panel[15].value,
                "colormap_col": self.command_panel[16].value,
                "colormap_by": self.command_panel[17].value,
                "nx_iterations": self.command_panel[18].value,
                "nx_k": self.command_panel[19].value,
                "nx_scale": self.command_panel[20].value,
                "random_state": self.command_panel[21].value,
                "width": self.command_panel[22].value,
                "height": self.command_panel[23].value,
            }
        )

    def interactive_output(self, **kwargs):

        Dashboard.interactive_output(self, **kwargs)

        with self.output:
            if self.column == self.by:
                for widget in self.command_panel[3:]:
                    widget.disabled = True
                self.set_enabled("By:")
                return
            else:
                for widget in self.command_panel[3:]:
                    widget.disabled = False

            if self.menu in ["Matrix", "Network interactive"]:
                self.set_disabled("Width:")
                self.set_disabled("Height:")
            else:
                self.set_enabled("Width:")
                self.set_enabled("Height:")

            if self.menu in ["Network", "Network interactive", "Slope chart"]:
                self.set_enabled("Colormap by:")
            else:
                self.set_disabled("Colormap by:")

            if self.menu == "Network":
                self.set_enabled("NX iterations:")
                self.set_enabled("NX K:")
                self.set_enabled("NX scale:")
                self.set_enabled("Random State:")
            else:
                self.set_disabled("NX iterations:")
                self.set_disabled("NX K:")
                self.set_disabled("NX scale:")
                self.set_disabled("Random State:")
