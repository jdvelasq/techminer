import textwrap
import re
import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from pyvis.network import Network
import matplotlib
import networkx as nx
from ipywidgets import GridspecLayout, Layout

import techminer.core.dashboard as dash
from techminer.core import (
    DASH,
    Network,
    TF_matrix,
    add_counters_to_axis,
    corpus_filter,
    limit_to_exclude,
    normalize_network,
    sort_by_axis,
    explode,
)

#  from techminer.core.params import EXCLUDE_COLS
from techminer.core import cluster_table_to_list
from techminer.core.dashboard import fig_height, fig_width
from techminer.plots import ChordDiagram
from techminer.plots import bubble_plot as bubble_plot_
from techminer.plots import counters_to_node_colors, counters_to_node_sizes
from techminer.plots import heatmap as heatmap_
from techminer.plots import (
    ax_text_node_labels,
    expand_ax_limits,
)
from techminer.plots import set_spines_invisible

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(
        self,
        data,
    ):

        self.data = data

        self.colormap = None
        self.column = None
        self.height = None
        self.keyword = None
        self.max_items = None
        self.min_occ = None
        self.normalization = None
        self.width = None

    def radial_diagram(self):

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
        ##  Adds counters to axis
        ##
        TF_matrix_ = add_counters_to_axis(
            X=TF_matrix_, axis=1, data=self.data, column=self.column
        )
        TF_matrix_ = sort_by_axis(
            data=TF_matrix_, sort_by="Num_Documents", ascending=False, axis=1
        )

        ##
        ##  Selected keyword + counters
        ##
        keyword = [
            w
            for w in TF_matrix_.columns.tolist()
            if (" ".join(w.split(" ")[:-1]).lower() == self.keyword)
        ]
        if len(keyword) > 0:
            keyword = keyword[0]
        else:
            return widgets.HTML(
                "<pre>No associations found for the selected parameters </pre>"
            )

        ##
        ##  Co-occurrence matrix and association index
        ##
        X = np.matmul(TF_matrix_.transpose().values, TF_matrix_[[keyword]].values)
        X = pd.DataFrame(X, columns=[keyword], index=TF_matrix_.columns)

        ##
        ## Selectec the corresponding column
        ##
        X = X.sort_values([keyword], ascending=False)
        X = X[X[keyword] > 0]
        X = sort_by_axis(data=X, sort_by="Num_Documents", ascending=False, axis=0)

        X = X[
            X.index.map(lambda w: int(w.split(" ")[-1].split(":")[0])) >= self.min_occ
        ]

        X = X.head(self.max_items)

        ##
        ## Network plot
        ##
        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = pyplot.cm.get_cmap(self.colormap)

        ##
        ## Selects the column with values > 0
        ##
        X = X[X.index != keyword]

        nodes = X.index.tolist() + [keyword]
        node_sizes = counters_to_node_sizes(nodes)
        node_colors = counters_to_node_colors(x=nodes, cmap=lambda w: w)
        node_colors = [cmap(t) for t in node_colors]

        G = nx.Graph()
        G.add_nodes_from(nodes)
        for i, w in zip(X.index, X[keyword]):
            G.add_edge(i, keyword, width=w)

        ##
        ## Layout
        ##
        pos = nx.kamada_kawai_layout(G)

        ##
        ## Draw network edges
        ##
        for e in G.edges.data():
            a, b, dict_ = e
            edge = [(a, b)]
            width = 0.5 + 2.0 * dict_["width"]
            nx.draw_networkx_edges(
                G,
                pos=pos,
                ax=ax,
                edgelist=edge,
                width=width,
                edge_color="k",
                node_size=1,
                alpha=0.5,
            )

        ##
        ## Draw network nodes
        ##
        for i_node, node in enumerate(G.nodes.data()):
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=[node[0]],
                node_size=node_sizes[i_node],
                node_color=node_colors[i_node],
                node_shape="o",
                edgecolors="k",
                linewidths=1,
                alpha=0.8,
            )

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for i_node, label in enumerate(nodes):
            x_point, y_point = pos[label]
            ax.text(
                x_point
                + 0.01 * (xlim[1] - xlim[0])
                + 0.001 * node_sizes[i_node] / 300 * (xlim[1] - xlim[0]),
                y_point
                - 0.01 * (ylim[1] - ylim[0])
                - 0.001 * node_sizes[i_node] / 300 * (ylim[1] - ylim[0]),
                s=label,
                fontsize=10,
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
                horizontalalignment="left",
                verticalalignment="top",
            )

        fig.set_tight_layout(True)
        expand_ax_limits(ax)
        set_spines_invisible(ax)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig

    def concordances(self):

        data = self.data.copy()
        data = data[
            ["Authors", "Historiograph_ID", "Abstract", "Global_Citations"]
        ].dropna()
        data["Authors"] = data.Authors.map(lambda w: w.replace(";", ", "))
        data["Global_Citations"] = data.Global_Citations.map(int)
        data["REF"] = (
            data.Authors
            + ". "
            + data.Historiograph_ID
            + ". Times Cited: "
            + data.Global_Citations.map(str)
        )
        data = data[["REF", "Abstract", "Global_Citations"]]
        data["Abstract"] = data.Abstract.map(lambda w: w.split(". "))
        data = data.explode("Abstract")
        data = data[data.Abstract.map(lambda w: self.keyword.lower() in w.lower())]
        data = data.groupby(["REF", "Global_Citations"], as_index=False).agg(
            {"Abstract": list}
        )
        data["Abstract"] = data.Abstract.map(lambda w: ". <br><br>".join(w))
        data["Abstract"] = data.Abstract.map(lambda w: w + ".")
        data = data.sort_values(["Global_Citations", "REF"], ascending=[False, True])
        data = data.head(50)
        pattern = re.compile(self.keyword, re.IGNORECASE)
        data["Abstract"] = data.Abstract.map(
            lambda w: pattern.sub("<b>" + self.keyword.upper() + "</b>", w)
        )

        HTML = ""
        for ref, phrase in zip(data.REF, data.Abstract):
            HTML += "=" * 100 + "<br>"
            HTML += ref + "<br><br>"
            phrases = textwrap.wrap(phrase, 80)
            for line in phrases:
                HTML += line + "<br>"
            HTML += "<br>"

        return widgets.HTML("<pre>" + HTML + "</pre>")


###############################################################################
##
##  DASHBOARD
##
###############################################################################

COLORMAPS = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "Pastel1",
    "Pastel2",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
]


class DASHapp(DASH, Model):
    def __init__(
        self,
    ):
        data = pd.read_csv("corpus.csv")
        self.limit_to = None
        self.exclude = None

        Model.__init__(self, data=data)

        self.command_panel = [
            dash.HTML("<b>Display:</b>", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.Dropdown(options=["Concordances", "Radial Diagram"]),
            dash.HTML("Keywords selection:"),
            dash.Dropdown(
                options=[z for z in data.columns if "keywords" in z.lower()],
                description="Column:",
            ),
            dash.Dropdown(
                options=[],
                description="Keyword:",
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.HTML("Visualization:"),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                "menu": self.command_panel[1],
                "column": self.command_panel[3],
                "keyword": self.command_panel[4],
                "min_occ": self.command_panel[5],
                "max_items": self.command_panel[6],
                "colormap": self.command_panel[8],
                "width": self.command_panel[9],
                "height": self.command_panel[10],
            },
        )

        DASH.__init__(self)

        self.interactive_output(
            **{
                "menu": self.command_panel[1].value,
                "column": self.command_panel[3].value,
            }
        )

    def interactive_output(self, **kwargs):

        with self.output:

            DASH.interactive_output(self, **kwargs)

            if self.menu == "Concordances":

                data = self.data[[self.column, "Abstract"]].dropna()
                data[self.column] = data[self.column].map(lambda w: w.split(";"))
                data = data.explode(self.column)
                data.index = list(range(len(data)))
                data["selected"] = [
                    (k, v) for k, v in zip(data[self.column], data.Abstract)
                ]
                data["selected"] = data["selected"].map(lambda w: w[0] in w[1].lower())
                data = data[data.selected]
                keywords = sorted(set(data[self.column].tolist()))
                self.command_panel[4].options = keywords

            if self.menu == "Radial Diagram":

                data = self.data[[self.column]].dropna()
                data[self.column] = data[self.column].map(lambda w: w.split(";"))
                data = data.explode(self.column)
                keywords = sorted(set(data[self.column].tolist()))
                self.command_panel[4].options = keywords
