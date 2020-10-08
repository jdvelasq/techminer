import textwrap
import re
import ipywidgets as widgets
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from pyvis.network import Network
import matplotlib
import networkx as nx

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
from techminer.plots import ChordDiagram
from techminer.plots import bubble_plot as bubble_plot_
from techminer.plots import counters_to_node_colors, counters_to_node_sizes
from techminer.plots import heatmap as heatmap_
from techminer.plots import (
    ax_text_node_labels,
    expand_ax_limits,
)
from techminer.plots import set_spines_invisible
from ipywidgets import GridspecLayout, Layout

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(
        self,
        data,
        top_n,
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
        self.top_n = top_n
        self.clusters = clusters
        self.cluster = cluster

        self.colormap = None
        self.column = None
        self.height = None
        self.keyword_a = None
        self.keyword_b = None
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

        ##
        ## Selected Keywords
        ##
        keyword_a = [
            w
            for w in TF_matrix_.columns.tolist()
            if (" ".join(w.split(" ")[:-1]).lower() == self.keyword_a)
        ]

        if len(keyword_a) > 0:
            keyword_a = keyword_a[0]
        else:
            return widgets.HTML("<pre>Keyword A not found in TF matrix</pre>")

        keyword_b = [
            w
            for w in TF_matrix_.columns.tolist()
            if (" ".join(w.split(" ")[:-1]).lower() == self.keyword_b)
        ]

        if len(keyword_b) > 0:
            keyword_b = keyword_b[0]
        else:
            return widgets.HTML("<pre>Keyword B not found in TF matrix</pre>")

        if keyword_a == keyword_b:
            return widgets.HTML("<pre>Keywords must be different!!!</pre>")

        ##
        ##  Co-occurrence matrix and association index
        ##
        X = np.matmul(
            TF_matrix_.transpose().values, TF_matrix_[[keyword_a, keyword_b]].values
        )
        X = pd.DataFrame(X, columns=[keyword_a, keyword_b], index=TF_matrix_.columns)

        ##
        ## Select occurrences > 0
        ##
        X = X[X.sum(axis=1) > 0]

        X = X[
            X.index.map(lambda w: int(w.split(" ")[-1].split(":")[0])) >= self.min_occ
        ]

        X = sort_by_axis(data=X, sort_by="Num_Documents", ascending=False, axis=0)

        link_keyword_a_keyword_b = X.loc[keyword_a, keyword_b]

        X = X.head(self.max_items)
        max_width = X.max().max()

        ##
        ## Network plot
        ##
        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = pyplot.cm.get_cmap(self.colormap)

        nodes = X.index.tolist()
        if keyword_a not in nodes:
            nodes.append(keyword_a)
        if keyword_b not in nodes:
            nodes.append(keyword_b)

        node_sizes = counters_to_node_sizes(nodes)
        node_colors = counters_to_node_colors(x=nodes, cmap=lambda w: w)
        node_colors = [cmap(t) for t in node_colors]

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edge(keyword_a, keyword_b, width=link_keyword_a_keyword_b)
        for i, w in zip(X.index, X[keyword_a]):
            if i != keyword_a:
                G.add_edge(i, keyword_a, width=w)
        for i, w in zip(X.index, X[keyword_b]):
            if i != keyword_b:
                G.add_edge(i, keyword_b, width=w)

        ##
        ## Layout
        ##
        pos = nx.spring_layout(G, weight=None)

        ##
        ## Draw network edges
        ##
        for e in G.edges.data():
            a, b, dict_ = e
            edge = [(a, b)]
            width = 1.0 + 5.0 * dict_["width"] / max_width
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
        for i_node, _ in enumerate(nodes):
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=[nodes[i_node]],
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
        data["Global_Citations"] = data.Global_Citations.map(int)
        data = data[
            ["Authors", "Historiograph_ID", "Abstract", "Global_Citations"]
        ].dropna()
        data["Authors"] = data.Authors.map(lambda w: w.replace(";", ", "))
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

        contains_a = data.Abstract.map(lambda w: self.keyword_a.lower() in w.lower())
        contains_b = data.Abstract.map(lambda w: self.keyword_b.lower() in w.lower())
        data = data[contains_a & contains_b]
        if len(data) == 0:
            return widgets.HTML("<pre>No concordances found!</pre>")

        data = data.groupby(["REF", "Global_Citations"], as_index=False).agg(
            {"Abstract": list}
        )
        data["Abstract"] = data.Abstract.map(lambda w: ". <br><br>".join(w))
        data["Abstract"] = data.Abstract.map(lambda w: w + ".")
        data = data.sort_values(["Global_Citations", "REF"], ascending=[False, True])
        data = data.head(50)
        pattern = re.compile(self.keyword_a, re.IGNORECASE)
        data["Abstract"] = data.Abstract.map(
            lambda w: pattern.sub("<b>" + self.keyword_a.upper() + "</b>", w)
        )
        pattern = re.compile(self.keyword_b, re.IGNORECASE)
        data["Abstract"] = data.Abstract.map(
            lambda w: pattern.sub("<b>" + self.keyword_b.upper() + "</b>", w)
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

        Model.__init__(
            self,
            data=data,
            top_n=None,
            limit_to=None,
            exclude=None,
            years_range=None,
        )

        self.command_panel = [
            widgets.HTML("<b>Display:</b>"),
            widgets.Dropdown(
                options=[
                    "Concordances",
                    "Radial Diagram",
                ],
                layout=Layout(width="auto"),
            ),
            widgets.HTML(
                "<hr><b>Keywords selection:</b>",
                layout=Layout(margin="20px 0px 0px 0px"),
            ),
            widgets.Dropdown(
                options=[z for z in data.columns if "keywords" in z.lower()],
                description="Column:",
                layout=Layout(width="auto"),
            ),
            widgets.Dropdown(
                options=[],
                description="Keyword A:",
                layout=Layout(width="auto"),
            ),
            widgets.Dropdown(
                options=[],
                description="Keyword B:",
                layout=Layout(width="auto"),
            ),
            widgets.Dropdown(
                options=list(range(1, 21)),
                description="Min OCC:",
                layout=Layout(width="auto"),
            ),
            widgets.Dropdown(
                options=list(range(5, 40, 1))
                + list(range(40, 100, 5))
                + list(range(100, 3001, 100)),
                description="Max items:",
                layout=Layout(width="auto"),
            ),
            widgets.HTML(
                "<hr><b>Visualization:</b>",
                layout=Layout(margin="20px 0px 0px 0px"),
            ),
            widgets.Dropdown(
                options=COLORMAPS,
                description="Colormap:",
                layout=Layout(width="auto"),
            ),
            widgets.Dropdown(
                description="Width:",
                options=range(5, 26, 1),
                layout=Layout(width="auto"),
            ),
            widgets.Dropdown(
                description="Height:",
                options=range(5, 26, 1),
                layout=Layout(width="auto"),
            ),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                "menu": self.command_panel[1],
                "column": self.command_panel[3],
                "keyword_a": self.command_panel[4],
                "keyword_b": self.command_panel[5],
                "min_occ": self.command_panel[6],
                "max_items": self.command_panel[7],
                "colormap": self.command_panel[9],
                "width": self.command_panel[10],
                "height": self.command_panel[11],
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

        DASH.interactive_output(self, **kwargs)

        #
        # Populate Keywords with all terms
        #
        x = explode(self.data, self.column)
        all_terms = pd.Series(x[self.column].unique())
        all_terms = all_terms[all_terms.map(lambda w: not pd.isna(w))]
        all_terms = all_terms.sort_values()
        self.command_panel[4].options = all_terms
        keywords_ = all_terms

        if "Abstract" in self.data.columns:

            ##
            ## Selected keyword in the GUI
            ##
            keyword_a = self.command_panel[4].value

            ##
            ## Keywords that appear in the same phrase
            ##
            data = self.data.copy()
            data = data[["Abstract"]]
            data = data.dropna()
            data["Abstract"] = data["Abstract"].map(lambda w: w.lower())
            data["Abstract"] = data["Abstract"].map(lambda w: w.split(". "))
            data = data.explode("Abstract")

            ##
            ## Extract phrases contain keyword_a
            ##
            data = data[data.Abstract.map(lambda w: keyword_a in w)]

            ##
            ## Extract keywords
            ##
            data["Abstract"] = data.Abstract.map(
                lambda w: [k for k in keywords_ if k in w]
            )
            data = data.explode("Abstract")
            all_terms = sorted(set(data.Abstract.tolist()))
            self.command_panel[5].options = all_terms

        else:

            self.command_panel[5].options = keywords_
