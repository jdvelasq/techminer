import math
import matplotlib
import matplotlib.pyplot as pyplot
import networkx as nx
import pandas as pd
from cdlib import algorithms
from pyvis.network import Network as Network_

from techminer.core.sort_axis import sort_axis
from techminer.plots import (
    counters_to_node_sizes,
    expand_ax_limits,
    set_spines_invisible,
)

cluster_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "cornflowerblue",
    "lightsalmon",
    "limegreen",
    "tomato",
    "mediumvioletred",
    "darkgoldenrod",
    "lightcoral",
    "silver",
    "darkkhaki",
    "skyblue",
    "dodgerblue",
    "orangered",
    "turquoise",
    "crimson",
    "violet",
    "goldenrod",
    "thistle",
    "grey",
    "yellowgreen",
    "lightcyan",
]

cluster_colors += cluster_colors + cluster_colors


class Network:
    def __init__(self, X, top_by, n_labels, clustering):

        X = X.copy()

        ##
        ##  Network generation
        ##
        G = nx.Graph()

        ##
        ##  Top terms for labels
        ##
        X = sort_axis(
            data=X,
            num_documents=(top_by == "Num Documents"),
            axis=1,
            ascending=False,
        )
        self.top_terms_ = X.columns.tolist()[:n_labels]

        ##
        ##  Add nodes to the network
        ##
        terms = X.columns.tolist()
        G.add_nodes_from(terms)

        ##
        ##  Adds size property to nodes
        ##
        node_sizes = counters_to_node_sizes(terms)
        for term, size in zip(terms, node_sizes):
            G.nodes[term]["size"] = size

        dict_ = {a: b for a, b in zip(terms, node_sizes)}

        ##
        ##  Add edges to the network
        ##
        m = X.stack().to_frame().reset_index()
        m = m[m.level_0 < m.level_1]
        m.columns = ["from_", "to_", "link_"]
        m = m[m.link_ > 0.001]
        m = m.reset_index(drop=True)
        for idx in range(len(m)):
            G.add_edge(
                m.from_[idx],
                m.to_[idx],
                width=m.link_[idx],
                color="lightgray",
                physics=False,
            )

        ##
        ##  Network clustering
        ##
        R = {
            "Label propagation": algorithms.label_propagation,
            "Leiden": algorithms.leiden,
            "Louvain": algorithms.louvain,
            "Walktrap": algorithms.walktrap,
        }[clustering](G).communities

        for i_community, community in enumerate(R):
            for item in community:
                G.nodes[item]["group"] = i_community
                dict_[item] = (dict_[item], i_community)

        self.dict_ = dict_
        ##
        ##  Cluster members
        ##
        n_communities = len(R)
        max_len = max([len(r) for r in R])
        communities = pd.DataFrame(
            "", columns=range(n_communities), index=range(max_len)
        )
        for i_community in range(n_communities):
            community = R[i_community]
            community = sorted(
                community, key=(lambda w: w.split(" ")[-1]), reverse=True
            )
            communities.at[0 : len(community) - 1, i_community] = community
        # communities = communities.head(n_labels)
        communities.columns = ["CLUST_{}".format(i) for i in range(n_communities)]

        self.cluster_members_ = communities

        ##
        ##  Saves the graph
        ##
        self.G_ = G

    def networkx_plot(self, layout, iterations, k, scale, seed, figsize):

        ##
        ## Creates the plot
        ##
        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=figsize)
        ax = fig.subplots()

        ##
        ## Compute positions of the network nodes
        ##
        pos = nx.spring_layout(
            self.G_, iterations=iterations, k=k, scale=scale, seed=seed
        )

        ##
        ## Draw the edges
        ##
        max_width = max([dict_["width"] for _, _, dict_ in self.G_.edges.data()])
        for e in self.G_.edges.data():
            a, b, dict_ = e
            edge = [(a, b)]
            width = 0.1 + 0.5 * dict_["width"] / max_width
            edge_color = (
                cluster_colors[self.dict_[a][1]]
                if self.dict_[a][0] > self.dict_[b][0]
                else cluster_colors[self.dict_[b][1]],
            )
            nx.draw_networkx_edges(
                self.G_,
                pos=pos,
                ax=ax,
                edgelist=edge,
                width=width,
                #  edge_color="k",
                edge_color=edge_color,
                node_size=1,
                alpha=0.7,
            )

        ##
        ## Draw the nodes
        ##
        node_sizes = [node[1]["size"] for node in self.G_.nodes.data()]
        for node, node_size in zip(self.G_.nodes.data(), node_sizes):
            nx.draw_networkx_nodes(
                self.G_,
                pos,
                ax=ax,
                nodelist=[node[0]],
                node_size=node_size,
                #  node_size=[node[1]["size"]],
                node_color=cluster_colors[node[1]["group"]],
                node_shape="o",
                edgecolors="k",
                linewidths=1,
                alpha=0.75,
            )

        ##
        ## Compute label positions
        ##
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        #  delta_xlim = xlim[1] - xlim[0]
        #  delta_ylim = ylim[1] - ylim[0]

        # coordinates of the nodes
        x_points = [pos[label][0] for label in self.top_terms_]
        y_points = [pos[label][1] for label in self.top_terms_]

        ## plot centers as black dots
        ax.scatter(
            x_points,
            y_points,
            marker="o",
            s=50,
            c="k",
            alpha=1.0,
            zorder=10,
        )

        #  Center of the plot
        x_mean = sum(x_points) / len(x_points)
        y_mean = sum(y_points) / len(y_points)

        factor = 0.1
        rx = factor * (xlim[1] - xlim[0])
        ry = factor * (ylim[1] - ylim[0])
        radious = math.sqrt(rx ** 2 + ry ** 2)

        for label, size in zip(self.top_terms_, node_sizes):

            x_point, y_point = pos[label]

            x_c = x_point - x_mean
            y_c = y_point - y_mean
            angle = math.atan(math.fabs(y_c / x_c))
            x_label = x_point + math.copysign(radious * math.cos(angle), x_c)
            y_label = y_point + math.copysign(radious * math.sin(angle), y_c)

            ha = "left" if x_point > x_mean else "right"
            #  va = "top" if y_point > y_mean else "bottom"
            va = "center"

            ax.text(
                #  x_point + delta_x,
                #  y_point + delta_y,
                x_label,
                y_label,
                s=label,
                fontsize=9,
                bbox=dict(
                    facecolor="w",
                    alpha=1.0,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
                horizontalalignment=ha,
                verticalalignment=va,
                alpha=0.8,
                zorder=13,
            )

            ax.plot(
                [x_point, x_label],
                [y_point, y_label],
                lw=1,
                ls="-",
                c="k",
                zorder=13,
            )

        fig.set_tight_layout(True)
        expand_ax_limits(ax)
        set_spines_invisible(ax)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig

    def pyvis_plot(self):

        nt = Network_("700px", "870px", notebook=True)
        nt.from_nx(self.G_)

        for i, _ in enumerate(nt.nodes):
            if nt.nodes[i]["label"] not in self.top_terms_:
                nt.nodes[i]["label"] = ""

        for i, _ in enumerate(nt.nodes):
            nt.nodes[i]["size"] = nt.nodes[i]["size"] / 100

        return nt.show("net.html")
