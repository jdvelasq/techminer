import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import Dashboard
from techminer.core.filter_records import filter_records
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as pyplot
import json

from techminer.plots import (
    counters_to_node_sizes,
    expand_ax_limits,
    set_spines_invisible,
)

from techminer.core.main_path import MainPath


class App(Dashboard):
    def __init__(self):

        self.data = filter_records(pd.read_csv("corpus.csv"))

        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.RadioButtons(
                options=[
                    "Main path table",
                    "Main path network",
                ],
                description="",
            ),
            dash.HTML("Parameters:"),
            #  dash.cmap(),
            #  dash.n_labels(),
            dash.nx_iterations(),
            dash.nx_k(),
            dash.nx_scale(),
            dash.random_state(),
            dash.fig_width(),
            dash.fig_height(),
        ]

        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                # Display:
                "menu": self.command_panel[1],
                # Parameters:
                "nx_iterations": self.command_panel[3],
                "nx_k": self.command_panel[4],
                "nx_scale": self.command_panel[5],
                "random_state": self.command_panel[6],
                "width": self.command_panel[7],
                "height": self.command_panel[8],
            },
        )

        Dashboard.__init__(self)

    def interactive_output(self, **kwargs):

        Dashboard.interactive_output(self, **kwargs)

    def create_sequence(self):
        data = self.data[["Historiograph_ID", "Local_References"]].dropna()
        data = {
            key: value.split(";")
            for key, value in zip(data.Historiograph_ID, data.Local_References)
        }
        return data

    def compute_global_key_route_main_paths(self):

        sequence = self.create_sequence()
        main_path = MainPath(nodes=sequence)
        main_path.search_sources()
        main_path.build_paths()
        main_path.search_path_count()
        main_path.global_key_route_search()
        self.links = main_path.links
        self.global_key_route_paths = main_path.global_key_route_paths

    def generate_filter(self):

        nodes = sorted(
            set(
                [a for a, _ in self.global_key_route_paths]
                + [b for _, b in self.global_key_route_paths]
            )
        )
        data = self.data[["Historiograph_ID", "ID"]]
        data = data[data.Historiograph_ID.map(lambda w: w in nodes)]

        with open("filters.json", "r") as f:
            filters = json.load(f)

        for key in filters.copy():
            if key not in [
                "bradford_law_zones",
                "citations_range",
                "citations",
                "document_types",
                "excluded_terms",
                "selected_cluster",
                "selected_types",
                "year_range",
                "years",
            ]:
                filters.pop(key)

        filters["Main Path"] = list(data["ID"])

        with open("filters.json", "w") as f:
            print(json.dumps(filters, indent=4, sort_keys=True), file=f)

    def main_path_table(self):

        self.compute_global_key_route_main_paths()
        self.generate_filter()

        nodes = sorted(
            set(
                [a for a, _ in self.global_key_route_paths]
                + [b for _, b in self.global_key_route_paths]
            )
        )
        data = self.data[["Title", "Historiograph_ID", "Local_References", "ID"]]
        data = data[data.Historiograph_ID.map(lambda w: w in nodes)]

        data["Local_References"] = data.Local_References.map(
            lambda w: ";".join([a for a in w.split(";") if a in nodes]),
            na_action="ignore",
        )

        return data

    def main_path_network(self):

        self.compute_global_key_route_main_paths()
        return self.main_path_network()

    def main_path_network(self):

        self.compute_global_key_route_main_paths()

        nodes = sorted(
            set(
                [a for a, _ in self.global_key_route_paths]
                + [b for _, b in self.global_key_route_paths]
            )
        )

        G = nx.Graph()
        G.add_nodes_from(nodes)

        for key in self.links.keys():
            a, b = key
            if a in nodes and b in nodes:
                G.add_edge(
                    a,
                    b,
                    width=float(self.links[key]),
                    color="gray",
                    physics=False,
                )
        self.G_ = G

        matplotlib.rc("font", size=11)
        fig = pyplot.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()

        pos = nx.spring_layout(
            self.G_,
            iterations=self.nx_iterations,
            k=self.nx_k,
            scale=self.nx_scale,
            seed=int(self.random_state),
        )

        ##
        ## Draw the edges
        ##
        max_width = max([dict_["width"] for _, _, dict_ in self.G_.edges.data()])
        for e in self.G_.edges.data():
            a, b, dict_ = e
            edge = [(a, b)]
            width = 0.1 + 0.5 * dict_["width"] / max_width
            nx.draw_networkx_edges(
                self.G_,
                pos=pos,
                ax=ax,
                edgelist=edge,
                width=width,
                #  edge_color="k",
                # edge_color=edge_color,
                node_size=1,
                alpha=0.7,
            )

        ##
        ## Draw the nodes
        ##
        #  node_sizes = [node[1]["size"] for node in self.G_.nodes.data()]
        for node in zip(self.G_.nodes.data()):
            print(node[0][0])
            nx.draw_networkx_nodes(
                self.G_,
                pos,
                ax=ax,
                nodelist=[node[0][0]],
                #  node_size=node_size,
                #  node_size=[node[1]["size"]],
                #  node_color=cluster_colors[node[1]["group"]],
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

        # # coordinates of the nodes
        # x_points = [pos[label][0] for label in self.top_terms_]
        # y_points = [pos[label][1] for label in self.top_terms_]

        ## plot centers as black dots
        # ax.scatter(
        #     x_points,
        #     y_points,
        #     marker="o",
        #     s=50,
        #     c="k",
        #     alpha=1.0,
        #     zorder=10,
        # )

        #  Center of the plot
        # x_mean = sum(x_points) / len(x_points)
        # y_mean = sum(y_points) / len(y_points)

        # factor = 0.1
        # rx = factor * (xlim[1] - xlim[0])
        # ry = factor * (ylim[1] - ylim[0])
        # radious = math.sqrt(rx ** 2 + ry ** 2)

        # for label, size in zip(self.top_terms_, node_sizes):

        #     x_point, y_point = pos[label]

        #     x_c = x_point - x_mean
        #     y_c = y_point - y_mean
        #     angle = math.atan(math.fabs(y_c / x_c))
        #     x_label = x_point + math.copysign(radious * math.cos(angle), x_c)
        #     y_label = y_point + math.copysign(radious * math.sin(angle), y_c)

        #     ha = "left" if x_point > x_mean else "right"
        #     #  va = "top" if y_point > y_mean else "bottom"
        #     va = "center"

        #     ax.text(
        #         #  x_point + delta_x,
        #         #  y_point + delta_y,
        #         x_label,
        #         y_label,
        #         s=label,
        #         fontsize=9,
        #         bbox=dict(
        #             facecolor="w",
        #             alpha=1.0,
        #             edgecolor="gray",
        #             boxstyle="round,pad=0.5",
        #         ),
        #         horizontalalignment=ha,
        #         verticalalignment=va,
        #         alpha=0.8,
        #         zorder=13,
        #     )

        #     ax.plot(
        #         [x_point, x_label],
        #         [y_point, y_label],
        #         lw=1,
        #         ls="-",
        #         c="k",
        #         zorder=13,
        #     )

        fig.set_tight_layout(True)
        expand_ax_limits(ax)
        set_spines_invisible(ax)
        ax.set_aspect("equal")
        ax.axis("off")

        return fig
