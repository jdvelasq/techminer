import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import Dashboard
from techminer.core.filter_records import filter_records
import pandas as pd
import networkx as nx
from cdlib import algorithms
import json


class App(Dashboard):
    def __init__(self):

        self.data = filter_records(pd.read_csv("corpus.csv"))
        # ##
        x = self.data.Bradford_Law_Zone.tolist()
        x = sorted(set(x))
        for a in x:
            print(a)
        ##

        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.RadioButtons(
                options=[
                    "Communities (table)",
                ],
                description="",
            ),
            dash.HTML("Parameters:"),
            dash.network_clustering(),
        ]

        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                # Display:
                "menu": self.command_panel[1],
                # Parameters:
                "clustering": self.command_panel[3],
            },
        )

        Dashboard.__init__(self)

    def interactive_output(self, **kwargs):

        Dashboard.interactive_output(self, **kwargs)

    def communities_table(self):

        data = self.data[["Historiograph_ID", "Local_References"]].dropna()
        sequences = {
            key: value.split(";")
            for key, value in zip(data.Historiograph_ID, data.Local_References)
        }

        G = nx.Graph()

        nodes = [key for key in sequences.keys()]
        for key in sequences.keys():
            nodes.extend(sequences[key])
        nodes = list(set(nodes))

        G.add_nodes_from(nodes)

        for key in sequences.keys():
            for value in sequences[key]:
                G.add_edge(key, value)

        R = {
            "Label propagation": algorithms.label_propagation,
            "Leiden": algorithms.leiden,
            "Louvain": algorithms.louvain,
            "Walktrap": algorithms.walktrap,
        }[self.clustering](G).communities

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
            community = sorted(community)
            communities.at[0 : len(community) - 1, i_community] = community
        # communities = communities.head(n_labels)
        communities.columns = ["CLUST_{}".format(i) for i in range(n_communities)]

        #
        #
        #
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

        for col in communities:
            x = communities[col].copy()
            x = [w for w in x if w != ""]
            data = self.data[["Historiograph_ID", "ID"]]
            data = data[data.Historiograph_ID.map(lambda w: w in x)]

            print(col, x, data.Historiograph_ID.tolist(), sep=" --- ")

            filters[col] = list(data["ID"])

        with open("filters.json", "w") as f:
            print(json.dumps(filters, indent=4, sort_keys=True), file=f)
            print(" ")

        return communities
