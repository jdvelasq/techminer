import ipywidgets as widgets


def cluster_table_to_list(table):

    members_lists = []
    for cluster in table.columns:
        x = table[cluster].tolist()
        x = [m for m in x if m.strip() != ""]
        x = [" ".join(m.split(" ")[:-1]) for m in x]
        members_lists.append("; ".join(x))
    HTML = ""
    for i_cluster, cluster in enumerate(table.columns):
        HTML += str(cluster) + ":<br><br>"
        HTML += members_lists[i_cluster] + "<br><br>"
    return widgets.HTML("<pre>" + HTML + "</pre>")
