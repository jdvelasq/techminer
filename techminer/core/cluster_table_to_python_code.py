import ipywidgets as widgets


def cluster_table_to_python_code(column, table):

    dict_ = {}
    for i_cluster, cluster in enumerate(table.columns):

        x = table[cluster].tolist()
        x = [m for m in x if m.strip() != ""]
        x = [" ".join(m.split(" ")[:-1]) for m in x]

        dict_[i_cluster] = x

    HTML = "CLUSTERS = [<br>"
    HTML += '    "' + column + '",<br>'
    HTML += "    {<br>"
    for key in dict_.keys():
        HTML += "        " + str(key) + ": [<br>"
        for value in dict_[key]:
            HTML += "            " + repr(value) + ",<br>"
        HTML += "        ],<br>"
    HTML += "    }<br>"
    HTML += "]"
    return widgets.HTML("<pre>" + HTML + "</pre>")
