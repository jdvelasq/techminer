def counters_to_node_colors(x, cmap):
    node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in x]
    max_citations = max(node_colors)
    min_citations = min(node_colors)

    if min_citations == max_citations:
        node_colors = [0.5] * len(node_colors)
    else:
        node_colors = [
            cmap(0.4 + 0.60 * (i - min_citations) / (max_citations - min_citations))
            for i in node_colors
        ]
    return node_colors
