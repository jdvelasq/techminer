def counters_to_node_sizes(x):
    node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in x]
    max_size = max(node_sizes)
    min_size = min(node_sizes)
    if min_size == max_size:
        node_sizes = [1000] * len(node_sizes)
    else:
        node_sizes = [
            100 + int(4000 * (w - min_size) / (max_size - min_size)) for w in node_sizes
        ]
    return node_sizes
