def APP(app_title, tab_titles, tab_widgets, tab=None):
    """Jupyter Lab dashboard.
    """

    if tab is not None:
        return AppLayout(
            header=widgets.HTML(
                value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                    app_title + " / " + tab_titles[tab]
                )
            ),
            center=tab_widgets[tab],
            pane_heights=["80px", "660px", 0],  # tamaño total de la ventana: Ok!
        )

    body = widgets.Tab()
    body.children = tab_widgets
    for i in range(len(tab_widgets)):
        body.set_title(i, tab_titles[i])
    return AppLayout(
        header=widgets.HTML(
            value="<h1>{}</h1><hr style='height:2px;border-width:0;color:gray;background-color:gray'>".format(
                app_title
            )
        ),
        center=body,
        pane_heights=["80px", "720px", 0],
    )


def TABapp(left_panel, server, output):

    #  defines interactive output
    args = {control["arg"]: control["widget"] for control in left_panel}
    with output:
        display(widgets.interactive_output(server, args,))

    grid = GridspecLayout(13, 6, height="650px")

    # left panel layout
    grid[0:, 0] = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.Label(value=left_panel[index]["desc"]),
                    left_panel[index]["widget"],
                ],
                layout=Layout(
                    display="flex", justify_content="flex-end", align_content="center",
                ),
            )
            for index in range(len(left_panel))
        ]
    )

    # output
    grid[:, 1:] = widgets.VBox(
        [output], layout=Layout(height="650px", border="2px solid gray")
    )

    return grid


def associations_map(X, selected, cmap, layout, figsize):

    if selected == ():
        return "There are not associations to show"

    #
    # Network generation
    #
    matplotlib.rc("font", size=11)
    fig = pyplot.Figure(figsize=figsize)
    ax = fig.subplots()
    G = nx.Graph(ax=ax)
    G.clear()

    Y = X[[t for t in X.columns if t in selected]]
    S = Y.sum(axis=1)
    S = S[S > 0]
    X = Y.loc[S.index, :]
    if len(X) == 0:
        return "There are not associations to show"

    terms = X.index.tolist()
    G.add_nodes_from(terms)

    max_width = 0
    for icol in range(len(X.columns)):
        for irow in range(len(X.index)):
            if X.index[irow] != X.columns[icol]:
                link = X.loc[X.index[irow], X.columns[icol]]
                if link > 0:
                    G.add_edge(X.index[irow], X.columns[icol], width=link)
                    if max_width < link:
                        max_width = link

    cmap = pyplot.cm.get_cmap(cmap)
    node_colors = [int(t.split(" ")[-1].split(":")[1]) for t in X.index]
    max_citations = max(node_colors)
    min_citations = min(node_colors)
    if max_citations == min_citations:
        node_colors = [cmap(0.9)] * len(X.index)
    else:
        node_colors = [
            cmap(0.2 + (t - min_citations) / (max_citations - min_citations))
            for t in node_colors
        ]

    node_sizes = [int(t.split(" ")[-1].split(":")[0]) for t in X.index]
    max_size = max(node_sizes)
    min_size = min(node_sizes)
    if max_size == min_size:
        node_sizes = [600] * len(node_sizes)
    else:
        node_sizes = [
            600 + int(2500 * (w - min_size) / (max_size - min_size)) for w in node_sizes
        ]

    pos = {
        "Circular": nx.circular_layout,
        "Kamada Kawai": nx.kamada_kawai_layout,
        "Planar": nx.planar_layout,
        "Random": nx.random_layout,
        "Spectral": nx.spectral_layout,
        "Spring": nx.spring_layout,
        "Shell": nx.shell_layout,
    }[layout](G)

    for e in G.edges.data():
        a, b, width = e
        edge = [(a, b)]
        width = 0.2 + 4.0 * width["width"] / max_width
        nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            edgelist=edge,
            width=width,
            edge_color="k",
            with_labels=False,
            node_size=1,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        edge_color="k",
        nodelist=terms,
        node_size=node_sizes,
        node_color=node_colors,
        node_shape="o",
        edgecolors="k",
        linewidths=1,
    )

    common.ax_text_node_labels(ax=ax, labels=terms, dict_pos=pos, node_sizes=node_sizes)
    common.ax_expand_limits(ax)
    common.set_ax_splines_invisible(ax)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.set_tight_layout(True)

    return fig


#
# Association Map
#
def __TAB1__(data, limit_to, exclude):
    # -------------------------------------------------------------------------
    #
    # UI
    #
    # -------------------------------------------------------------------------
    COLUMNS = sorted([column for column in data.columns if column not in EXCLUDE_COLS])
    #
    left_panel = [
        # 0
        {
            "arg": "column",
            "desc": "Column:",
            "widget": widgets.Dropdown(
                options=[z for z in COLUMNS if z in data.columns],
                layout=Layout(width="55%"),
            ),
        },
        # 1
        {
            "arg": "top_by",
            "desc": "Top by:",
            "widget": widgets.Dropdown(
                options=["Num Documents", "Global Citations",],
                layout=Layout(width="55%"),
            ),
        },
        # 2
        dash.top_n(),
        # 3
        dash.normalization(),
        dash.cmap(),
        # 5
        dash.nx_layout(),
        # 6
        {
            "arg": "width",
            "desc": "Width:",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1), ensure_option=True, layout=Layout(width="55%"),
            ),
        },
        # 7
        {
            "arg": "height",
            "desc": "Height:",
            "widget": widgets.Dropdown(
                options=range(5, 15, 1), ensure_option=True, layout=Layout(width="55%"),
            ),
        },
        # 8
        {
            "arg": "selected",
            "desc": "Seleted Cols:",
            "widget": widgets.widgets.SelectMultiple(
                options=[], layout=Layout(width="95%", height="212px"),
            ),
        },
    ]
    # -------------------------------------------------------------------------
    #
    # Logic
    #
    # -------------------------------------------------------------------------
    def server(**kwargs):
        #
        # Logic
        #
        column = kwargs["column"]
        cmap = kwargs["cmap"]
        top_by = kwargs["top_by"]
        top_n = int(kwargs["top_n"])
        layout = kwargs["layout"]
        normalization = kwargs["normalization"]
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        selected = kwargs["selected"]

        matrix = co_occurrence_matrix(
            data=data,
            column=column,
            top_by=top_by,
            top_n=top_n,
            normalization=normalization,
            limit_to=limit_to,
            exclude=exclude,
        )

        left_panel[-1]["widget"].options = sorted(matrix.columns)

        output.clear_output()
        with output:
            display(
                associations_map(
                    X=matrix,
                    selected=selected,
                    cmap=cmap,
                    layout=layout,
                    figsize=(width, height),
                )
            )
        #
        return

    ###
    output = widgets.Output()
    return dash.TABapp(left_panel=left_panel, server=server, output=output)
