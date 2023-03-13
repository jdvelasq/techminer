"""
Chord Diagram Plot
==================================================================================================



"""
import textwrap

import matplotlib.pyplot as pyplot
import numpy as np

TEXTLEN = 50


class ChordDiagram:
    """Class for plotting chord diagrams.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> chord = ChordDiagram()
    >>> chord.add_nodes_from('abcdef', color='black', s=10)
    >>> chord.add_edge('a', 'b')
    >>> chord.add_edges_from([('a', 'b'), ('c', 'd'), ('e', 'f')])
    >>> _= pyplot.figure(figsize=(6, 6))
    >>> _ = chord.plot(R=1, dist=0.05, fontsize=20, color='red')
    >>> pyplot.savefig('sphinx/images/chord_demo_1.png')

    .. image:: images/chord_demo_1.png
        :width: 400px
        :align: center

    >>> chord = ChordDiagram()
    >>> chord.add_nodes_from('abcd', color='black', s=20)
    >>> chord.add_node('e', s=100, color='black')
    >>> chord.add_node('g', s=200, color='black')
    >>> chord.add_edge('a', 'b', linestyle=':', color='red')
    >>> _ = pyplot.figure(figsize=(6, 6))
    >>> _ = chord.plot(R=1, dist=0.05, fontsize=20, color='red')
    >>> pyplot.savefig('sphinx/images/chord_demo_2.png')

    .. image:: images/chord_demo_2.png
        :width: 400px
        :align: center


    """

    def __init__(self):
        self._nodes = {}
        self._edges = {}

    def add_node(self, node_to_add, **attr):
        """

        Examples
        ----------------------------------------------------------------------------------------------


        >>> chord = ChordDiagram()
        >>> chord.add_node('A', nodeA_prop=1)
        >>> chord.add_node('B', nodeB_prop=2)
        >>> chord._nodes
        {'A': {'nodeA_prop': 1}, 'B': {'nodeB_prop': 2}}

        >>> chord.add_node('C', nodeC_prop1=10, nodeC_prop2=20)

        """
        if node_to_add in self._nodes:
            self._nodes[node_to_add] = {**self._nodes[node_to_add], **attr}
        else:
            self._nodes[node_to_add] = attr

    def add_nodes_from(self, nodes_for_adding, **attr):
        """

        Examples
        ----------------------------------------------------------------------------------------------

        >>> chord = ChordDiagram()
        >>> chord.add_nodes_from('abcde')
        >>> chord._nodes
        {'a': {}, 'b': {}, 'c': {}, 'd': {}, 'e': {}}

        >>> chord = ChordDiagram()
        >>> chord.add_nodes_from('abc', linewidth=1)
        >>> chord._nodes
        {'a': {'linewidth': 1}, 'b': {'linewidth': 1}, 'c': {'linewidth': 1}}

        >>> chord = ChordDiagram()
        >>> chord.add_nodes_from( [('a', dict(linewidth=1)), ('b', {'linewidth':2}), ('c',{}) ], linewidth=10)
        >>> chord._nodes
        {'a': {'linewidth': 1}, 'b': {'linewidth': 2}, 'c': {'linewidth': 10}}

        """
        for n in nodes_for_adding:
            if isinstance(n, tuple):
                nn, ndict = n
                self.add_node(nn, **{**attr, **ndict})
            else:
                self.add_node(n, **attr)

    def add_edge(self, u, v, **attr):
        """

        Examples
        ----------------------------------------------------------------------------------------------

        >>> chord = ChordDiagram()
        >>> chord.add_edge('A','B', edgeAB_prop1=1, edgeAB_prop2=2)
        >>> chord._edges
        {('A', 'B'): {'edgeAB_prop1': 1, 'edgeAB_prop2': 2}}


        """
        u, v = sorted([u, v])
        edge = (u, v)
        if edge in self._edges:
            self._edges[edge] = {**self._edges[edge], **attr}
        else:
            self._edges[edge] = attr

    def add_edges_from(self, edges_to_add, **attr):
        """

        Examples
        ----------------------------------------------------------------------------------------------

        >>> chord = ChordDiagram()
        >>> chord.add_edges_from([(2, 3, {'linewidth': 1}), (3, 4), (5, 4)], linewidth=20)
        >>> chord._edges
        {(2, 3): {'linewidth': 1}, (3, 4): {'linewidth': 20}, (4, 5): {'linewidth': 20}}

        """
        for e in edges_to_add:
            if len(e) == 3:
                u, v, d = e
            else:
                u, v = e
                d = {}
            u, v = sorted([e[0], e[1]])
            d = {**attr, **d}
            self.add_edge(u, v, **d)

    def get_node_data(self, node, **default):
        """


        Examples
        ----------------------------------------------------------------------------------------------

        """
        if node not in self._nodes:
            return default
        else:
            return {**default, **self._nodes[node]}

    def get_edge_data(self, u, v, **default):
        """

        Examples
        ----------------------------------------------------------------------------------------------

        """
        u, v = sorted([u, v])
        key = (u, v)
        if key not in self._edges:
            return default
        else:
            return {**default, **self._edges[key]}

    def plot(self, figsize=(10, 10), R=3, dist=0.2, n_bezier=100, **text_attr):
        """Plots the diagram
        """

        def compute_node_properties():

            n_nodes = len(self._nodes)

            theta = np.linspace(0.0, 2 * np.pi, n_nodes, endpoint=False)

            node_x = [R * np.cos(t) for t in theta]
            node_y = [R * np.sin(t) for t in theta]

            label_x = [(R + dist) * np.cos(t) for t in theta]
            label_y = [(R + dist) * np.sin(t) for t in theta]

            rotation = [t / (2 * np.pi) * 360 for t in theta]
            rotation = [t - 180 if t > 180 else t for t in rotation]
            rotation = [t - 180 if t > 90 else t for t in rotation]

            ha = ["left" if xt >= 0 else "right" for xt in label_x]

            return dict(
                theta={n: p for n, p in zip(self._nodes.keys(), theta)},
                node_x={n: p for n, p in zip(self._nodes.keys(), node_x)},
                node_y={n: p for n, p in zip(self._nodes.keys(), node_y)},
                label_x={n: p for n, p in zip(self._nodes.keys(), label_x)},
                label_y={n: p for n, p in zip(self._nodes.keys(), label_y)},
                rotation={n: p for n, p in zip(self._nodes.keys(), rotation)},
                ha={n: p for n, p in zip(self._nodes.keys(), ha)},
                s={n: p for n, p in zip(self._nodes.keys(), ha)},
            )

        def draw_points():
            """Draws node and label.
            """

            for node in self._nodes:

                x = node_properties["node_x"][node]
                y = node_properties["node_y"][node]
                ax.scatter(
                    x,
                    y,
                    zorder=10,
                    edgecolors="k",
                    linewidths=0.5,
                    **self.get_node_data(node),
                )

            for label in self._nodes:

                x = node_properties["label_x"][label]
                y = node_properties["label_y"][label]
                rotation = node_properties["rotation"][label]
                ha = node_properties["ha"][label]

                attr = {**dict(backgroundcolor="white"), **text_attr}
                ax.text(
                    x,
                    y,
                    textwrap.shorten(text=label, width=TEXTLEN),
                    rotation=rotation,
                    ha=ha,
                    va="center",
                    rotation_mode="anchor",
                    bbox=dict(
                        facecolor="w",
                        alpha=1.0,
                        edgecolor="gray",
                        boxstyle="round,pad=0.5",
                    ),
                    zorder=11,
                    **attr,
                )

        def draw_edges():
            """Draws edges on the plot.
            """

            def bezier(p0, p1, p2, **kwargs):
                x0, y0 = p0
                x1, y1 = p1
                x2, y2 = p2
                xb = [
                    (1 - t) ** 2 * x0 + 2 * t * (1 - t) * x1 + t ** 2 * x2
                    for t in np.linspace(0.0, 1.0, n_bezier)
                ]
                yb = [
                    (1 - t) ** 2 * y0 + 2 * t * (1 - t) * y1 + t ** 2 * y2
                    for t in np.linspace(0.0, 1.0, n_bezier)
                ]
                ax.plot(xb, yb, **kwargs)

            for edge in self._edges:

                u, v = edge

                x0, y0, a0 = (
                    node_properties["node_x"][u],
                    node_properties["node_y"][u],
                    node_properties["theta"][u],
                )
                x2, y2, a2 = (
                    node_properties["node_x"][v],
                    node_properties["node_y"][v],
                    node_properties["theta"][v],
                )

                angle = a0 + (a2 - a0) / 2

                # if angle > np.pi:
                #     angle_corr = angle - np.pi
                # else:
                #     angle_corr = angle

                distance = np.abs(a2 - a0)
                if distance > np.pi:
                    distance = distance - np.pi
                distance = (1.0 - 1.0 * distance / np.pi) * R / 2.5
                x1 = distance * np.cos(angle)
                y1 = distance * np.sin(angle)
                x1 = 0
                y1 = 0

                ## dibuja los arcos
                bezier(
                    [x0, y0], [x1, y1], [x2, y2], **self._edges[edge],
                )

        fig = pyplot.Figure(figsize=figsize)
        ax = fig.subplots()

        node_properties = compute_node_properties()
        draw_points()
        draw_edges()

        #
        # Figure size
        #
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        dx = 0.15 * (xlim[1] - xlim[0])
        dy = 0.15 * (ylim[1] - ylim[0])
        ax.set_xlim(xlim[0] - dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] + dy)

        ax.set_axis_off()
        ax.set_aspect("equal")

        fig.set_tight_layout(True)

        return fig
