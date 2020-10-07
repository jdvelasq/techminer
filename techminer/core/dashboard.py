import datetime
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from ipywidgets import GridspecLayout, Layout

COLORMAPS = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "ocean",
    "gnuplot",
    "gnuplot2",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "binary",
    "gist_yarg",
    "gist_gray",
    "gray",
    "bone",
    "pink",
    "spring",
    "summer",
    "autumn",
    "winter",
    "cool",
    "Wistia",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
    "twilight",
    "twilight_shifted",
    "hsv",
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    "flag",
    "prism",
    "gist_earth",
    "terrain",
    "gist_stern",
    "CMRmap",
    "cubehelix",
    "brg",
    "gist_rainbow",
    "rainbow",
    "jet",
    "nipy_spectral",
    "gist_ncar",
]

#
# Common controls GUI definition
#
def affinity():
    return widgets.Dropdown(
        description="Affinity",
        options=["euclidean", "l1", "l2", "manhattan", "cosine"],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def ascending():
    return widgets.Dropdown(
        description="Ascending:",
        options=[True, False],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def c_axis_ascending():
    return widgets.Dropdown(
        description="C-axis ascending:",
        options=[True, False],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def r_axis_ascending():
    return widgets.Dropdown(
        description="R-axis ascending:",
        options=[True, False],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def clustering_method():
    return widgets.Dropdown(
        description="Clustering Method:",
        options=[
            "Affinity Propagation",
            "Agglomerative Clustering",
            "Birch",
            "DBSCAN",
            "KMeans",
            "Mean Shift",
        ],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def cmap(description="Colormap:"):
    return widgets.Dropdown(
        description=description,
        options=COLORMAPS,
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def color_scheme():
    return widgets.Dropdown(
        description="Color Scheme:",
        options=[
            "4 Quadrants",
            "Clusters",
            "Greys",
            "Purples",
            "Blues",
            "Greens",
            "Oranges",
            "Reds",
        ],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def decomposition_method():
    return widgets.Dropdown(
        description="Decompostion method:",
        options=["Factor Analysis", "PCA", "Fast ICA", "SVD"],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def dropdown(description, options):
    return widgets.Dropdown(
        description=description,
        options=options,
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def fig_height():
    return widgets.Dropdown(
        description="Height:",
        options=range(5, 26, 1),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def fig_width():
    return widgets.Dropdown(
        description="Width:",
        options=range(5, 26, 1),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def linkage():
    return widgets.Dropdown(
        description="Linkage:",
        options=["ward", "complete", "average", "single"],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def max_iter():
    return widgets.Dropdown(
        description="Max iterations:",
        options=list(range(50, 501, 50)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def max_items(description="Max items:"):
    return widgets.Dropdown(
        description=description,
        options=list(range(5, 40, 1))
        + list(range(40, 100, 5))
        + list(range(100, 3001, 100)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def n_labels():
    return widgets.Dropdown(
        description="N labels:",
        options=list(range(5, 151, 5)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def min_occurrence(description="Min OCC:"):
    return widgets.Dropdown(
        description=description,
        options=list(range(1, 21)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def n_iter():
    return widgets.Dropdown(
        description="Iterations:",
        options=list(range(5, 51, 1)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def n_clusters(m=3, n=21, i=1):
    return widgets.Dropdown(
        description="N Clusters:",
        options=list(range(m, n, i)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def n_clusters_ac():
    return widgets.Dropdown(
        options=["None"] + list(range(2, 21)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def n_components():
    return widgets.Dropdown(
        description="N components:",
        options=list(range(2, 11)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def normalization(include_none=True):
    options = sorted(
        [
            "Association",
            "Jaccard",
            "Dice",
            "Salton/Cosine",
            "Equivalence",
            "Inclusion",
            "Mutual Information",
        ]
    )
    if include_none is True:
        options = ["None"] + options
    return widgets.Dropdown(
        description="Normalization:",
        options=options,
        layout=Layout(width="auto"),
        value="Association",
        style={"description_width": "140px"},
    )


def nx_scale():
    return widgets.Dropdown(
        description="NX scale:",
        options=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
        ],
        layout=Layout(width="auto"),
        value=1.0,
        style={"description_width": "140px"},
    )


def nx_k():
    return widgets.Dropdown(
        description="NX K:",
        options=[
            0.00001,
            0.0001,
            0.001,
            0.005,
            0.01,
            0.05,
        ]
        + [k / 10 for k in range(1, 21)]
        + [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def nx_iterations():
    return widgets.Dropdown(
        description="NX iterations:",
        options=list(range(5, 101, 1)),
        layout=Layout(width="auto"),
        value=50,
        style={"description_width": "140px"},
    )


def nx_layout():
    return widgets.Dropdown(
        description="Layout:",
        options=[
            "Circular",
            "Kamada Kawai",
            "Planar",
            "Random",
            "Spectral",
            "Spring",
            "Shell",
        ],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def random_state():
    return widgets.Dropdown(
        description="Random State:",
        options=[
            "0012345",
            "0123456",
            "0234567",
            "0345678",
            "0456789",
            "0567890",
            "0678901",
            "0789012",
            "0890123",
            "0901234",
            "1012345",
            "1123456",
            "1234567",
            "1345678",
            "1456789",
            "1567890",
            "1678901",
            "1789012",
            "1890123",
            "1901234",
            "2012345",
            "2123456",
            "2234567",
            "2345678",
            "2456789",
            "2567890",
            "2678901",
            "2789012",
            "2890123",
            "2901234",
            "3012345",
        ],
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def separator(text):
    return widgets.HTML("<b>" + text + "</b><hr>")


def top_n(m=10, n=51, i=5):
    return widgets.Dropdown(
        description="Top N:",
        options=list(range(m, n, i)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def x_axis(n=10):
    return widgets.Dropdown(
        description="X-axis:",
        options=list(range(n)),
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


def y_axis(n=10, value=1):
    return widgets.Dropdown(
        description="Y-axis:",
        options=list(range(n)),
        value=1,
        layout=Layout(width="auto"),
        style={"description_width": "140px"},
    )


################################################################
################################################################


class DASH:
    def __init__(self):

        ## layout
        self.app_layout = []

        ## Panel controls
        self.command_panel = []
        self.output = None

        ## display pandas options
        self.pandas_max_rows = 100
        self.pandas_max_columns = 100

    def logging_info(self, msg):
        with self.output:
            print(
                "{} - INFO - {}".format(
                    datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), msg
                )
            )

    def interactive_output(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def run(self):

        #
        # Grid size
        #
        self.app_layout = GridspecLayout(
            max(9, len(self.command_panel) + 1), 4, height="820px"
        )

        #
        # Calculate button
        #
        calculate_button = widgets.Button(
            description="Apply",
            layout=Layout(width="auto", border="2px solid gray"),
            #  button_style="warning",
            style={"button_color": "#BDC3C7"},
        )
        calculate_button.on_click(self.on_click)
        self.command_panel += [calculate_button]

        #
        # Creates left panel
        #
        self.app_layout[:, 0] = widgets.VBox(
            self.command_panel,
            layout=Layout(
                margin="10px 0px 4px 4px",
            ),
        )

        #
        # Output area
        #
        self.output = widgets.Output().add_class("output_color")
        self.app_layout[0:, 1:] = widgets.VBox(
            [self.output],
            layout=Layout(margin="10px 4px 4px 4px", border="1px solid gray"),
        )

        #
        # interactive
        #
        args = {}
        for widget in self.command_panel:
            if widget._model_name not in ["HTMLModel", "ButtonModel"]:
                description = self.text_transform(widget.description)
                args = {**args, **{description: widget}}

        widgets.interactive_output(
            self.interactive_output,
            args,
        )

        return self.app_layout

    def text_transform(self, text):
        return (
            text.replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(":", "")
            .lower()
        )

    def on_click(self, button):

        menu = (
            self.menu.replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .lower()
        )
        self.output.clear_output()
        with self.output:
            print("Processing ...")
        result = getattr(self, menu)()
        self.output.clear_output()
        with pd.option_context(
            "max_rows", self.pandas_max_rows, "max_columns", self.pandas_max_columns
        ):
            with self.output:
                display(result)

    def set_enabled(self, name):
        for widget in self.command_panel:
            x = self.text_transform(widget.description)
            name = self.text_transform(name)
            if x == name:
                widget.disabled = False
                return

    def set_disabled(self, name):
        for widget in self.command_panel:
            x = self.text_transform(widget.description)
            name = self.text_transform(name)
            if x == name:
                widget.disabled = True
                return

    # def set_options(self, name, options):
    #     for index, _ in enumerate(self.command_panel):
    #         x = self.text_transform(self.command_panel[index]["desc"])
    #         name = self.text_transform(name)
    #         if x == name:
    #             self.command_panel[index]["widget"].options = options
    #             return

    def enable_disable_clustering_options(self, include_random_state=False):

        if self.clustering_method in ["Affinity Propagation"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_enabled("Random State:")

        if self.clustering_method in ["Agglomerative Clustering"]:
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["Birch"]:
            self.set_enabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["DBSCAN"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["Feature Agglomeration"]:
            self.set_enabled("N Clusters:")
            self.set_enabled("Affinity:")
            self.set_enabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["KMeans"]:
            self.set_enabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")

        if self.clustering_method in ["Mean Shift"]:
            self.set_disabled("N Clusters:")
            self.set_disabled("Affinity:")
            self.set_disabled("Linkage:")
            if include_random_state is True:
                self.set_disabled("Random State:")


#  def processing():
#     html = """
#         <style>
#         .loader {
#         border: 16px solid #f3f3f3;
#         border-radius: 50%;
#         border-top: 16px solid #3498db;
#         width: 70px;
#         height: 70px;
#         -webkit-animation: spin 2s linear infinite; /* Safari */
#         animation: spin 2s linear infinite;
#         }

#         /* Safari */
#         @-webkit-keyframes spin {
#         0% { -webkit-transform: rotate(0deg); }
#         100% { -webkit-transform: rotate(360deg); }
#         }

#         @keyframes spin {
#         0% { transform: rotate(0deg); }
#         100% { transform: rotate(360deg); }
#         }
#         </style>
#         </head>
#         <h3>Processing ... </h3>
#         <div class="loader"></div>

#         """
#     return widgets.HTML(html)
