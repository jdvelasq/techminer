from techminer.gui.bigraph_analysis import App
import pandas as pd

import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display


class App:
    def __init__(self) -> None:

        self.app_layout = GridspecLayout(9, 4, height="870px")
        self.output = widgets.Output().add_class("output_color")
        self.app_layout[0:, 0:] = widgets.VBox(
            [self.output],
            layout=Layout(margin="10px 10px 10px 10px", border="1px solid gray"),
        )

    def run(self):

        x = pd.read_csv("corpus.csv")
        columns = sorted(x.columns)

        with self.output:
            display(
                pd.DataFrame(
                    {
                        "Column": columns,
                        "Number of items": [
                            len(x) - x[col].isnull().sum() for col in columns
                        ],
                        "Coverage (%)": [
                            "{:5.2%}".format((len(x) - x[col].isnull().sum()) / len(x))
                            for col in columns
                        ],
                    }
                )
            )

        return self.app_layout
