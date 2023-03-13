import numpy as np
import pandas as pd

import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display

from techminer.core import explode

from techminer.core.filter_records import filter_records


class App:
    def __init__(self) -> None:

        self.app_layout = GridspecLayout(9, 4, height="870px")
        self.output = widgets.Output().add_class("output_color")
        self.app_layout[0:, 0:] = widgets.VBox(
            [self.output],
            layout=Layout(margin="10px 10px 10px 10px", border="1px solid gray"),
        )

    def run(self):

        x = filter_records(pd.read_csv("corpus.csv"))
        self.data = x.copy()
        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    "Source_title",
                    "Num_Documents",
                    "ID",
                ]
            ],
            "Source_title",
        )
        m = x.groupby("Source_title", as_index=True).agg(
            {
                "Num_Documents": np.sum,
            }
        )
        m = m[["Num_Documents"]]
        m = m.groupby(["Num_Documents"]).size()
        w = [str(round(100 * a / sum(m), 2)) + " %" for a in m]
        m = pd.DataFrame(
            {"Num Sources": m.tolist(), "%": w, "Documents published": m.index}
        )

        m = m.sort_values(["Documents published"], ascending=False)
        m["Acum Num Sources"] = m["Num Sources"].cumsum()
        m["% Acum"] = [
            str(round(100 * a / sum(m["Num Sources"]), 2)) + " %"
            for a in m["Acum Num Sources"]
        ]

        m["Tot Documents published"] = m["Num Sources"] * m["Documents published"]
        m["Num Documents"] = m["Tot Documents published"].cumsum()
        m["Tot Documents"] = m["Num Documents"].map(
            lambda w: str(round(w / m["Num Documents"].max() * 100, 2)) + " %"
        )

        bradford1 = int(len(self.data) / 3)
        bradford2 = 2 * bradford1

        m["Bradford's Group"] = m["Num Documents"].map(
            lambda w: 3 if w > bradford2 else (2 if w > bradford1 else 1)
        )

        m = m[
            [
                "Num Sources",
                "%",
                "Acum Num Sources",
                "% Acum",
                "Documents published",
                "Tot Documents published",
                "Num Documents",
                "Tot Documents",
                "Bradford's Group",
            ]
        ]

        m = m.reset_index(drop=True)

        with self.output:
            display(m)

        return self.app_layout
