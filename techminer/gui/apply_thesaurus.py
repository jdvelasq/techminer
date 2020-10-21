import pandas as pd
import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import Dashboard
import string
import pandas as pd
import datetime
from techminer.core.keywords import Keywords
from techminer.core.thesaurus import text_clustering
import glob
from techminer.core.thesaurus import read_textfile
from techminer.core.map import map_


class App(Dashboard):
    def __init__(self):

        self.menu = "apply_thesaurus"
        self.data = pd.read_csv("corpus.csv")
        self.command_panel = [
            dash.HTML("Thesaurus file:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.Dropdown(
                options=glob.glob("TH_*.txt"),
            ),
            dash.HTML("Include unmatched items:"),
            dash.Checkbox(description=""),
            dash.HTML("Apply to column:"),
            dash.Dropdown(options=sorted(self.data.columns)),
            # dash.HTML("New column:"),
            #  dash.Text(
            #      description="",
            #      placeholder="Column name",
            #  ),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                # parameters:
                "thesarus": self.command_panel[1],
                "unmatched": self.command_panel[3],
                "column": self.command_panel[5],
            },
        )

        DASH.__init__(self)

    def apply_thesaurus(self):
        def f_strict(x):
            return th.apply_as_dict(x, strict=True)

        def f_unstrict(x):
            return th.apply_as_dict(x, strict=False)

        th = read_textfile(self.thesaurus)
        th = th.compile_as_dict()

        if self.unmatched is True:
            self.data[self.column] = map_(self.data, self.column, f_unstrict)
        else:
            self.data[self.column] = map_(self.data, self.column, f_strict)

        self.data.to_csv("corpus.csv", index=False)

        return self.data[self.column].head(15)
