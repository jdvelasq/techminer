import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import GridspecLayout, Layout

from techminer.core import corpus_filter, explode, record_to_HTML
import techminer.core.dashboard as dash
from techminer.core.filter_records import filter_records


class App:
    def __init__(self):

        #
        # Data
        #
        self.data = pd.read_csv("corpus.csv")

        #
        # Left panel controls
        #
        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            widgets.Checkbox(
                value=True,
                description="Show only abstract",
            ),
            dash.HTML("First column:"),
            dash.Dropdown(
                options=sorted(self.data.columns),
                description="Column:",
            ),
            dash.Dropdown(
                options=[],
                description="Value:",
            ),
            dash.HTML("Second column:"),
            dash.Dropdown(
                options=sorted(self.data.columns),
                description="Column:",
            ),
            dash.Dropdown(
                options=[],
                description="Value:",
            ),
            dash.HTML("Found articles:"),
            widgets.Select(
                options=[],
                layout=Layout(
                    height="320pt", width="auto", margin="0px, 0px, 0px, 5px"
                ),
            ),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                "only_abstract": self.command_panel[1],
                "main_column": self.command_panel[3],
                "first_value": self.command_panel[4],
                "by_column": self.command_panel[6],
                "second_value": self.command_panel[7],
                "article_title": self.command_panel[9],
            },
        )

        #
        # Grid size (Generic)
        #
        self.app_layout = GridspecLayout(
            max(9, len(self.command_panel) + 1), 4, height="820px"
        )

        #
        # Creates command panel (Generic)
        #
        self.app_layout[:, 0] = widgets.VBox(
            self.command_panel,
            layout=Layout(
                margin="10px 8px 5px 10px",
            ),
        )

        #
        # Output area (Generic)
        #
        self.output = widgets.Output().add_class("output_color")
        self.app_layout[0:, 1:] = widgets.VBox(
            [self.output],
            layout=Layout(margin="10px 4px 4px 4px", border="1px solid gray"),
        )

        self.execute()

    def run(self):
        return self.app_layout

    def interactive_output(self, **kwargs):

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        self.execute()

    def execute(self):

        #
        # Populate main_column with top_n terms
        #
        xdf = self.data.copy()
        xdf["_key1_"] = xdf[self.main_column]
        xdf["_key2_"] = xdf[self.by_column]
        xdf = explode(xdf, "_key1_")

        top_terms = pd.Series(xdf["_key1_"].unique())
        top_terms = top_terms[top_terms.map(lambda w: not pd.isna(w))]
        top_terms = top_terms.sort_values()
        self.command_panel[4].options = top_terms

        #
        # Keyword selection
        #
        keyword1 = self.command_panel[4].value

        #
        # Subset selection
        #
        xdf = explode(xdf, "_key2_")
        xdf = xdf[xdf["_key1_"] == keyword1]
        terms = sorted(pd.Series(xdf["_key2_"].dropna().unique()))

        self.command_panel[7].options = terms

        #
        # Keyword selection
        #
        keyword2 = self.command_panel[7].value

        #
        # Title
        #
        xdf = xdf[xdf["_key2_"] == keyword2]
        if len(xdf):
            self.command_panel[9].options = sorted(xdf["Title"].tolist())
        else:
            self.command_panel[9].options = []
        #
        # Print info from selected title
        #
        out = self.data[self.data["Title"] == self.command_panel[9].value]
        out = out.reset_index(drop=True)
        out = out.iloc[0]
        self.output.clear_output()
        with self.output:
            display(
                widgets.HTML(
                    record_to_HTML(
                        out,
                        only_abstract=self.only_abstract,
                        keywords_to_highlight=[keyword1, keyword2],
                    )
                )
            )


#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
