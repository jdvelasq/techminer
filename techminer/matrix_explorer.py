"""
Data Viewer
==================================================================================================


"""

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import GridspecLayout, Layout

from techminer.core import corpus_filter, explode, record_to_HTML


class DASHapp:
    def __init__(self, top_n=100, only_abstract=True):
        self.top_n = top_n
        self.only_abstract = only_abstract

    def run(self):

        df = pd.read_csv("corpus.csv")

        COLUMNS = sorted(
            [
                "Author_Keywords",
                "Author_Keywords_CL",
                "Authors",
                "Countries",
                "Country_1st_Author",
                "Index_Keywords",
                "Index_Keywords_CL",
                "Institution_1st_Author",
                "Institutions",
                "Keywords_CL",
                "Source_title",
                "Title",
                "Year",
                "Abstract_words",
                "Abstract_words_CL",
                "Title_words",
                "Title_words_CL",
            ]
        )
        #
        # -------------------------------------------------------------------------
        #
        # UI
        #
        # -------------------------------------------------------------------------
        left_panel = [
            # 0
            {
                "arg": "main_column",
                "desc": "Column:",
                "widget": widgets.Dropdown(
                    options=[z for z in COLUMNS if z in df.columns],
                    layout=Layout(width="auto"),
                ),
            },
            # 1
            {
                "arg": "main_term",
                "desc": "Term in Column:",
                "widget": widgets.Dropdown(
                    options=[],
                    layout=Layout(width="auto"),
                ),
            },
            # 2
            {
                "arg": "by_column",
                "desc": "By column:",
                "widget": widgets.Dropdown(
                    options=[z for z in COLUMNS if z in df.columns],
                    layout=Layout(width="auto"),
                ),
            },
            # 3
            {
                "arg": "by_term",
                "desc": "Term in By column:",
                "widget": widgets.Dropdown(
                    options=[],
                    layout=Layout(width="auto"),
                ),
            },
            # 4
            {
                "arg": "title",
                "desc": "Title:",
                "widget": widgets.Select(
                    options=[],
                    layout=Layout(height="270pt", width="auto"),
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
            # Columns
            #
            main_column = kwargs["main_column"]
            by_column = kwargs["by_column"]

            #
            # Populate main_column with top_n terms
            #
            xdf = df.copy()
            xdf["_key1_"] = xdf[main_column]
            xdf["_key2_"] = xdf[by_column]
            #  if main_column in MULTIVALUED_COLS:
            #    xdf = explode(xdf, "_key1_")
            xdf = explode(xdf, "_key1_")

            if self.top_n is not None:

                y = xdf.copy()
                y["Num_Documents"] = 1
                y = explode(
                    y[
                        [
                            "_key1_",
                            "Num_Documents",
                            "Global_Citations",
                            "ID",
                        ]
                    ],
                    "_key1_",
                )
                y = y.groupby("_key1_", as_index=True).agg(
                    {
                        "Num_Documents": np.sum,
                        "Global_Citations": np.sum,
                    }
                )
                y["Global_Citations"] = y["Global_Citations"].map(lambda w: int(w))
                top_terms_freq = set(
                    y.sort_values("Num_Documents", ascending=False)
                    .head(self.top_n)
                    .index
                )
                top_terms_cited_by = set(
                    y.sort_values("Global_Citations", ascending=False)
                    .head(self.top_n)
                    .index
                )
                top_terms = sorted(top_terms_freq | top_terms_cited_by)
                left_panel[1]["widget"].options = top_terms
            else:
                top_terms = pd.Series(xdf["_key1_"].unique())
                top_terms = top_terms[top_terms.map(lambda w: not pd.isna(w))]
                top_terms = top_terms.sort_values()
                left_panel[1]["widget"].options = top_terms

            #
            # Keyword selection
            #
            keyword1 = left_panel[1]["widget"].value

            #
            # Subset selection
            #
            #  if by_column in MULTIVALUED_COLS:
            #    xdf = explode(xdf, "_key2_")
            xdf = explode(xdf, "_key2_")
            xdf = xdf[xdf["_key1_"] == keyword1]
            terms = sorted(pd.Series(xdf["_key2_"].dropna().unique()))

            left_panel[3]["widget"].options = terms

            #
            # Keyword selection
            #
            keyword2 = left_panel[3]["widget"].value

            #
            # Title
            #
            xdf = xdf[xdf["_key2_"] == keyword2]
            if len(xdf):
                left_panel[4]["widget"].options = sorted(xdf["Title"].tolist())
            else:
                left_panel[4]["widget"].options = []
            #
            # Print info from selected title
            #
            out = df[df["Title"] == left_panel[4]["widget"].value]
            out = out.reset_index(drop=True)
            out = out.iloc[0]
            output.clear_output()
            with output:
                display(
                    widgets.HTML(
                        record_to_HTML(
                            out,
                            only_abstract=self.only_abstract,
                            keywords_to_highlight=[keyword1, keyword2],
                        )
                    )
                )

        # -------------------------------------------------------------------------
        #
        # Generic
        #
        # -------------------------------------------------------------------------
        args = {control["arg"]: control["widget"] for control in left_panel}
        output = widgets.Output()
        with output:
            display(
                widgets.interactive_output(
                    server,
                    args,
                )
            )
        #
        #
        grid = GridspecLayout(10, 4, height="820px")
        for i in range(0, len(left_panel) - 1):
            grid[i, 0] = widgets.VBox(
                [
                    widgets.Label(value=left_panel[i]["desc"]),
                    left_panel[i]["widget"],
                ],
                layout=Layout(
                    margin="0px 0px 4px 4px",
                ),
            )

        grid[len(left_panel) - 1 : len(left_panel) + 5, 0] = widgets.VBox(
            [
                widgets.Label(value=left_panel[-1]["desc"]),
                left_panel[-1]["widget"],
            ],
            layout=Layout(
                margin="0px 0px 4px 4px",
            ),
        )

        grid[:, 1:] = widgets.VBox(
            [output],
            layout=Layout(margin="10px 4px 4px 4px", border="1px solid gray"),
        )

        return grid


#
#
#
if __name__ == "__main__":

    import doctest

    doctest.testmod()
