from techminer.bigraph_analysis import DASHapp
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import GridspecLayout, Layout

from techminer.core import explode, record_to_HTML


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
                "Bradford_Law_Zone",
                "Abstract_phrase_words",
            ]
        )

        # -------------------------------------------------------------------------
        #
        # UI
        #
        # -------------------------------------------------------------------------
        left_panel = [
            # 0
            {
                "arg": "column",
                "desc": "Column:",
                "widget": widgets.Dropdown(
                    options=[z for z in COLUMNS if z in df.columns],
                    layout=Layout(width="auto"),
                ),
            },
            # 1
            {
                "arg": "value",
                "desc": "Term:",
                "widget": widgets.Dropdown(
                    options=[],
                    layout=Layout(width="auto"),
                ),
            },
            # 2
            {
                "arg": "title",
                "desc": "Title:",
                "widget": widgets.Select(
                    options=[],
                    layout=Layout(height="380pt", width="auto"),
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
            column = kwargs["column"]

            x = explode(df, column)

            if self.top_n is not None:
                #
                # Populate value control with top_n terms
                #
                y = df.copy()
                y["Num_Documents"] = 1
                y = explode(
                    y[
                        [
                            column,
                            "Num_Documents",
                            "Global_Citations",
                            "ID",
                        ]
                    ],
                    column,
                )
                y = y.groupby(column, as_index=True).agg(
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
                all_terms = pd.Series(x[column].unique())
                all_terms = all_terms[all_terms.map(lambda w: not pd.isna(w))]
                all_terms = all_terms.sort_values()
                left_panel[1]["widget"].options = all_terms
            #
            # Populate titles
            #
            keyword = left_panel[1]["widget"].value
            s = x[x[column] == keyword]
            s = s[["Global_Citations", "Title"]]
            s = s.sort_values(["Global_Citations", "Title"], ascending=[False, True])
            s = s[["Title"]].drop_duplicates()
            left_panel[2]["widget"].options = s["Title"].tolist()

            #
            # Print info from selected title
            #
            out = df[df["Title"] == left_panel[2]["widget"].value]
            out = out.reset_index(drop=True)
            out = out.iloc[0]
            output.clear_output()
            with output:
                display(
                    widgets.HTML(
                        record_to_HTML(
                            out,
                            only_abstract=self.only_abstract,
                            keywords_to_highlight=[keyword],
                        )
                    )
                )

        # -------------------------------------------------------------------------
        #
        # Generic
        #
        # -------------------------------------------------------------------------
        args = {control["arg"]: control["widget"] for control in left_panel}
        output = widgets.Output().add_class("output_color")
        with output:
            display(
                widgets.interactive_output(
                    server,
                    args,
                )
            )

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

        grid[len(left_panel) - 1 :, 0] = widgets.VBox(
            [
                widgets.Label(value=left_panel[-1]["desc"]),
                left_panel[-1]["widget"],
            ],
            layout=Layout(
                margin="0px 0px 4px 4px",
            ),
        )

        grid[0:, 1:] = widgets.VBox(
            [
                output,
            ],
            layout=Layout(margin="10px 4px 4px 4px", border="1px solid gray"),
        )

        return grid
