from techminer.gui.bigraph_analysis import App
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import GridspecLayout, Layout

from techminer.core import explode, record_to_HTML
import techminer.core.dashboard as dash


class App:
    def __init__(self, top_n=100, only_abstract=True):

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
            dash.HTML("Column:"),
            dash.Dropdown(options=sorted(self.data.columns)),
            dash.HTML("Term:"),
            dash.Dropdown(options=[]),
            dash.HTML("Found articles:"),
            widgets.Select(
                options=[],
                layout=Layout(height="360pt", width="auto"),
            ),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                "only_abstract": self.command_panel[1],
                "column": self.command_panel[3],
                "value": self.command_panel[5],
                "article_title": self.command_panel[7],
            },
        )

        #
        # Grid size (Generic)
        #
        self.app_layout = GridspecLayout(
            max(9, len(self.command_panel) + 1), 4, height="870px"
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

    def execute(self):

        with self.output:

            column = self.column

            x = explode(self.data, column)

            all_terms = pd.Series(x[column].unique())
            all_terms = all_terms[all_terms.map(lambda w: not pd.isna(w))]
            all_terms = all_terms.sort_values()
            self.command_panel[5].options = all_terms

            #
            # Populate titles
            #
            keyword = self.command_panel[5].value
            s = x[x[column] == keyword]
            s = s[["Global_Citations", "Title"]]
            s = s.sort_values(["Global_Citations", "Title"], ascending=[False, True])
            s = s[["Title"]].drop_duplicates()
            self.command_panel[7].options = s["Title"].tolist()

            #
            # Print info from selected title
            #
            out = self.data[self.data["Title"] == self.command_panel[7].value]
            out = out.reset_index(drop=True)
            out = out.iloc[0]
            self.output.clear_output()
            with self.output:
                display(
                    widgets.HTML(
                        record_to_HTML(
                            out,
                            only_abstract=self.only_abstract,
                            keywords_to_highlight=[keyword],
                        )
                    )
                )

    def interactive_output(self, **kwargs):

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        self.execute()

    # def run_(self):

    #     COLUMNS = sorted(
    #         [
    #             "Author_Keywords",
    #             "Author_Keywords_CL",
    #             "Authors",
    #             "Countries",
    #             "Country_1st_Author",
    #             "Index_Keywords",
    #             "Index_Keywords_CL",
    #             "Institution_1st_Author",
    #             "Institutions",
    #             "Keywords_CL",
    #             "Source_title",
    #             "Title",
    #             "Year",
    #             "Abstract_words",
    #             "Abstract_words_CL",
    #             "Title_words",
    #             "Title_words_CL",
    #             "Bradford_Law_Zone",
    #             "Abstract_phrase_words",
    #         ]
    #     )

    #     # -------------------------------------------------------------------------
    #     #
    #     # UI
    #     #
    #     # -------------------------------------------------------------------------
    #     left_panel = [
    #         # 0
    #         {
    #             "arg": "column",
    #             "desc": "Column:",
    #             "widget": widgets.Dropdown(
    #                 options=[z for z in COLUMNS if z in df.columns],
    #                 layout=Layout(width="auto"),
    #             ),
    #         },
    #         # 1
    #         {
    #             "arg": "value",
    #             "desc": "Term:",
    #             "widget": widgets.Dropdown(
    #                 options=[],
    #                 layout=Layout(width="auto"),
    #             ),
    #         },
    #         # 2
    #         {
    #             "arg": "title",
    #             "desc": "Title:",
    #             "widget": widgets.Select(
    #                 options=[],
    #                 layout=Layout(height="380pt", width="auto"),
    #             ),
    #         },
    #     ]
    #     # -------------------------------------------------------------------------
    #     #
    #     # Logic
    #     #
    #     # -------------------------------------------------------------------------
    #     def server(**kwargs):
    #         #
    #         column = kwargs["column"]

    #         x = explode(df, column)

    #         if self.top_n is not None:
    #             #
    #             # Populate value control with top_n terms
    #             #
    #             y = df.copy()
    #             y["Num_Documents"] = 1
    #             y = explode(
    #                 y[
    #                     [
    #                         column,
    #                         "Num_Documents",
    #                         "Global_Citations",
    #                         "ID",
    #                     ]
    #                 ],
    #                 column,
    #             )
    #             y = y.groupby(column, as_index=True).agg(
    #                 {
    #                     "Num_Documents": np.sum,
    #                     "Global_Citations": np.sum,
    #                 }
    #             )
    #             y["Global_Citations"] = y["Global_Citations"].map(lambda w: int(w))
    #             top_terms_freq = set(
    #                 y.sort_values("Num_Documents", ascending=False)
    #                 .head(self.top_n)
    #                 .index
    #             )
    #             top_terms_cited_by = set(
    #                 y.sort_values("Global_Citations", ascending=False)
    #                 .head(self.top_n)
    #                 .index
    #             )
    #             top_terms = sorted(top_terms_freq | top_terms_cited_by)
    #             left_panel[1]["widget"].options = top_terms
    #         else:
    #             all_terms = pd.Series(x[column].unique())
    #             all_terms = all_terms[all_terms.map(lambda w: not pd.isna(w))]
    #             all_terms = all_terms.sort_values()
    #             left_panel[1]["widget"].options = all_terms
    #         #
    #         # Populate titles
    #         #
    #         keyword = left_panel[1]["widget"].value
    #         s = x[x[column] == keyword]
    #         s = s[["Global_Citations", "Title"]]
    #         s = s.sort_values(["Global_Citations", "Title"], ascending=[False, True])
    #         s = s[["Title"]].drop_duplicates()
    #         left_panel[2]["widget"].options = s["Title"].tolist()

    #         #
    #         # Print info from selected title
    #         #
    #         out = df[df["Title"] == left_panel[2]["widget"].value]
    #         out = out.reset_index(drop=True)
    #         out = out.iloc[0]
    #         output.clear_output()
    #         with output:
    #             display(
    #                 widgets.HTML(
    #                     record_to_HTML(
    #                         out,
    #                         only_abstract=self.only_abstract,
    #                         keywords_to_highlight=[keyword],
    #                     )
    #                 )
    #             )

    #     # -------------------------------------------------------------------------
    #     #
    #     # Generic
    #     #
    #     # -------------------------------------------------------------------------
    #     args = {control["arg"]: control["widget"] for control in left_panel}
    #     output = widgets.Output().add_class("output_color")
    #     with output:
    #         display(
    #             widgets.interactive_output(
    #                 server,
    #                 args,
    #             )
    #         )

    #     grid = GridspecLayout(10, 4, height="820px")
    #     for i in range(0, len(left_panel) - 1):
    #         grid[i, 0] = widgets.VBox(
    #             [
    #                 widgets.Label(value=left_panel[i]["desc"]),
    #                 left_panel[i]["widget"],
    #             ],
    #             layout=Layout(
    #                 margin="0px 0px 4px 4px",
    #             ),
    #         )

    #     grid[len(left_panel) - 1 :, 0] = widgets.VBox(
    #         [
    #             widgets.Label(value=left_panel[-1]["desc"]),
    #             left_panel[-1]["widget"],
    #         ],
    #         layout=Layout(
    #             margin="0px 0px 4px 4px",
    #         ),
    #     )

    #     grid[0:, 1:] = widgets.VBox(
    #         [
    #             output,
    #         ],
    #         layout=Layout(margin="10px 4px 4px 4px", border="1px solid gray"),
    #     )

    #     return grid
