import pandas as pd
import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import Dashboard
import glob
import string
import pandas as pd
import datetime
from techminer.core.keywords import Keywords
import re


class App(Dashboard):
    def __init__(self):

        self.menu = "extract_keywords"
        self.data = pd.read_csv("corpus.csv")
        self.command_panel = [
            dash.HTML("Parameters:", margin="0px 0px 0px 5px", hr=False),
            dash.Dropdown(
                description="Terms to extract:",
                options=sorted(self.data.columns),
            ),
            dash.Dropdown(
                description="From column:",
                options=sorted(self.data.columns),
            ),
            dash.Text(
                description="New column:",
                placeholder="Column name",
            ),
            dash.HTML("Options:"),
            dash.Checkbox(description="Full match"),
            dash.Checkbox(description="Ignore case"),
            dash.Checkbox(description="Use re"),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                # parameters:
                "keywords_list": self.command_panel[1],
                "from_column": self.command_panel[2],
                "new_column": self.command_panel[3],
                # Parameters:
                "full_match": self.command_panel[5],
                "ignore_case": self.command_panel[6],
                "use_re": self.command_panel[7],
            },
        )

        Dashboard.__init__(self)

    def extract_keywords(self):

        valid_set = string.ascii_letters + string.digits + "_"

        if len(self.new_column) == 0:
            text = self.new_column
            text += "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "A name for the new column must be specified",
            )
            text += "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Process aborted",
            )
            text += "->" + self.new_column + "<-"
            return widgets.HTML(text)

        if self.new_column.strip(valid_set):
            text = self.new_column
            text += "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Invalid column name",
            )
            text += "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Process aborted",
            )
            text += "-->" + self.new_column + "<--"
            return widgets.HTML(text)

        self.data[self.new_column] = pd.NA

        with open(self.keywords_list, "rt") as f:
            keywords_list = f.readlines()
        keywords_list = [k.replace("\n", "") for k in keywords_list]

        keywords = Keywords(
            ignore_case=self.ignore_case, full_match=self.full_match, use_re=self.use_re
        )
        keywords.add_keywords(keywords_list)
        keywords.compile()

        self.data[self.new_column] = self.data[self.from_column].map(
            lambda w: keywords.extract_from_text(w), na_action="ignore"
        )

        self.data[self.new_column] = self.data[self.new_column].map(
            lambda w: pd.NA if w is None else w, na_action="ignore"
        )

        self.data.to_csv("corpus.csv", index=False)

        return self.data[self.new_column].dropna().head(15)

    def interactive_output(self, **kwargs):
        Dashboard.interactive_output(self, **kwargs)