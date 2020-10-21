import pandas as pd
import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import Dashboard
import string
import pandas as pd
import datetime
from techminer.core.keywords import Keywords
from techminer.core.thesaurus import text_clustering


class App(Dashboard):
    def __init__(self):

        self.menu = "create_thesaurus"
        self.data = pd.read_csv("corpus.csv")
        self.command_panel = [
            dash.HTML("Column:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.Dropdown(options=sorted(self.data.columns)),
            dash.HTML("Parameters:"),
            dash.RadioButtons(
                options=[
                    ("Fingerprint", "fingerprint"),
                    ("1-gram", "1-gram"),
                    ("2-gram", "2-gram"),
                    ("Porter stemmer", "porter"),
                    ("Snowball stemmer", "snowball"),
                ],
                description="Key method",
            ),
            dash.RadioButtons(
                options=[
                    ("Most frequent", "mostfrequent"),
                    ("Longest", "longest"),
                    ("Shortest", "shortest"),
                ],
                description="Name selection",
            ),
            dash.Text(
                description="Filename:",
                placeholder="Enter a file name (TH_*.txt)",
            ),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                # parameters:
                "column": self.command_panel[1],
                "key_method": self.command_panel[3],
                "name_selection": self.command_panel[4],
                "filename": self.command_panel[5],
            },
        )

        DASH.__init__(self)

    def create_thesaurus(self):

        if len(self.filename) == 0:
            text = "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "A name for the thesaurus file must be specified",
            )
            text += "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Process aborted",
            )
            return widgets.HTML(text)

        th = text_clustering(
            x=self.data[self.column],
            name_strategy=self.name_selection,
            key=self.key_method,
            transformer=None,
        )

        th.to_textfile(self.filename)
        return widgets.HTML(
            "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Thesaurus file created!",
            )
        )
