import pandas as pd
import ipywidgets as widgets
import techminer.core.dashboard as dash
from techminer.core import DASH
import glob
import string
import pandas as pd
import datetime
from techminer.core.keywords import Keywords
import re


class DASHapp(DASH):
    def __init__(self):

        self.menu = "manage"
        self.data = pd.read_csv("corpus.csv")
        self.command_panel = [
            dash.HTML("Operation:", margin="0px 0px 0px 5px", hr=False),
            dash.RadioButtons(
                options=[
                    "Rename",
                    "Copy",
                    "Delete",
                    "Merge",
                    "Extract country",
                    "Extract institution",
                ]
            ),
            dash.HTML("Parameters:"),
            dash.SelectMultiple(
                description="Column:",
                options=sorted(self.data.columns),
                rows=10,
            ),
            dash.Text(
                description="New column:",
                placeholder="Column name",
            ),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                # parameters:
                "operation": self.command_panel[1],
                "column": self.command_panel[3],
                "new_column": self.command_panel[4],
            },
        )

        DASH.__init__(self)

    def manage(self):

        if (
            self.operation
            in ["Rename", "Copy", "Extract country", "Extract institutions"]
            and len(self.column) != 1
        ):

            text = "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Only one column must be selected",
            )
            text += "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Process aborted",
            )

            return widgets.HTML(text)

        if self.operation in ["Merge"] and len(self.column) < 2:

            text = "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "At least two columns must be selected",
            )
            text += "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Process aborted",
            )
            return widgets.HTML(text)

        if len(self.new_column) == 0 and self.operation != "Delete":
            text = self.new_column
            text += "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "A name for the new column must be specified",
            )
            text += "<pre>{} - INFO - {}</pre>".format(
                datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Process aborted",
            )
            return widgets.HTML(text)

        valid_set = string.ascii_letters + string.digits + "_"
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
            return widgets.HTML(text)

        if self.operation == "Rename":
            self.data = self.data.rename(columns={self.column[0]: self.new_column})
            self.data.to_csv("corpus.csv", index=False)
            self.command_panel[3].options = sorted(self.data.columns)
            return self.data[self.new_column].head()

        if self.operation == "Copy":
            self.data[self.new_column] = self.data[self.column[0]].copy()
            self.data.to_csv("corpus.csv", index=False)
            self.command_panel[3].options = sorted(self.data.columns)
            return self.data[self.new_column].head()

        if self.operation == "Delete":
            for col in self.column:
                self.data.pop(col)
            self.data.to_csv("corpus.csv", index=False)
            self.command_panel[3].options = sorted(self.data.columns)
            return widgets.HTML("Column deleted.")

        if self.operation == "Merge":
            self.data[self.new_column] = self.data[self.column[0]].copy()

            for col in self.column[1:]:
                not_na = self.data[col].map(lambda w: not pd.isna(w))
                self.data.loc[not_na, self.new_column] = (
                    self.data.loc[not_na, self.new_column]
                    + ";"
                    + self.data.loc[not_na, col]
                )

                is_na = self.data[self.new_column].map(lambda w: pd.isna(w))
                self.data.loc[is_na, self.new_column] = self.data.loc[is_na, col]

            self.data[self.new_column] = self.data[self.new_column].map(
                lambda w: ";".join(sorted(set(w.split(";")))), na_action="ignore"
            )

            self.data.to_csv("corpus.csv", index=False)
            self.command_panel[3].options = sorted(self.data.columns)
            return self.data[self.new_column].head()
