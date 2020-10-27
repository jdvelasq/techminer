import pandas as pd
import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display
import techminer.core.dashboard as dash
from techminer.core.filter_records import filter_records


class App:
    def __init__(self):

        command_panel = [
            dash.HTML("Top N:", hr=False, margin="0px, 0px, 0px, 5px"),
            widgets.IntSlider(
                value=7,
                min=10,
                max=50,
                step=1,
                disabled=False,
                continuous_update=False,
                orientation="horizontal",
                readout=True,
                readout_format="d",
                layout=Layout(
                    width="90%",
                    display="flex",
                    flex_direction="row",
                    justify_content="center",
                ),
                description=" ",
                style={"description_width": "50px"},
            ),
            widgets.HTML("<hr>", layout=Layout(margin="20px 0px 0px 0px")),
            widgets.HTML("<b>Citations:</b>"),
            widgets.ToggleButtons(
                options=[("Global", True), ("Local", False)],
                disabled=False,
                button_style="",
                layout=Layout(
                    width="auto",
                    display="flex",
                    justify_content="center",
                ),
                style={"button_width": "50px"},
            ),
            widgets.HTML("<hr>", layout=Layout(margin="20px 0px 0px 0px")),
            widgets.HTML("<b>Normalized citations:</b>"),
            widgets.ToggleButtons(
                options=[("Yes", True), ("No", False)],
                value=False,
                disabled=False,
                button_style="",
                layout=Layout(
                    width="auto",
                    display="flex",
                    justify_content="center",
                ),
                style={"button_width": "50px"},
            ),
            widgets.HTML("<hr>", layout=Layout(margin="20px 0px 0px 0px")),
            widgets.HTML("<b>Detailed data:</b>"),
            widgets.ToggleButtons(
                options=[("Yes", True), ("No", False)],
                disabled=False,
                button_style="",
                layout=Layout(
                    width="auto",
                    display="flex",
                    justify_content="center",
                ),
                style={"button_width": "50px"},
            ),
            widgets.HTML("<hr>", layout=Layout(margin="20px 0px 0px 0px")),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                "top_n": command_panel[1],
                "global_citations": command_panel[4],
                "normalized_citations": command_panel[7],
                "detailed_data": command_panel[10],
            },
        )

        #
        # Grid size (Generic)
        #
        self.app_layout = GridspecLayout(
            max(9, len(command_panel) + 1), 4, height="870px"
        )

        #
        # Calculate button (Generic)
        #
        calculate_button = widgets.Button(
            description="Apply",
            layout=Layout(width="auto", border="2px solid gray"),
            style={"button_color": "#BDC3C7"},
        )
        calculate_button.on_click(self.on_click)
        command_panel += [calculate_button]

        #
        # Creates command panel (Generic)
        #
        self.app_layout[:, 0] = widgets.VBox(
            command_panel,
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

    def run(self):
        return self.app_layout

    def on_click(self, button):

        data = filter_records(pd.read_csv("corpus.csv"))
        max_year = data["Year"].dropna().max()
        data["Global_Normalized_Citations"] = data.Global_Citations.map(
            lambda w: round(w / max_year, 3), na_action="ignore"
        )
        data["Local_Normalized_Citations"] = data.Local_Citations.map(
            lambda w: round(w / max_year, 3), na_action="ignore"
        )

        data["Global_Citations"] = data.Global_Citations.map(int, na_action="ignore")

        citations_column = {
            (True, True): "Global_Normalized_Citations",
            (True, False): "Global_Citations",
            (False, True): "Local_Normalized_Citations",
            (False, False): "Local_Citations",
        }[(self.global_citations, self.normalized_citations)]

        data = data.sort_values(citations_column, ascending=False).head(self.top_n)
        data = data.reset_index(drop=True)

        if self.detailed_data:
            data = data[
                [
                    "Authors",
                    "Year",
                    "Title",
                    "Source_title",
                    citations_column,
                ]
            ]
            data["Authors"] = data.Authors.map(lambda w: ", ".join(w.split(";")))
            self.output.clear_output()
            with self.output:
                display(data)

        else:
            self.output.clear_output()
            with self.output:
                for i in range(len(data)):

                    print(
                        data.Authors[i].replace(";", ", ")
                        + ". "
                        + str(data.Year[i])
                        + ". "
                        + data.Title[i]
                        + ".\t"
                        + str(data[citations_column][i])
                    )

    def interactive_output(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
