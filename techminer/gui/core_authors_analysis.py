import matplotlib
import matplotlib.pyplot as plt
from techminer.core import explode
import pandas as pd
import numpy as np

import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display
from techminer.core import Dashboard
import techminer.core.dashboard as dash

from techminer.core.filter_records import filter_records

###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(self, data):
        self.data = data

    def core_authors(self):
        x = self.data.copy()

        ##
        ##  Num_Documents per Author
        ##
        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    "Authors",
                    "Num_Documents",
                    "ID",
                ]
            ],
            "Authors",
        )
        result = x.groupby("Authors", as_index=True).agg(
            {
                "Num_Documents": np.sum,
            }
        )
        z = result
        authors_dict = {
            author: num_docs
            for author, num_docs in zip(z.index, z.Num_Documents)
            if not pd.isna(author)
        }

        ##
        ##  Num Authors x Documents written per Author
        ##
        z = z[["Num_Documents"]]
        z = z.groupby(["Num_Documents"]).size()
        w = [str(round(100 * a / sum(z), 2)) + " %" for a in z]
        z = pd.DataFrame(
            {"Num Authors": z.tolist(), "%": w, "Documents written per Author": z.index}
        )
        z = z.sort_values(["Documents written per Author"], ascending=False)
        z["Acum Num Authors"] = z["Num Authors"].cumsum()
        z["% Acum"] = [
            str(round(100 * a / sum(z["Num Authors"]), 2)) + " %"
            for a in z["Acum Num Authors"]
        ]
        m = explode(self.data[["Authors", "ID"]], "Authors")
        m = m.dropna()
        m["Documents_written"] = m.Authors.map(lambda w: authors_dict[w])

        n = []
        for k in z["Documents written per Author"]:
            s = m.query("Documents_written >= " + str(k))
            s = s[["ID"]]
            s = s.drop_duplicates()
            n.append(len(s))

        k = []
        for index in range(len(n) - 1):
            k.append(n[index + 1] - n[index])
        k = [n[0]] + k
        z["Num Documents"] = k
        z["% Num Documents"] = [str(round(i / max(n) * 100, 2)) + "%" for i in k]
        z["Acum Num Documents"] = n
        z["% Acum Num Documents"] = [str(round(i / max(n) * 100, 2)) + "%" for i in n]

        z = z[
            [
                "Num Authors",
                "%",
                "Acum Num Authors",
                "% Acum",
                "Documents written per Author",
                "Num Documents",
                "% Num Documents",
                "Acum Num Documents",
                "% Acum Num Documents",
            ]
        ]

        z = z.reset_index(drop=True)

        return z

    def lotka_law_plot(self):

        data = self.core_authors()
        percentage_authors = data["%"].map(lambda w: float(w[:-2])).tolist()
        documents_written = data["Documents written per Author"].tolist()

        matplotlib.rc("font", size=11)
        fig = plt.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = plt.cm.get_cmap(self.colormap)
        color = cmap(0.6)

        percentage_authors.reverse()
        documents_written.reverse()

        ax.plot(
            documents_written,
            percentage_authors,
            linestyle="-",
            linewidth=2,
            color="k",
        )
        ax.fill_between(
            documents_written,
            percentage_authors,
            color=color,
            alpha=0.6,
        )

        ##
        ## Theoretical
        ##
        total_authors = data["Num Authors"].max()
        theoretical = [total_authors / float(x * x) for x in documents_written]
        total_theoretical = sum(theoretical)
        perc_theoretical_authors = [w / total_theoretical * 100 for w in theoretical]

        ax.plot(
            documents_written,
            perc_theoretical_authors,
            linestyle=":",
            linewidth=4,
            color="k",
        )

        for x in ["top", "right", "left", "bottom"]:
            ax.spines[x].set_visible(False)

        ax.grid(axis="y", color="gray", linestyle=":")
        ax.grid(axis="x", color="gray", linestyle=":")
        ax.set_ylabel("% of Authors")
        ax.set_xlabel("Documets written per Author")

        fig.set_tight_layout(True)

        return fig


class App(Dashboard, Model):
    def __init__(self):

        data = filter_records(pd.read_csv("corpus.csv"))

        Model.__init__(self, data=data)

        self.command_panel = [
            dash.HTML("Display:", hr=False, margin="0px, 0px, 0px, 5px"),
            dash.RadioButtons(
                options=["Core Authors", "Lotka Law plot"],
                description="",
            ),
            dash.HTML("Colormap:"),
            dash.cmap(description=None),
            dash.HTML("Figure size:"),
            dash.fig_width(slider=True),
            dash.fig_height(slider=True),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                # Display:
                "menu": self.command_panel[1],
                # Colormap:
                "colormap": self.command_panel[3],
                # FIgure size
                "width": self.command_panel[5],
                "height": self.command_panel[6],
            },
        )

        Dashboard.__init__(self)

        self.interactive_output(
            **{
                # Display:
                "menu": self.command_panel[1].value,
                # Visualization:
                "colormap": self.command_panel[3].value,
                "width": self.command_panel[5].value,
                "height": self.command_panel[6].value,
            }
        )

    def interactive_output(self, **kwargs):

        Dashboard.interactive_output(self, **kwargs)

        if self.menu == "Core Authors":
            self.set_disabled("Colormap:")
            self.set_disabled("Width:")
            self.set_disabled("Height:")

        if self.menu == "Lotka Law plot":
            self.set_enabled("Colormap:")
            self.set_enabled("Width:")
            self.set_enabled("Height:")
