"""
Analysis by Term
==========================================================================

"""
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from nltk.corpus.reader.ieer import documents

import techminer.core.dashboard as dash
from techminer.core import (
    DASH,
    add_counters_to_axis,
    corpus_filter,
    explode,
    limit_to_exclude,
    sort_axis,
    sort_by_axis,
)
from techminer.plots import (
    bar_plot,
    barh_plot,
    pie_plot,
    stacked_bar,
    stacked_barh,
    treemap,
    wordcloud_,
    worldmap,
)

#  from techminer.core.dashboard import max_items, min_occurrence


###############################################################################
##
##  MODEL
##
###############################################################################


class Model:
    def __init__(
        self,
        data,
        limit_to,
        exclude,
        years_range,
    ):

        if years_range is not None:
            initial_year, final_year = years_range
            data = data[(data.Year >= initial_year) & (data.Year <= final_year)]

        self.data = data
        self.limit_to = limit_to
        self.exclude = exclude

        self.column = None
        self.top_by = None
        self.sort_by = None
        self.ascending = None
        self.colormap = None
        self.height = None
        self.width = None
        self.view = None

    def core_source_titles(self):

        x = self.data.copy()
        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    "Source_title",
                    "Num_Documents",
                    "ID",
                ]
            ],
            "Source_title",
        )
        m = x.groupby("Source_title", as_index=True).agg(
            {
                "Num_Documents": np.sum,
            }
        )
        m = m[["Num_Documents"]]
        m = m.groupby(["Num_Documents"]).size()
        w = [str(round(100 * a / sum(m), 2)) + " %" for a in m]
        m = pd.DataFrame(
            {"Num Sources": m.tolist(), "%": w, "Documents published": m.index}
        )

        m = m.sort_values(["Documents published"], ascending=False)
        m["Acum Num Sources"] = m["Num Sources"].cumsum()
        m["% Acum"] = [
            str(round(100 * a / sum(m["Num Sources"]), 2)) + " %"
            for a in m["Acum Num Sources"]
        ]

        m["Tot Documents published"] = m["Num Sources"] * m["Documents published"]
        m["Num Documents"] = m["Tot Documents published"].cumsum()
        m["Tot Documents"] = m["Num Documents"].map(
            lambda w: str(round(w / m["Num Documents"].max() * 100, 2)) + " %"
        )

        bradford1 = int(len(self.data) / 3)
        bradford2 = 2 * bradford1

        m["Bradford's Group"] = m["Num Documents"].map(
            lambda w: 3 if w > bradford2 else (2 if w > bradford1 else 1)
        )

        m = m[
            [
                "Num Sources",
                "%",
                "Acum Num Sources",
                "% Acum",
                "Documents published",
                "Tot Documents published",
                "Num Documents",
                "Tot Documents",
                "Bradford's Group",
            ]
        ]

        m = m.reset_index(drop=True)
        return m

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

    def most_global_cited_documents(self):

        data = self.data
        data = data.sort_values(["Global_Citations", "Year"], ascending=[False, True])
        data = data.head(50)
        data["Global_Citations"] = data.Global_Citations.map(lambda w: int(w))
        data["Local_Citations"] = data.Local_Citations.map(lambda w: int(w))
        data = data[
            [
                "Authors",
                "Year",
                "Title",
                "Source_title",
                "Global_Citations",
                "Local_Citations",
            ]
        ]
        data = data.reset_index(drop=True)
        return data

    def most_local_cited_documents(self):

        data = self.data
        data = data.sort_values(["Local_Citations", "Year"], ascending=[False, True])
        data["Global_Citations"] = data.Global_Citations.map(lambda w: int(w))
        data["Local_Citations"] = data.Local_Citations.map(lambda w: int(w))
        data = data.head(50)
        data = data[data.Local_Citations > 0]
        data = data[
            [
                "Authors",
                "Year",
                "Title",
                "Source_title",
                "Global_Citations",
                "Local_Citations",
            ]
        ]
        data = data.reset_index(drop=True)
        return data

    def worldmap(self):

        x = self.data.copy()
        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    self.column,
                    "Num_Documents",
                    "Global_Citations",
                    "ID",
                ]
            ],
            self.column,
        )
        result = x.groupby(self.column, as_index=True).agg(
            {
                "Num_Documents": np.sum,
                "Global_Citations": np.sum,
            }
        )
        top_by = self.top_by.replace(" ", "_")
        return worldmap(
            x=result[top_by],
            figsize=(self.width, self.height),
            cmap=self.colormap,
        )

    def single_multiple_publication(self):

        x = self.data.copy()
        x["SD"] = x[self.column].map(
            lambda w: 1 if isinstance(w, str) and len(w.split(";")) == 1 else 0
        )
        x["MD"] = x[self.column].map(
            lambda w: 1 if isinstance(w, str) and len(w.split(";")) > 1 else 0
        )
        x = explode(
            x[
                [
                    self.column,
                    "SD",
                    "MD",
                    "ID",
                ]
            ],
            self.column,
        )
        result = x.groupby(self.column, as_index=False).agg(
            {
                "SD": np.sum,
                "MD": np.sum,
            }
        )
        result["SMR"] = [
            round(MD / max(SD, 1), 2) for SD, MD in zip(result.SD, result.MD)
        ]
        result = result.set_index(self.column)

        ## limit to / exclude options
        result = limit_to_exclude(
            data=result,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ## counters in axis names
        result = add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )

        ## Top by / Top N
        result = sort_by_axis(data=result, sort_by=self.top_by, ascending=False, axis=0)
        result = result.head(self.max_items)

        ## Sort by
        if self.sort_by in result.columns:
            result = result.sort_values(self.sort_by, ascending=self.ascending)
        else:
            result = sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )

        if self.view == "Table":
            return result

        if self.view == "Bar plot":
            return stacked_bar(
                X=result[["SD", "MD"]],
                cmap=self.colormap,
                ylabel="Num Documents",
                figsize=(self.width, self.height),
            )

        if self.view == "Horizontal bar plot":
            return stacked_barh(
                X=result[["SD", "MD"]],
                cmap=self.colormap,
                xlabel="Num Documents",
                figsize=(self.width, self.height),
            )

    def impact(self):
        x = self.data.copy()
        last_year = x.Year.max()
        x["Num_Documents"] = 1
        x["First_Year"] = x.Year
        if self.column == "Authors":
            x = explode(
                x[
                    [
                        self.column,
                        "Frac_Num_Documents",
                        "Num_Documents",
                        "Global_Citations",
                        "First_Year",
                        "ID",
                    ]
                ],
                self.column,
            )
            result = x.groupby(self.column, as_index=False).agg(
                {
                    "Frac_Num_Documents": np.sum,
                    "Num_Documents": np.sum,
                    "Global_Citations": np.sum,
                    "First_Year": np.min,
                }
            )
        else:
            x = explode(
                x[
                    [
                        self.column,
                        "Num_Documents",
                        "Global_Citations",
                        "First_Year",
                        "ID",
                    ]
                ],
                self.column,
            )
            result = x.groupby(self.column, as_index=False).agg(
                {
                    "Num_Documents": np.sum,
                    "Global_Citations": np.sum,
                    "First_Year": np.min,
                }
            )

        result["Last_Year"] = last_year
        result = result.assign(Years=result.Last_Year - result.First_Year + 1)
        result = result.assign(
            Global_Citations_per_Year=result.Global_Citations / result.Years
        )
        result["Global_Citations_per_Year"] = result["Global_Citations_per_Year"].map(
            lambda w: round(w, 2)
        )
        result = result.assign(
            Avg_Global_Citations=result.Global_Citations / result.Num_Documents
        )
        result["Avg_Global_Citations"] = result["Avg_Global_Citations"].map(
            lambda w: round(w, 2)
        )

        result["Global_Citations"] = result["Global_Citations"].map(lambda x: int(x))

        #
        # Indice H
        #
        z = x[[self.column, "Global_Citations", "ID"]].copy()
        z = (
            x.assign(
                rn=x.sort_values("Global_Citations", ascending=False)
                .groupby(self.column)
                .cumcount()
                + 1
            )
        ).sort_values(
            [self.column, "Global_Citations", "rn"], ascending=[False, False, True]
        )
        z["rn2"] = z.rn.map(lambda w: w * w)

        q = z.query("Global_Citations >= rn")
        q = q.groupby(self.column, as_index=False).agg({"rn": np.max})
        h_dict = {key: value for key, value in zip(q[self.column], q.rn)}

        result["H_index"] = result[self.column].map(
            lambda w: h_dict[w] if w in h_dict.keys() else 0
        )

        #
        # indice M
        #
        result = result.assign(M_index=result.H_index / result.Years)
        result["M_index"] = result["M_index"].map(lambda w: round(w, 2))

        #
        # indice G
        #
        q = z.query("Global_Citations >= rn2")
        q = q.groupby(self.column, as_index=False).agg({"rn": np.max})
        h_dict = {key: value for key, value in zip(q[self.column], q.rn)}
        result["G_index"] = result[self.column].map(
            lambda w: h_dict[w] if w in h_dict.keys() else 0
        )

        ## counters in axis names
        result.index = result[self.column]

        ## limit to / exclude options
        result = limit_to_exclude(
            data=result,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        result = add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )

        ## Top by / Top N
        top_by = self.top_by.replace(" ", "_").replace("-", "_").replace("/", "_")
        if top_by in ["Num_Documents", "Global_Citations"]:
            result = sort_axis(
                data=result,
                num_documents=(top_by == "Num_Documents"),
                axis=0,
                ascending=False,
            )
        else:
            result = result.sort_values(top_by, ascending=False)
        result = result.head(self.max_items)

        ## Sort by
        sort_by = self.sort_by.replace(" ", "_").replace("-", "_").replace("/", "_")
        if sort_by in ["Alphabetic", "Num_Documents", "Global_Citations"]:
            result = sort_by_axis(
                data=result, sort_by=self.sort_by, ascending=self.ascending, axis=0
            )
        else:
            result = result.sort_values(sort_by, ascending=self.ascending)

        if self.view == "Table":
            result.pop(self.column)
            result.pop("Num_Documents")
            result.pop("Global_Citations")
            result.pop("First_Year")
            result.pop("Last_Year")
            result.pop("Years")
            return result

        if self.view == "Bar plot":
            top_by = self.top_by.replace(" ", "_")
            return bar_plot(
                height=result[top_by],
                cmap=self.colormap,
                ylabel=self.top_by,
                figsize=(self.width, self.height),
            )

        if self.view == "Horizontal bar plot":
            top_by = self.top_by.replace(" ", "_")
            return barh_plot(
                width=result[top_by],
                cmap=self.colormap,
                xlabel=self.top_by,
                figsize=(self.width, self.height),
            )

    def compute_general_table(self):

        x = self.data.copy()

        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    self.column,
                    "Num_Documents",
                    "Global_Citations",
                    "Local_Citations",
                    "ID",
                ]
            ],
            self.column,
        )
        result = x.groupby(self.column, as_index=True).agg(
            {
                "Num_Documents": np.sum,
                "Global_Citations": np.sum,
                "Local_Citations": np.sum,
            }
        )
        result["Global_Citations"] = result["Global_Citations"].map(lambda w: int(w))
        result["Local_Citations"] = result["Local_Citations"].map(lambda w: int(w))

        ##
        ## Limit To / Exclude items
        ##
        result = limit_to_exclude(
            data=result,
            axis=0,
            column=self.column,
            limit_to=self.limit_to,
            exclude=self.exclude,
        )

        ##
        ## Minimal occurrence
        ##
        result = result[result.Num_Documents >= self.min_occ]

        ##
        ## Counters
        ##
        result = add_counters_to_axis(
            X=result, axis=0, data=self.data, column=self.column
        )

        ##
        ## Top by
        ##
        columns = {
            "Num Documents": ["Num_Documents", "Global_Citations", "Local_Citations"],
            "Global Citations": [
                "Global_Citations",
                "Num_Documents",
                "Local_Citations",
            ],
            "Local Citations": ["Local_Citations", "Global_Citations", "Num_Documents"],
        }[self.top_by]
        result = result.sort_values(by=columns, ascending=False)

        # top_by = self.top_by.replace(" ", "_").replace("-", "_").replace("/", "_")
        # result = sort_axis(
        #     data=result,
        #     num_documents=(top_by == "Num_Documents"),
        #     axis=0,
        #     ascending=False,
        # )

        result = result.head(self.max_items)

        ##
        ## Sort by
        ##
        # result = sort_axis(
        #      data=result,
        #      num_documents=(self.sort_by == "Num Documents"),
        #      axis=0,
        #      ascending=self.ascending,
        #  )
        result = result.sort_values(by=columns, ascending=self.ascending)

        return result

    def general(self):

        result = self.compute_general_table()

        if self.view == "Table":
            return result

        if self.top_by == "Num Documents":
            values = result.Num_Documents
            darkness = result.Global_Citations
        elif self.top_by == "Global Citations":
            values = result.Global_Citations
            darkness = result.Num_Documents

        elif self.top_by == "Local Citations":
            values = result.Local_Citations
            darkness = result.Num_Documents
        else:
            values = None
            darkness = None

        if self.view == "Bar plot":
            return bar_plot(
                height=values,
                darkness=darkness,
                cmap=self.colormap,
                ylabel=self.top_by,
                figsize=(self.width, self.height),
            )

        if self.view == "Horizontal bar plot":
            return barh_plot(
                width=values,
                darkness=darkness,
                cmap=self.colormap,
                xlabel=self.top_by,
                figsize=(self.width, self.height),
            )
        if self.view == "Pie plot":
            return pie_plot(
                x=values,
                darkness=darkness,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )

        if self.view == "Wordcloud":
            ## remueve num_documents:global_citations from terms
            values.index = [" ".join(term.split(" ")[:-1]) for term in values.index]
            darkness.index = [" ".join(term.split(" ")[:-1]) for term in darkness.index]
            return wordcloud_(
                x=values,
                darkness=darkness,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )

        if self.view == "Treemap":
            return treemap(
                x=values,
                darkness=darkness,
                cmap=self.colormap,
                figsize=(self.width, self.height),
            )

    def list_of_core_source_titles(self):

        x = self.data.copy()
        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    "Source_title",
                    "Num_Documents",
                    "ID",
                ]
            ],
            "Source_title",
        )
        m = x.groupby("Source_title", as_index=True).agg(
            {
                "Num_Documents": np.sum,
            }
        )
        m = m[["Num_Documents"]]
        m = m.sort_values(by="Num_Documents", ascending=False)
        m["Cum_Num_Documents"] = m.Num_Documents.cumsum()
        m = m[m.Cum_Num_Documents <= int(len(self.data) / 3)]
        HTML = "1 st. Bradford' Group<br>"
        for value in m.Num_Documents.unique():
            n = m[m.Num_Documents == value]
            HTML += "======================================================<br>"
            HTML += "Num Documents Published:" + str(value) + "<br>"
            HTML += "<br>"
            for source in n.index:
                HTML += "    " + source + "<br>"
            HTML += "<br><br>"
        return widgets.HTML("<pre>" + HTML + "</pre>")

    def limit_to_python_code(self):

        result = self.compute_general_table()
        items = result.index.tolist()
        items = [" ".join(item.split(" ")[:-1]) for item in items]
        HTML = "LIMIT_TO = {<br>"
        HTML += '    "' + self.column + '": [<br>'
        for item in sorted(items):
            HTML += '        "' + item + '",<br>'
        HTML += "    ]<br>"
        HTML += "}<br>"
        return widgets.HTML("<pre>" + HTML + "</pre>")

    def bradford_law(self):

        x = self.data.copy()

        x["Num_Documents"] = 1
        x = explode(
            x[
                [
                    "Source_title",
                    "Num_Documents",
                    "Global_Citations",
                    "ID",
                ]
            ],
            self.column,
        )
        result = x.groupby("Source_title", as_index=False).agg(
            {
                "Num_Documents": np.sum,
                "Global_Citations": np.sum,
            }
        )
        result["Global_Citations"] = result["Global_Citations"].map(lambda w: int(w))
        result = result.sort_values(
            by=["Num_Documents", "Global_Citations"], ascending=False
        )
        result = result.reset_index()
        result["Cum_Num_Documents"] = result.Num_Documents.cumsum()

        matplotlib.rc("font", size=11)
        fig = plt.Figure(figsize=(self.width, self.height))
        ax = fig.subplots()
        cmap = plt.cm.get_cmap(self.colormap)
        color = cmap(0.6)

        ax.plot(
            list(range(1, len(result) + 1)),
            result.Num_Documents.tolist(),
            linestyle="-",
            linewidth=3,
            color="k",
        )
        ax.fill_between(
            list(range(1, len(result) + 1)),
            result.Num_Documents.tolist(),
            color=color,
            alpha=0.6,
        )

        ##
        ## Compute core sources
        ##
        core_documents = int(len(self.data) / 3)
        core_sources = result[
            result.Cum_Num_Documents <= core_documents
        ].Source_title.head(self.max_items)
        core_sources = core_sources.map(
            lambda w: w[0:27] + " [...]" if len(w) > 30 else w
        )

        ax.fill_between(
            [1, len(core_sources) + 1],
            [ax.get_ylim()[1], ax.get_ylim()[1]],
            color=color,
            alpha=0.2,
        )

        for x in ["top", "right", "left", "bottom"]:
            ax.spines[x].set_visible(False)

        ax.set_xscale("log")

        ax.grid(axis="y", color="gray", linestyle=":")
        ax.grid(axis="x", color="gray", linestyle=":")
        ax.set_ylabel("Num Documnets")

        ax.tick_params(axis="x", labelrotation=90)

        ax.set_xticks(list(range(1, len(core_sources) + 1)))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticklabels(core_sources.tolist())

        fig.set_tight_layout(True)

        return fig

    def lotka_law(self):

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
            color="K",
        )

        for x in ["top", "right", "left", "bottom"]:
            ax.spines[x].set_visible(False)

        ax.grid(axis="y", color="gray", linestyle=":")
        ax.grid(axis="x", color="gray", linestyle=":")
        ax.set_ylabel("% of Authors")
        ax.set_xlabel("Docuemts written per Author")

        fig.set_tight_layout(True)

        return fig


###############################################################################
##
##  DASHBOARD
##
###############################################################################


class DASHapp(DASH, Model):
    def __init__(
        self,
        limit_to=None,
        exclude=None,
        years_range=None,
    ):
        """Dashboard app"""

        data = pd.read_csv("corpus.csv")

        Model.__init__(
            self,
            data=data,
            limit_to=limit_to,
            exclude=exclude,
            years_range=years_range,
        )
        DASH.__init__(self)

        COLUMNS = sorted([column for column in data.columns])

        self.command_panel = [
            dash.dropdown(
                description="MENU:",
                options=[
                    "General",
                    "Worldmap",
                    "Impact",
                    "Single/Multiple publication",
                    "Core authors",
                    "Core source titles",
                    "Most global cited documents",
                    "Most local cited documents",
                    "List of core source titles",
                    "Bradford law",
                    "Lotka law",
                    "LIMIT TO python code",
                ],
            ),
            dash.dropdown(
                description="Column:",
                options=[z for z in COLUMNS if z in data.columns],
            ),
            dash.min_occurrence(),
            dash.max_items(),
            dash.separator(text="Visualization"),
            dash.dropdown(
                description="View:",
                options=[
                    "Table",
                    "Bar plot",
                    "Horizontal bar plot",
                    "Pie plot",
                    "Wordcloud",
                    "Worldmap",
                    "Treemap",
                    "S/D Ratio (bar)",
                    "S/D Ratio (barh)",
                ],
            ),
            dash.dropdown(
                description="Top by:",
                options=[
                    "Num Documents",
                    "Global Citations",
                    "Frac Num Documents",
                    "Global Citations per Year",
                    "Avg Global Citations",
                    "H index",
                    "M index",
                    "G index",
                ],
            ),
            dash.dropdown(
                description="Sort by:",
                options=[
                    "Num Documents",
                    "Frac Num Documents",
                    "Global Citations",
                    "Global Citations per Year",
                    "Avg Global Citations",
                    "H index",
                    "M index",
                    "G index",
                    "*Index*",
                ],
            ),
            dash.ascending(),
            dash.cmap(),
            dash.fig_width(),
            dash.fig_height(),
        ]

    def set_menu_options(self):

        config = {
            "General": {
                "Column:": sorted([column for column in self.data.columns]),
                "View:": [
                    "Table",
                    "Bar plot",
                    "Horizontal bar plot",
                    "Pie plot",
                    "Wordcloud",
                    "Treemap",
                ],
                "Top by:": ["Num Documents", "Global Citations", "Local Citations"],
                "Sort by:": ["Num Documents", "Global Citations", "Local Citations"],
            },
            "Impact": {
                "Column:": [
                    "Authors",
                    "Institutions",
                    "Institution_1st_Author",
                    "Countries",
                    "Country_1st_Author",
                    "Source_title",
                ],
                "View:": [
                    "Table",
                    "Bar plot",
                    "Horizontal bar plot",
                ],
                "Top by:": [
                    "Num Documents",
                    "Global Citations",
                    "Global Citations per Year",
                    "Avg Global Citations",
                    "H index",
                    "M index",
                    "G index",
                ],
                "Sort by:": [
                    "Alphabetic",
                    "Num Documents",
                    "Global Citations",
                    "Global Citations per Year",
                    "Avg Global Citations",
                    "H index",
                    "M index",
                    "G index",
                ],
            },
            "Single/Multiple publication": {
                "Column:": [
                    "Authors",
                    "Institutions",
                    "Institution_1st_Author",
                    "Countries",
                    "Country_1st_Author",
                ],
                "View:": [
                    "Table",
                    "Bar plot",
                    "Horizontal bar plot",
                ],
                "Top by:": ["Num Documents"],
                "Sort by:": [
                    "Alphabetic",
                    "Num Documents",
                    "Global Citations",
                    "SD",
                    "MD",
                    "SMR",
                ],
            },
            "Worldmap": {
                "Column:": [
                    "Countries",
                    "Country_1st_Author",
                ],
                "Top by:": ["Num Documents", "Global Citations"],
                "View": ["Worldmap"],
            },
            "Core authors": {},
            "Core source titles": {},
            "Most global cited documents": {},
            "Most local cited documents": {},
            "List of core source titles": {},
            "Bradford law": {},
            "Lotka law": {},
            "LIMIT TO python code": {
                "Column:": sorted([column for column in self.data.columns])
            },
        }

        options = config[self.menu]
        for key in options.keys():
            self.set_options(key, options[key])
            #  if key == "View:":
            #      self.view = options[key][0]

    def set_menu_interactive(self):

        menu_names = {
            0: "Column:",
            1: "Min occurrence:",
            2: "Max items:",
            3: "View:",
            4: "Top by:",
            5: "Sort by:",
            6: "Ascending:",
            7: "Colormap:",
            8: "Width:",
            9: "Height:",
        }

        config = {
            "General": {
                "Table": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                ],
                "Bar plot": [True] * 10,
                "Horizontal bar plot": [True] * 10,
                "Pie plot": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
                "Wordcloud": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
                "Treemap": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
            },
            "Impact": {
                "Table": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                ],
                "Bar plot": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                "Horizontal bar plot": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            },
            "Single/Multiple publication": {
                "Table": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                ],
                "Bar plot": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                "Horizontal bar plot": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            },
            "Worldmap": {
                "Worldmap": [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            },
            "Core authors": None,
            "Core source titles": None,
            "Most global cited documents": None,
            "Most local cited documents": None,
            "List of core source titles": None,
            "LIMIT TO python code": [
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "Bradford law": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
            ],
            "Lotka law": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
            ],
        }

        options = config[self.menu]

        if options is not None:

            if isinstance(options, dict):
                options = options[self.menu]

            for i_option, option_value in enumerate(options):

                if option_value is True:
                    self.set_enabled(menu_names[i_option])
                else:
                    self.set_disabled(menu_names[i_option])

        else:

            for name in menu_names.values():
                self.set_disabled(name)

    def interactive_output(self, **kwargs):

        DASH.interactive_output(self, **kwargs)
        self.set_menu_options()
        self.set_menu_interactive()


###############################################################################
##
##  EXTERNAL INTERFACE
##
###############################################################################


def by_term_analysis(
    limit_to=None,
    exclude=None,
    years_range=None,
):
    return DASHapp(
        limit_to=limit_to,
        exclude=exclude,
        years_range=years_range,
    ).run()
