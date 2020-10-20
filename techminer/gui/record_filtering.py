import re
from techminer.core.logging_info import logging_info
import warnings
from os.path import dirname, join, exists, isfile
from os import makedirs

import numpy as np
import pandas as pd

from IPython.display import display
from ipywidgets import GridspecLayout, Layout
import ipywidgets as widgets

from techminer.core import explode
from techminer.core.extract_country_name import extract_country_name
from techminer.core.map import map_
from techminer.core.text import remove_accents

from techminer.core.thesaurus import load_file_as_dict
from techminer.core import DASH
import techminer.core.dashboard as dash
from techminer.core.create_institutions_thesaurus import create_institutions_thesaurus
from techminer.core.apply_institutions_thesaurus import apply_institutions_thesaurus
from techminer.core.create_keywords_thesaurus import create_keywords_thesaurus
from techminer.core.apply_keywords_thesaurus import apply_keywords_thesaurus

import json
import glob

warnings.filterwarnings("ignore")

from nltk import word_tokenize


class DASHapp(DASH):
    def __init__(self):

        with open("filters.json", "r") as f:
            self.filters = json.load(f)
        clusters = [
            key
            for key in self.filters
            if key
            not in [
                "bradford_law_zones",
                "citations_range",
                "citations",
                "document_types",
                "excluded_terms",
                "selected_cluster",
                "selected_types",
                "year_range",
                "years",
            ]
        ]

        self.command_panel = [
            dash.HTML("<b>Document types:</b>", hr=False),
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            dash.Checkbox(
                                value=True
                                if doctype in self.filters["document_types"]
                                else False,
                                description=doctype,
                                disabled=True
                                if doctype not in self.filters["document_types"]
                                else False,
                                indent=False,
                            )
                            for doctype in [
                                "Article",
                                "Article in Press",
                                "Book",
                                "Book Chapter",
                                "Business Article",
                                "Conference Paper",
                                "Conference Review",
                                "Data Paper",
                                "Editorial",
                            ]
                        ],
                        # layout=Layout(width="auto"),
                    ),
                    widgets.VBox(
                        [
                            dash.Checkbox(
                                value=True
                                if doctype in self.filters["document_types"]
                                else False,
                                description=doctype,
                                disabled=True
                                if doctype not in self.filters["document_types"]
                                else False,
                                indent=False,
                            )
                            for doctype in [
                                "Letter",
                                "Note",
                                "Review",
                                "Short Survey",
                                "Erratum",
                                "Report",
                                "Retracted",
                                "Abstract Report",
                                "Undefinided",
                            ]
                        ]
                    ),
                ]
            ),
            dash.HTML("Year range:"),
            dash.IntRangeSlider(
                value=self.filters["years"],
                min=self.filters["year_range"][0],
                max=self.filters["year_range"][1],
                step=1,
            ),
            dash.HTML("Bradford law zones:"),
            dash.Dropdown(
                options=[
                    ("Core sources", 0),
                    ("Core + Zone 2 sources", 1),
                    ("All sources", 2),
                ],
                value=self.filters["bradford_law_zones"],
            ),
            dash.HTML("Global Citations:"),
            dash.IntRangeSlider(
                value=self.filters["citations"],
                min=self.filters["citations_range"][0],
                max=self.filters["citations_range"][1],
                step=1,
            ),
            dash.HTML("Filters:"),
            dash.Dropdown(
                description="Exclude:",
                options=["---"] + glob.glob("KW_*.txt"),
                value=self.filters["excluded_terms"],
            ),
            dash.Dropdown(
                description="Clusters:",
                options=["---"] + clusters,
                value=self.filters["excluded_terms"],
            ),
        ]

        #
        # interactive output function
        #
        widgets.interactive_output(
            f=self.interactive_output,
            controls={
                "article": self.command_panel[1].children[0].children[0],
                "article_in_press": self.command_panel[1].children[0].children[1],
                "book": self.command_panel[1].children[0].children[2],
                "book_chapter": self.command_panel[1].children[0].children[3],
                "business_article": self.command_panel[1].children[0].children[4],
                "conference_paper": self.command_panel[1].children[0].children[5],
                "conference_review": self.command_panel[1].children[0].children[6],
                "data_paper": self.command_panel[1].children[0].children[7],
                "editorial": self.command_panel[1].children[0].children[8],
                "letter": self.command_panel[1].children[1].children[0],
                "note": self.command_panel[1].children[1].children[1],
                "review": self.command_panel[1].children[1].children[2],
                "short_survey": self.command_panel[1].children[1].children[3],
                "erratum": self.command_panel[1].children[1].children[4],
                "report": self.command_panel[1].children[1].children[5],
                "retracted": self.command_panel[1].children[1].children[6],
                "abstract_report": self.command_panel[1].children[1].children[7],
                "undefinided": self.command_panel[1].children[1].children[8],
                "years": self.command_panel[3],
                "bradford_law_zones": self.command_panel[5],
                "citations": self.command_panel[7],
                "excluded_terms": self.command_panel[9],
                "selected_cluster": self.command_panel[10],
            },
        )

        DASH.__init__(self)

    def on_click(self, args):

        self.filters["selected_types"] = []
        if self.article is True:
            self.filters["selected_types"] += ["Article"]
        if self.article_in_press is True:
            self.filters["selected_types"] += ["Article in Press"]
        if self.book is True:
            self.filters["selected_types"] += ["Book"]
        if self.book_chapter is True:
            self.filters["selected_types"] += ["Book Chapter"]
        if self.business_article is True:
            self.filters["selected_types"] += ["Business Article"]
        if self.conference_paper is True:
            self.filters["selected_types"] += ["Conference Paper"]
        if self.conference_review is True:
            self.filters["selected_types"] += ["Conference Review"]
        if self.data_paper is True:
            self.filters["selected_types"] += ["Data Paper"]
        if self.editorial is True:
            self.filters["selected_types"] += ["Editorial"]
        if self.letter is True:
            self.filters["selected_types"] += ["Letter"]
        if self.note is True:
            self.filters["selected_types"] += ["Note"]
        if self.review is True:
            self.filters["selected_types"] += ["Review"]
        if self.short_survey is True:
            self.filters["selected_types"] += ["Short Survey"]
        if self.erratum is True:
            self.filters["selected_types"] += ["Erratum"]
        if self.report is True:
            self.filters["selected_types"] += ["Report"]
        if self.retracted is True:
            self.filters["selected_types"] += ["Retracted"]
        if self.abstract_report is True:
            self.filters["selected_types"] += ["Abstract Report"]
        if self.undefinided is True:
            self.filters["selected_types"] += ["Undefinided"]

        self.filters["years"] = self.years
        self.filters["citations"] = self.citations
        self.filters["bradford_law_zones"] = self.bradford_law_zones
        self.filters["excluded_terms"] = self.excluded_terms
        self.filters["selected_cluster"] = self.selected_cluster

        with open("filters.json", "w") as f:
            print(json.dumps(self.filters, indent=4, sort_keys=True), file=f)

        self.output.clear_output()
        with self.output:
            print(json.dumps(self.filters, indent=4, sort_keys=True))