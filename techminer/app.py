import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display
import pandas as pd

# from techminer.column_explorer import column_explorer
from techminer.bigraph_analysis import DASHapp as BigraphAnalyzer
from techminer.by_term_analysis import DASHapp as TermAnalyzer
from techminer.by_term_per_year_analysis import MatrixDASHapp as TermYearAnalyzer
from techminer.by_year_analysis import DASHapp as YearAnalyzer
from techminer.co_word_analysis import DASHapp as CoWordAnalysis
from techminer.conceptual_structure import DASHapp as ConceptualStructure
from techminer.correlation_analysis import DASHapp as CorrelationAnalysis
from techminer.comparative_analysis import DASHapp as ComparativeAnalysis
from techminer.document_term_analysis import DASHapp as DocumentTermAnalysis
from techminer.factor_analysis import DASHapp as FactorAnalysis
from techminer.graph_analysis import DASHapp as GraphAnalyzer
from techminer.growth_indicators import DASHapp as GrowthIndicators
from techminer.keywords_association import DASHapp as KeywordsAssociation
from techminer.keywords_comparison import DASHapp as KeywordsComparison
from techminer.latent_semantic_analysis import DASHapp as LatentSemanticAnalysis
from techminer.scopus_importer import ScopusImporter
from techminer.thematic_analysis import DASHapp as ThematicAnalysis
from techminer.column_explorer import DASHapp as ColumnExplorer
from techminer.matrix_explorer import DASHapp as MatrixExplorer
from techminer.top_documents import DASHapp as TopDocuments
from techminer.worldmap import DASHapp as WorldMap

from techminer.coverage import DASHapp as CoverageReporter
from techminer.descriptive_stats import DASHapp as StatisticsReporter
from techminer.core_authors import DASHapp as CoreAuthors

#  from techminer.svd import DASHapp as SingleValueDecomposition


header_style = """
<style>
.hbox_style{
    width:99.9%;
    border : 2px solid #ff8000;
    height: auto;
    background-color:#ff8000;
    box-shadow: 1px 5px  4px #BDC3C7;
}

.app{
    background-color:#F4F6F7;
}

.output_color{
    background-color:#FFFFFF;
}

.select > select {background-color: #ff8000; color: white; border-color: light-gray;}

</style>
"""

APPS = {
    "Scopus importer": ScopusImporter,
    "*** Document term analysis": DocumentTermAnalysis,
    "*** Growth indicators": GrowthIndicators,
    "*** Term analysis": TermAnalyzer,
    "*** Term per year analysis": TermYearAnalyzer,
    "*** Thematic analysis": ThematicAnalysis,
    "Bigraph analysis": BigraphAnalyzer,
    "Co-word analysis": CoWordAnalysis,
    "Column explorer": ColumnExplorer,
    "Comparative analysis": ComparativeAnalysis,
    "Conceptual structure": ConceptualStructure,
    "Core Authors": CoreAuthors,
    "Correlation analysis": CorrelationAnalysis,
    "Coverage": CoverageReporter,
    "Descriptive Statistics": StatisticsReporter,
    "Factor analysis": FactorAnalysis,
    "Graph analyzer": GraphAnalyzer,
    "Keywords association": KeywordsAssociation,
    "Keywords comparison": KeywordsComparison,
    "Latent semantic analysis": LatentSemanticAnalysis,
    "Matrix explorer": MatrixExplorer,
    "Time analysis": YearAnalyzer,
    "Top documents": TopDocuments,
    "Worldmap": WorldMap,
}


class App:
    def __init__(self):

        #
        # APPs menu
        #
        apps_dropdown = widgets.Dropdown(
            options=[key for key in APPS.keys()],
            layout=Layout(width="70%"),
        ).add_class("select")

        #
        # Grid layout definition
        #
        self.app_layout = GridspecLayout(
            11,
            4,
            height="902px",  # layout=Layout(border="1px solid #ff8000")
        ).add_class("app")

        #
        # Populates the grid
        #
        self.app_layout[0, :] = widgets.HBox(
            [
                widgets.HTML(header_style),
                widgets.HTML('<h2 style="color:white;">TechMiner</h2>'),
                apps_dropdown,
            ],
            layout=Layout(
                display="flex",
                justify_content="space-between",
                align_items="center",
            ),
        ).add_class("hbox_style")

        #
        # Interative output
        #
        widgets.interactive_output(
            self.interactive_output,
            {"selected-app": apps_dropdown},
        )

    def run(self):
        return self.app_layout

    def interactive_output(self, **kwargs):
        self.app_layout[1:, :] = APPS[kwargs["selected-app"]]().run()
