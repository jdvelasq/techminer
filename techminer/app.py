import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display
import pandas as pd


from techminer.gui.bigraph_analysis import DASHapp as BigraphAnalyzer
from techminer.gui.bradford_law import DASHapp as BradfordLaw
from techminer.gui.by_term_analysis import DASHapp as TermAnalyzer
from techminer.gui.by_term_per_year_analysis import MatrixDASHapp as TermYearAnalyzer
from techminer.gui.by_year_analysis import DASHapp as YearAnalyzer
from techminer.gui.co_word_analysis import DASHapp as CoWordAnalysis
from techminer.gui.collaboration_analysis import DASHapp as CollaborationAnalysis
from techminer.gui.column_explorer import DASHapp as ColumnExplorer
from techminer.gui.comparative_analysis import DASHapp as ComparativeAnalysis
from techminer.gui.conceptual_structure import DASHapp as ConceptualStructure
from techminer.gui.core_authors import DASHapp as CoreAuthors
from techminer.gui.core_sources import DASHapp as CoreSources
from techminer.gui.correlation_analysis import DASHapp as CorrelationAnalysis
from techminer.gui.coverage import DASHapp as CoverageReporter
from techminer.gui.descriptive_stats import DASHapp as StatisticsReporter
from techminer.gui.document_term_analysis import DASHapp as DocumentTermAnalysis
from techminer.gui.extract_keywords_from import DASHapp as ExtractKeywordsFrom
from techminer.gui.factor_analysis import DASHapp as FactorAnalysis
from techminer.gui.graph_analysis import DASHapp as GraphAnalyzer
from techminer.gui.growth_indicators import DASHapp as GrowthIndicators
from techminer.gui.impact_analysis import DASHapp as ImpactAnalysis
from techminer.gui.keywords_association import DASHapp as KeywordsAssociation
from techminer.gui.keywords_comparison import DASHapp as KeywordsComparison
from techminer.gui.latent_semantic_analysis import DASHapp as LatentSemanticAnalysis
from techminer.gui.manage_columns import DASHapp as ManageColumns
from techminer.gui.matrix_explorer import DASHapp as MatrixExplorer
from techminer.gui.scopus_importer import ScopusImporter
from techminer.gui.thematic_analysis import DASHapp as ThematicAnalysis
from techminer.gui.top_documents import DASHapp as TopDocuments
from techminer.gui.worldmap import DASHapp as WorldMap
from techminer.gui.text_clustering import DASHapp as TextClustering

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
    "Bigraph analysis": BigraphAnalyzer,
    "Bradford law": BradfordLaw,
    "Co-word analysis": CoWordAnalysis,
    "Collaboration analysis": CollaborationAnalysis,
    "Column explorer": ColumnExplorer,
    "Comparative analysis": ComparativeAnalysis,
    "Conceptual structure": ConceptualStructure,
    "Core authors analysis": CoreAuthors,
    "Core sources analysis": CoreSources,
    "Correlation analysis": CorrelationAnalysis,
    "Coverage": CoverageReporter,
    "Descriptive Statistics": StatisticsReporter,
    "Extract user keywords": ExtractKeywordsFrom,
    "Factor analysis": FactorAnalysis,
    "Graph analyzer": GraphAnalyzer,
    "Growth indicators": GrowthIndicators,
    "Impact analysis": ImpactAnalysis,
    "Keywords association": KeywordsAssociation,
    "Keywords comparison": KeywordsComparison,
    "Latent semantic analysis": LatentSemanticAnalysis,
    "Manage columns": ManageColumns,
    "Matrix explorer": MatrixExplorer,
    "Scopus importer": ScopusImporter,
    "Term analysis": TermAnalyzer,
    "Term per year analysis": TermYearAnalyzer,
    "Text clustering": TextClustering,
    "TF*IDF analysis": DocumentTermAnalysis,
    "Thematic analysis": ThematicAnalysis,
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
            value="Scopus importer",
        ).add_class("select")

        #
        # Grid layout definition
        #
        self.app_layout = GridspecLayout(
            11, 4, height="902px", layout=Layout(border="1px solid #E0E9EF")  #
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
