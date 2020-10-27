import ipywidgets as widgets
from ipywidgets import GridspecLayout, Layout
from IPython.display import display
import pandas as pd


from techminer.gui.apply_thesaurus import App as ApplyThesaurus
from techminer.gui.bigraph_analysis import App as BiGraphAnalysis
from techminer.gui.bradford_law import App as BradfordLaw
from techminer.gui.by_term_analysis import App as TermAnalysis
from techminer.gui.by_term_per_year_analysis import MatrixApp as TermYearAnalysis
from techminer.gui.by_year_analysis import App as YearAnalysis
from techminer.gui.co_word_analysis import App as CoWordAnalysis
from techminer.gui.collaboration_analysis import App as CollaborationAnalysis
from techminer.gui.column_explorer import App as ColumnExplorer
from techminer.gui.comparative_analysis import App as ComparativeAnalysis
from techminer.gui.conceptual_structure import App as ConceptualStructure
from techminer.gui.core_authors import App as CoreAuthors
from techminer.gui.core_sources import App as CoreSources
from techminer.gui.correlation_analysis import App as CorrelationAnalysis
from techminer.gui.coverage import App as Coverage
from techminer.gui.descriptive_stats import App as DescriptiveStats
from techminer.gui.document_term_analysis import App as DocumentTermAnalysis
from techminer.gui.extract_user_keywords import App as ExtractUserKeywords
from techminer.gui.factor_analysis import App as FactorAnalysis
from techminer.gui.graph_analysis import App as GraphAnalysis
from techminer.gui.growth_indicators import App as GrowthIndicators
from techminer.gui.impact_analysis import App as ImpactAnalysis
from techminer.gui.keywords_association import App as KeywordsAssociation
from techminer.gui.keywords_comparison import App as KeywordsComparison
from techminer.gui.latent_semantic_analysis import App as LatentSemanticAnalysis
from techminer.gui.main_path_analysis import App as MainPathAnalysis
from techminer.gui.manage_columns import App as ManageColumns
from techminer.gui.matrix_explorer import App as MatrixExplorer
from techminer.gui.record_filtering import App as RecordFiltering
from techminer.gui.scopus_importer import App as ScopusImporter
from techminer.gui.text_clustering import App as TextClustering
from techminer.gui.thematic_analysis import App as ThematicAnalysis
from techminer.gui.top_documents import App as TopDocuments
from techminer.gui.worldmap import App as Worldmap
from techminer.gui.citation_analysis import App as CitationAnalysis


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
    "Apply thesaurus": ApplyThesaurus,
    "Bigraph analysis": BiGraphAnalysis,
    "Bradford law": BradfordLaw,
    "Citation analysis": CitationAnalysis,
    "Co-word analysis": CoWordAnalysis,
    "Collaboration analysis": CollaborationAnalysis,
    "Column explorer": ColumnExplorer,
    "Comparative analysis": ComparativeAnalysis,
    "Conceptual structure": ConceptualStructure,
    "Core authors analysis": CoreAuthors,
    "Core sources analysis": CoreSources,
    "Correlation analysis": CorrelationAnalysis,
    "Coverage": Coverage,
    "Descriptive Statistics": DescriptiveStats,
    "Extract user keywords": ExtractUserKeywords,
    "Factor analysis": FactorAnalysis,
    "Graph analysis": GraphAnalysis,
    "Growth indicators": GrowthIndicators,
    "Impact analysis": ImpactAnalysis,
    "Keywords association": KeywordsAssociation,
    "Keywords comparison": KeywordsComparison,
    "Latent semantic analysis": LatentSemanticAnalysis,
    "Main path analysis": MainPathAnalysis,
    "Manage columns": ManageColumns,
    "Matrix explorer": MatrixExplorer,
    "Record filtering": RecordFiltering,
    "Scopus importer": ScopusImporter,
    "Term analysis": TermAnalysis,
    "Term per year analysis": TermYearAnalysis,
    "Text clustering": TextClustering,
    "TF*IDF analysis": DocumentTermAnalysis,
    "Thematic analysis": ThematicAnalysis,
    "Time analysis": YearAnalysis,
    "Top documents": TopDocuments,
    "Worldmap": Worldmap,
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
            13, 4, height="944px", layout=Layout(border="1px solid #E0E9EF")  #
        ).add_class(
            "app"
        )  #  902

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
