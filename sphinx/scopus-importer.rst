Scopus Importer
===============================================================================

.. image:: ./images/scopus-importer.png
    :width: 800px
    :align: center


This app imports a scopus CSV file for analysis. The following settings are 
recomended

.. image:: ./images/scopus-settings.png
    :width: 800px
    :align: center

This app creates several files. The file `corpus.csv` has the imported dataset, but
all created files must be saved in order to reproduce the analysis. 

The actions executed by the app include:

* Rename and select columns of the original dataset.

* Remove accents. 

* Format author names.

* Disambiguate author names.

* Extract countries and institutions for authors.

* Remove copyright mark from the abstracts.

* Translate british spelling to american spelling.

* Create new columns with aditional information.

The file `corpus.csv` can be read by the user for aditional analyses.


This app can be executed using:

.. code:: python
    
    import techminer as tech

    tech.gui.scopus_importer.App().run()


