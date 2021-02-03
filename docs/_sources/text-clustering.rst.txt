Text Clustering
===============================================================================

.. image:: ./images/text-clustering.png
    :width: 800px
    :align: center

Creates a thesaurus for the terms in the selected column. The thesaurus is saved
to disk in text format. Any text editor can be used to modify the file. The
*Apply thesaurus* app can be used to apply the generated thesaurus to a column
of the biblographic dataset.

This app can be executed using:

.. code:: python
    
    import techminer as tech

    tech.gui.text_clustering.App().run()


