TF*IDF Analysis
===============================================================================

.. image:: ./images/tfidf-analysis.png
    :width: 800px
    :align: center

Computes the TF-IDF values for the selected column. The app uses sklearn for 
computing the values. Options description and computations are described in 
https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting


*Min OCC* is the minimum occurrency value for selected terms. *Max items* is the 
maximum number of items reported in the app.

This app can be executed using:

.. code:: python
    
    import techminer as tech

    tech.gui.document_term_analysis.App().run()


