TechMiner: A Tool for Analyzing Bibliographic Data using Python
============================================================================================

**TechMiner** is a package for mining relevant information about topics related to Research and 
Development (R&D) literature extracted from bibliographical databases as Scopus. **techMiner**
facilitates studies of systematic mapping of literature and Tech mining studies. The package can 
be used by users with basic knowledge of Python programming. However, users with advanced 
knowledge in programming and text mining can easily incorporate their codes to maximize the power 
of the library and develop advanced analysis. The package can be used to:

* Realize analyzes based on document-by-term pattern, for example, number of documents by author, by source or by keyword. 

* Calculate and plot the number of documents or citations by year.

* Realize analyzes based on term-by-term pattern, for example, number of documents by keywords and by author, by keyword and by year and so on.

* Compute and plot co-ocurrence, correlation and autocorrelation matrices.

* Realize Principal Component Analysis to detect and analyze clusters of data.

* Plot heatmaps, networks and many other types of plots for analyzing data.

**TechMiner** is an open source (distributed under the MIT license) and friendly-user
package developed and tested in Python version 3.6. 


Getting Started
---------------------------------------------------------

The current stable version can be installed from the command line using:

``$ pip3 install techminer``

**TechMiner** can be used interactively at the Python’s command prompt, but a better
experience is achieved when IPython or Jupyter’s notebook are used, allows the
user to fully document the analysis and draw conclusions. Due to the design of
the package, it is easy to use techMiner with the tools available in the ecosystem
of open source tools.


The current development version can be installed by clonning the GitHub repo 
`<https://github.com/jdvelasq/techMiner>`_ and executing 

``$ python3 setup.py install develop``

at the command prompt.

The `Tutorial <https://jdvelasq.github.io/techMiner/tutorial/index.html>`_ section contains
a basic tutorial for analyzing bibliographical data extracted from Scopus by the user.


Release Information
---------------------------------------------------

* **Authors**:

    | Prof. Juan David Velásquez-Henao, MSc, PhD
    | Universidad Nacional de Colombia, Sede Medellín.
    | jdvelasq@unal.edu.co

    | Ivanohe Johana Garcés-Correa, MSc.
    | Universidad Nacional de Colombia, Sede Medellín.
    | ijgarcesc@unal.edu.co

* **Date**: 

    September 01, 2019  **Version**: 0.0.0

* **Binary Installers:** 
   
    `<https://pypi.org/project/techminer>`_

* **Source Repository**: 

    `<https://github.com/jdvelasq/techminer>`_

* **Documentation**: 

    `<https://jdvelasq.github.io/techminer/>`_


Table of Contents
------------------------------------------------------------------------------------------


.. toctree::
   :maxdepth: 2

   modules

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

MIT license
-------------------------------------------------------------------------------

Copyright (c) 2020 Juan David Velásquez-Henao, Ivanohe Johana Garces-Correa.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
