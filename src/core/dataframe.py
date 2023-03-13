"""
DataFrame object
==================================================================================================

Column names
--------------------------------------------------------------------------------------------------

The column names in the dataframe follows the convetion used in WoS.

* `AB`: Abstract.
* `AF`: Author full name.
* `AR`: Article Number.
* `AU`: Authors.
* `AI`: Authors Identifiers.
* `BA`: Book Authors.
* `BE`: Editors.
* `BF`: Book Authors Full Name.
* `BN`: International Standard Book Number (ISBN).
* `BP`: Begining page.
* `BS`: Book Series Subtitle.
* `C1`: Author Address.
* `CA`: Group Authors.
* `CL`: Conference Location.
* `CR`: Cited References.
* `CT`: Conference Title.
* `CY`: Conference Date.
* `D2`: Book DOI.
* `DA`:	Date this report was generated.
* `DE`: Author keywords.
* `DI`: DOI
* `DT`: Document Type.
* `EA`: Early access date.
* `EF`:	End of File.
* `EI`: Electronic International Standard Serial Number (eISSN).
* `EM`: E-mail Address.
* `EP`: Ending page.
* `ER`:	End of Record.
* `EY`: Early access year.
* `FN`: File Name.
* `FU`: Funding Agency and Grant Number.
* `FX`: Funding Text.
* `GA`:	Document Delivery Number.
* `GP`: Book Group Authors.
* `HC`:	ESI Highly Cited Paper. Note that this field is valued only for ESI subscribers.
* `HO`: Conference Host.
* `HP`:	ESI Hot Paper. Note that this field is valued only for ESI subscribers.
* `ID`: Keyword plus.
* `IS`: Issue.
* `J9`: 29-Character Source Abbreviation.
* `JI`: ISO Source Abbreviation
* `LA`: Language.
* `MA`: Meeting Abstract.
* `NR`: Cited Reference Count.
* `OA`:	Open Access Indicator.
* `OI`: ORCID Identifier (Open Researcher and Contributor ID).
* `P2`: Chapter Count (Book Citation Index).
* `PA`: Publisher Address.
* `PG`: Page count.
* `PI`: Publisher City.
* `PM`:	PubMed ID.
* `PN`: Part Number.
* `PR`: Reprint Address.
* `PT`: Publication Type (J=Journal; B=Book; S=Series; P=Patent).
* `PU`: Publisher.
* `PY`: Year Published.
* `RI`: ResearcherID Number.
* `SC`: Research Areas.
* `SE`: Book Series Title.
* `SI`: Special Issue.
* `SN`: International Standard Serial Number (ISSN).
* `SO`: Publication Name.
* `SP`: Conference Sponsors.
* `SU`: Supplement.
* `TC`: Web of Science Core Collection Global Citations Count.
* `TI`: Document Title.
* `U1`: Usage Count (Last 180 Days).
* `U2`: Usage Count (Since 2013).
* `UT`:	Accession Number.
* `VL`: Volume.
* `VR`: Version Number.
* `WC`: Web of Science Categories.
* `Z9`: Total Global Citations Count.


"""
# import json
# import math
# import re
# from os.path import dirname, join

# import numpy as np
#  import pandas as pd

# from sklearn.decomposition import PCA


# from techminer.by_term import documents_by_term, citations_by_term


# def sort_by_numdocuments(
#     df, matrix, axis=0, ascending=True, kind="quicksort", axis_name=None, axis_sep=None
# ):
#     """Sorts a matrix axis by the number of documents.


#     Args:
#         df (pandas.DataFrame): dataframe with bibliographic information.
#         matrix (pandas.DataFrame): matrix to sort.
#         axis ({0 or ‘index’, 1 or ‘columns’}), default 0: axis to be sorted.
#         ascending (bool): sort ascending?.
#         kind (str): ‘quicksort’, ‘mergesort’, ‘heapsort’.
#         axis_name (str): column name used to sort by number of documents.
#         axis_sep (str): character used to separate the internal values of column axis_name.

#     Returns:
#         DataFrame sorted.

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...         "c0": ["D"] * 4 + ["B"] * 3 + ["C"] * 2 + ["A"],
#     ...         "c1": ["a"] * 4 + ["c"] * 3 + ["b"] * 2 + ["d"],
#     ...         "Global_Citations": list(range(10)),
#     ...         "ID": list(range(10)),
#     ...     },
#     ... )
#     >>> df
#       c0 c1  Global_Citations   ID
#     0  D  a         0   0
#     1  D  a         1   1
#     2  D  a         2   2
#     3  D  a         3   3
#     4  B  c         4   4
#     5  B  c         5   5
#     6  B  c         6   6
#     7  C  b         7   7
#     8  C  b         8   8
#     9  A  d         9   9

#     >>> matrix = pd.DataFrame(
#     ...     {"D": [0, 1, 2, 3], "B": [4, 5, 6, 7], "A": [8, 9, 10, 11], "C": [12, 13, 14, 15],},
#     ...     index=list("badc"),
#     ... )
#     >>> matrix
#        D  B   A   C
#     b  0  4   8  12
#     a  1  5   9  13
#     d  2  6  10  14
#     c  3  7  11  15

#     >>> sort_by_numdocuments(df, matrix, axis='columns', ascending=True, axis_name='c0')
#         A  B   C  D
#     b   8  4  12  0
#     a   9  5  13  1
#     d  10  6  14  2
#     c  11  7  15  3

#     >>> sort_by_numdocuments(df, matrix, axis='columns', ascending=False, axis_name='c0')
#        D   C  B   A
#     b  0  12  4   8
#     a  1  13  5   9
#     d  2  14  6  10
#     c  3  15  7  11

#     >>> sort_by_numdocuments(df, matrix, axis='index', ascending=True, axis_name='c1')
#        D  B   A   C
#     a  1  5   9  13
#     b  0  4   8  12
#     c  3  7  11  15
#     d  2  6  10  14

#     >>> sort_by_numdocuments(df, matrix, axis='index', ascending=False, axis_name='c1')
#        D  B   A   C
#     d  2  6  10  14
#     c  3  7  11  15
#     b  0  4   8  12
#     a  1  5   9  13

#     """
#     terms = documents_by_term(df, column=axis_name)
#     terms_sorted = (
#         terms.sort_values(by=axis_name, kind=kind, ascending=ascending)
#         .iloc[:, 0]
#         .tolist()
#     )
#     if axis == "index":
#         return matrix.loc[terms_sorted, :]
#     return matrix.loc[:, terms_sorted]


# def sort_by_citations(
#     df, matrix, axis=0, ascending=True, kind="quicksort", axis_name=None, axis_sep=None
# ):
#     """Sorts a matrix axis by the citations.


#     Args:
#         df (pandas.DataFrame): dataframe with bibliographic information.
#         matrix (pandas.DataFrame): matrix to sort.
#         axis ({0 or ‘index’, 1 or ‘columns’}), default 0: axis to be sorted.
#         ascending (bool): sort ascending?.
#         kind (str): ‘quicksort’, ‘mergesort’, ‘heapsort’.
#         axis_name (str): column name used to sort by citations.
#         axis_sep (str): character used to separate the internal values of column axis_name.

#     Returns:
#         DataFrame sorted.

#     >>> import pandas as pd
#     >>> df = pd.DataFrame(
#     ...     {
#     ...         "c0": ["D"] * 4 + ["B"] * 3 + ["C"] * 2 + ["A"],
#     ...         "c1": ["a"] * 4 + ["c"] * 3 + ["b"] * 2 + ["d"],
#     ...         "Global_Citations": list(range(10)),
#     ...         "ID": list(range(10)),
#     ...     },
#     ... )
#     >>> df
#       c0 c1  Global_Citations   ID
#     0  D  a         0   0
#     1  D  a         1   1
#     2  D  a         2   2
#     3  D  a         3   3
#     4  B  c         4   4
#     5  B  c         5   5
#     6  B  c         6   6
#     7  C  b         7   7
#     8  C  b         8   8
#     9  A  d         9   9

#     >>> matrix = pd.DataFrame(
#     ...     {"D": [0, 1, 2, 3], "B": [4, 5, 6, 7], "A": [8, 9, 10, 11], "C": [12, 13, 14, 15],},
#     ...     index=list("badc"),
#     ... )
#     >>> matrix
#        D  B   A   C
#     b  0  4   8  12
#     a  1  5   9  13
#     d  2  6  10  14
#     c  3  7  11  15

#     >>> sort_by_citations(df, matrix, axis='columns', ascending=True, axis_name='c0')
#         A  B   C  D
#     b   8  4  12  0
#     a   9  5  13  1
#     d  10  6  14  2
#     c  11  7  15  3

#     >>> sort_by_citations(df, matrix, axis='columns', ascending=False, axis_name='c0')
#        D   C  B   A
#     b  0  12  4   8
#     a  1  13  5   9
#     d  2  14  6  10
#     c  3  15  7  11

#     >>> sort_by_citations(df, matrix, axis='index', ascending=True, axis_name='c1')
#        D  B   A   C
#     a  1  5   9  13
#     b  0  4   8  12
#     c  3  7  11  15
#     d  2  6  10  14

#     >>> sort_by_citations(df, matrix, axis='index', ascending=False, axis_name='c1')
#        D  B   A   C
#     d  2  6  10  14
#     c  3  7  11  15
#     b  0  4   8  12
#     a  1  5   9  13

#     """
#     terms = citations_by_term(df, column=axis_name)
#     terms_sorted = (
#         terms.sort_values(by=axis_name, kind=kind, ascending=ascending)
#         .iloc[:, 0]
#         .tolist()
#     )
#     if axis == "index":
#         return matrix.loc[terms_sorted, :]
#     return matrix.loc[:, terms_sorted]
