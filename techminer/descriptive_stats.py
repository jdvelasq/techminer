import numpy as np
import pandas as pd

#  from techminer.core.params import EXCLUDE_COLS, MULTIVALUED_COLS

##
##
## Term extraction and count
##
##


def _extract_terms(x, column):
    x = x.copy()
    x[column] = x[column].map(
        lambda w: w.split(";") if not pd.isna(w) and isinstance(w, str) else w
    )
    x = x.explode(column)
    x[column] = x[column].map(lambda w: w.strip() if isinstance(w, str) else w)
    x = pd.unique(x[column].dropna())
    x = np.sort(x)
    return pd.DataFrame({column: x})


def _count_terms(x, column):
    return len(_extract_terms(x, column))


def descriptive_stats(input_file="techminer.csv"):

    x = pd.read_csv(input_file)

    ##
    ##  General information
    ##
    general = {}

    general["Documents:"] = str(len(x))

    if "Year" in x.columns:
        general["Years:"] = str(min(x.Year)) + "-" + str(max(x.Year))
        n = max(x.Year) - min(x.Year) + 1
        Po = len(x.Year[x.Year == min(x.Year)])
        Pn = len(x)
        cagr = str(round(100 * (np.power(Pn / Po, 1 / n) - 1), 2)) + " %"
        general["Compound annual growth rate:"] = cagr

    if "Global_Citations" in x.columns:
        general["Average citations per document:"] = "{:4.2f}".format(
            x["Global_Citations"].mean()
        )

    if "Global_Citations" in x.columns and "Year" in x.columns:
        general["Average citations per document per year:"] = "{:4.2f}".format(
            x["Global_Citations"].sum() / (len(x) * (x.Year.max() - x.Year.min() + 1))
        )

    if "Global_References" in x.columns:
        general["Total references:"] = round(_count_terms(x, "Global_References"))
        general["Average global references per document:"] = round(
            _count_terms(x, "Global_References") / len(x)
        )

    if "Source_title" in x.columns:
        general["Source titles:"] = round(_count_terms(x, "Source_title"))
        general["Average documents per Source title:"] = round(
            len(x) / _count_terms(x, "Source_title")
        )
        x.pop("Source_title")

    if "Abb_Source_Title" in x.columns:
        general["Abbreviated Source titles:"] = round(
            _count_terms(x, "Abb_Source_Title")
        )
        x.pop("Abb_Source_Title")

    ##
    ##  Document types
    ##
    document_types = {}
    if "Document_Type" in x.columns:
        z = x[["Document_Type"]].groupby("Document_Type").size()
        for index, value in zip(z.index, z):
            document_types[index + ":"] = value
        x.pop("Document_Type")
    ##
    ##  Authors
    ##
    authors = {}

    if "Authors" in x.columns:

        authors["Authors:"] = _count_terms(x, "Authors")

        m = x.Authors
        m = m.dropna()
        m = m.map(lambda w: w.split(";"), na_action="ignore")
        m = m.explode()
        authors["Author appearances:"] = len(m)

        authors["Documents per author:"] = round(len(x) / _count_terms(x, "Authors"), 2)
        authors["Authors per document:"] = round(_count_terms(x, "Authors") / len(x), 2)

    if "Num_Authors" in x.columns:
        authors["Single-authored documents:"] = len(x[x["Num_Authors"] == 1])
        authors["Multi-authored documents:"] = len(x[x["Num_Authors"] > 1])
        authors["Co-authors per document:"] = round(x["Num_Authors"].mean(), 2)
        authors["Collaboration index:"] = round(
            _count_terms(x[x.Num_Authors > 1], "Authors") / len(x[x.Num_Authors > 1]),
            2,
        )

    if "Institutions" in x.columns:
        authors["Institutions:"] = _count_terms(x, "Institutions")
        x.pop("Institutions")

    if "Institution_1st_Author" in x.columns:
        authors["Institutions (1st author):"] = _count_terms(
            x, "Institution_1st_Author"
        )
        x.pop("Institution_1st_Author")

    if "Countries" in x.columns:
        authors["Countries:"] = _count_terms(x, "Countries")
        if "Countries" in x.columns:
            x.pop("Countries")

    if "Country_1st_Author" in x.columns:
        authors["Countries (1st author):"] = _count_terms(x, "Country_1st_Author")
        x.pop("Country_1st_Author")

    ##
    ##  Keywords
    ##
    keywords = {}

    if "Author_Keywords" in x.columns:
        keywords["Author Keywords (raw):"] = round(_count_terms(x, "Author_Keywords"))
        x.pop("Author_Keywords")

    if "Author_Keywords_CL" in x.columns:
        keywords["Author Keywords (cleaned):"] = round(
            _count_terms(x, "Author_Keywords_CL")
        )
        x.pop("Author_Keywords_CL")

    if "Index_Keywords" in x.columns:
        keywords["Index Keywords (raw):"] = round(_count_terms(x, "Index_Keywords"))
        x.pop("Index_Keywords")

    if "Index_Keywords_CL" in x.columns:
        keywords["Index Keywords (cleaned):"] = round(
            _count_terms(x, "Index_Keywords_CL")
        )
        x.pop("Index_Keywords_CL")

    if "Keywords_CL" in x.columns:
        keywords["Keywords (cleaned):"] = round(_count_terms(x, "Keywords_CL"))
        x.pop("Keywords_CL")

    if "Title_words" in x.columns:
        keywords["Title words (raw):"] = round(_count_terms(x, "Title_words"))
        x.pop("Title_words")

    if "Title_words_CL" in x.columns:
        keywords["Title words (cleaned):"] = round(_count_terms(x, "Title_words_CL"))
        x.pop("Title_words_CL")

    if "Abstract_words" in x.columns:
        keywords["Abstract words (raw)"] = round(_count_terms(x, "Abstract_words"))
        x.pop("Abstract_words")

    if "Abstract_words_CL" in x.columns:
        keywords["Abstract words (cleaned)"] = round(
            _count_terms(x, "Abstract_words_CL")
        )
        x.pop("Abstract_words_CL")

    ##
    ##  Report
    ##

    if "Frac_Num_Documents" in x.columns:
        x.pop("Frac_Num_Documents")

    if "Historiograph_ID" in x.columns:
        x.pop("Historiograph_ID")

    d = []
    d += [key for key in general.keys()]
    d += [key for key in document_types.keys()]
    d += [key for key in authors.keys()]
    d += [key for key in keywords.keys()]

    v = []
    v += [general[key] for key in general.keys()]
    v += [document_types[key] for key in document_types.keys()]
    v += [authors[key] for key in authors.keys()]
    v += [keywords[key] for key in keywords.keys()]

    ##
    ##  Other columns in the dataset
    ##

    others = {}
    for column in sorted(x.columns):

        if column + ":" in d or column in [
            "Abstract",
            "Abstract_Author_Keywords",
            "Abstract_Author_Keywords_CL",
            "Abstract_Index_Keywords",
            "Abstract_Index_Keywords_CL",
            "Abstract_Keywords",
            "Abstract_Keywords_CL",
            "Authors_ID",
            "Bradford_Law_Zone",
            "Global_Citations",
            "Global_References",
            "ID",
            "Keywords",
            "Local_Citations",
            "Local_References",
            "Num_Authors",
            "Title",
            "Year",
            "Affiliations",
        ]:
            continue

        others[column] = round(_count_terms(x, column))

    if len(others):
        d += [key for key in others.keys()]
        v += [others[key] for key in others.keys()]

    return pd.DataFrame(
        v,
        columns=["value"],
        index=pd.MultiIndex.from_arrays(
            [
                ["GENERAL"] * len(general)
                + ["DOCUMENT TYPES"] * len(document_types)
                + ["AUTHORS"] * len(authors)
                + ["KEYWORDS"] * len(keywords)
                + ["OTHERS"] * len(others),
                d,
            ],
            names=["Category", "Item"],
        ),
    )
