import pandas as pd

from techminer.core.map import map_
from techminer.core.thesaurus import read_textfile


def apply_keywords_thesaurus(
    input_file="corpus.csv",
    thesaurus_file="TH_keywords.txt",
    output_file="corpus.csv",
):

    df = pd.read_csv(input_file)

    ##
    ## Loads the thesaurus
    ##
    th = read_textfile(thesaurus_file)
    th = th.compile_as_dict()

    ##
    ## Author keywords cleaning
    ##
    if "Author_Keywords" in df.columns:
        df["Author_Keywords_CL"] = map_(df, "Author_Keywords", th.apply_as_dict)

    ##
    ## Index keywords cleaning
    ##
    if "Index_Keywords" in df.columns:
        df["Index_Keywords_CL"] = map_(df, "Index_Keywords", th.apply_as_dict)

    ##
    ## Keywords new field creation
    ##
    if "Author_Keywords" in df.columns and "Index_Keywords" in df.columns:
        df["Keywords"] = (
            df.Author_Keywords.map(lambda w: "" if pd.isna(w) else w)
            + ";"
            + df.Index_Keywords.map(lambda w: "" if pd.isna(w) else w)
        )
        df["Keywords"] = df.Keywords.map(
            lambda w: pd.NA if w[0] == ";" and len(w) == 1 else w
        )
        df["Keywords"] = df.Keywords.map(
            lambda w: w[1:] if w[0] == ";" else w, na_action="ignore"
        )
        df["Keywords"] = df.Keywords.map(
            lambda w: w[:-1] if w[-1] == ";" else w, na_action="ignore"
        )
        df["Keywords"] = df.Keywords.map(
            lambda w: ";".join(sorted(set(w.split(";")))), na_action="ignore"
        )

    ##
    ## Keywords_CL new field creation
    ##
    if "Author_Keywords_CL" in df.columns and "Index_Keywords_CL" in df.columns:
        df["Keywords_CL"] = (
            df.Author_Keywords_CL.map(lambda w: "" if pd.isna(w) else w)
            + ";"
            + df.Index_Keywords_CL.map(lambda w: "" if pd.isna(w) else w)
        )
        df["Keywords_CL"] = df.Keywords_CL.map(
            lambda w: pd.NA if w[0] == ";" and len(w) == 1 else w
        )
        df["Keywords_CL"] = df.Keywords_CL.map(
            lambda w: w[1:] if w[0] == ";" else w, na_action="ignore"
        )
        df["Keywords_CL"] = df.Keywords_CL.map(
            lambda w: w[:-1] if w[-1] == ";" else w, na_action="ignore"
        )
        df["Keywords_CL"] = df.Keywords_CL.map(
            lambda w: ";".join(sorted(set(w.split(";")))), na_action="ignore"
        )

    ##
    ## Title Keywords
    ##
    if "Title_Keywords" in df.columns:
        df["Title_Keywords_CL"] = map_(df, "Title_Keywords", th.apply_as_dict)

    ##
    ## Abstract
    ##
    for column in [
        "Abstract_Author_Keywords",
        "Abstract_Index_Keywords",
        "Abstract_Keywords",
    ]:
        if column in df.columns:
            df[column + "_CL"] = map_(df, column, th.apply_as_dict)

    ##
    ## Saves!
    ##
    df.to_csv(output_file, index=False)
