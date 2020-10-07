import pandas as pd

#  from techminer.core.params import MULTIVALUED_COLS


def map_(x, column, f):

    x = x.copy()
    if x[column].dtype != "int64" and column != "Abstract" and column != "Title":
        z = x[column].map(lambda w: w.split(";") if not pd.isna(w) else w)
        z = z.map(lambda w: [f(z.strip()) for z in w] if isinstance(w, list) else w)
        z = z.map(
            lambda w: [z for z in w if not pd.isna(z)] if isinstance(w, list) else w
        )
        z = z.map(lambda w: ";".join(w) if isinstance(w, list) else w)
        return z

    # if column in [
    #     "Abstract_Phrase_Keywords",
    #     "Abstract_Phrase_Keywords_CL",
    #     "Abstract_Phrase_Author_Keywords",
    #     "Abstract_Phrase_Author_Keywords_CL",
    #     "Abstract_Phrase_Index_Keywords",
    #     "Abstract_Phrase_Index_Keywords_CL",
    # ]:
    #     z = x[column].map(lambda w: w.split("//"), na_action="ignore")
    #     z = z.map(lambda w: [z.split(";") for z in w], na_action="ignore")
    #     z = z.map(lambda w: [[f(y.strip()) for y in z] for z in w], na_action="ignore")
    #     z = z.map(lambda w: [";".join(z) for z in w], na_action="ignore")
    #     z = z.map(lambda w: "//".join(w), na_action="ignore")
    #     return z

    return x[column].map(lambda w: f(w))
