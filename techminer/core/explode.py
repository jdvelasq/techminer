#  from techminer.core.params import MULTIVALUED_COLS


def explode(x, column):

    # if column == "Abstract_phrase_words":
    #     x = x.copy()
    #     x[column] = x[column].map(lambda w: w.split("//"), na_action="ignore")
    #     x = x.explode(column)
    #     x["ID"] = x.ID.map(str) + "-" + x.groupby(["ID"]).cumcount().map(str)
    #     x[column] = x[column].map(lambda w: w.split(";"), na_action="ignore")
    #     x = x.explode(column)
    #     print(x[column].head())
    #     x[column] = x[column].map(lambda w: w.strip(), na_action="ignore")
    #     x = x.reset_index(drop=True)
    #     return x

    # if column in [
    #     "Abstract_Phrase_Author_Keywords_CL",
    #     "Abstract_Phrase_Author_Keywords",
    #     "Abstract_Phrase_Index_Keywords_CL",
    #     "Abstract_Phrase_Index_Keywords",
    #     "Abstract_Phrase_Keywords_CL",
    #     "Abstract_Phrase_Keywords",
    # ]:
    #     print("******")
    #     x = x.copy()
    #     x[column] = x[column].map(lambda w: w.split("//"), na_action="ignore")
    #     x = x.explode(column)
    #     x["ID"] = x.ID.map(str) + "-" + x.groupby(["ID"]).cumcount().map(str)
    #     x[column] = x[column].map(lambda w: w.split(";"), na_action="ignore")
    #     x = x.explode(column)
    #     x[column] = x[column].map(lambda w: w.strip(), na_action="ignore")
    #     x = x.reset_index(drop=True)
    #     print("=======")
    #     return x

    # if column in MULTIVALUED_COLS:
    #     x = x.copy()
    #     x[column] = x[column].map(
    #         lambda w: sorted(list(set(w.split(";")))) if isinstance(w, str) else w
    #     )
    #     x = x.explode(column)
    #     x[column] = x[column].map(lambda w: w.strip() if isinstance(w, str) else w)
    #     x = x.reset_index(drop=True)
    # return x

    if x[column].dtype != "int64":
        x = x.copy()
        x[column] = x[column].map(
            lambda w: sorted(list(set(w.strip().split(";"))))
            if isinstance(w, str)
            else w
        )
        x = x.explode(column)
        x[column] = x[column].map(lambda w: w.strip() if isinstance(w, str) else w)
        x = x.reset_index(drop=True)
    return x