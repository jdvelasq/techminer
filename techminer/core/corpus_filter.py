import pandas as pd


def corpus_filter(data, clusters, cluster):

    data = data.copy()
    data["SELECTED"] = False
    column = clusters[0]
    members = set(clusters[1][cluster])
    data["COLUMN"] = data[column].copy()
    data["COLUMN"] = data.COLUMN.map(lambda w: set(w.split(";")), na_action="ignore")
    data["COLUMN"] = data.COLUMN.map(
        lambda w: True if len(w & members) > 0 else False, na_action="ignore"
    )
    data["COLUMN"] = data.COLUMN.map(lambda w: False if pd.isna(w) else w)
    data = data[data.COLUMN]
    data.pop("COLUMN")
    data.pop("SELECTED")
    return data
