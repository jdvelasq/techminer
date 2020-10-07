import numpy as np
import pandas as pd

from techminer.core.explode import explode


def TF_matrix(data, column, scheme=None, min_occurrence=1, max_occurrence=10000):
    #
    X = data[[column, "ID"]].copy()
    X["value"] = 1.0
    X = explode(X, column)
    X = X.groupby([column, "ID"], as_index=False).agg({"value": np.sum})
    result = pd.pivot_table(
        data=X,
        index="ID",
        columns=column,
        margins=False,
        fill_value=0.0,
    )
    result.columns = [b for _, b in result.columns]
    result = result.reset_index(drop=True)

    terms = result.sum(axis=0)
    terms = terms.sort_values(ascending=False)
    terms = terms[terms >= min_occurrence]
    terms = terms[terms < max_occurrence]
    result = result.loc[:, terms.index]

    rows = result.sum(axis=1)
    rows = rows[rows > 0]
    result = result.loc[rows.index, :]

    if scheme is None or scheme == "raw":
        return result

    if scheme == "binary":
        result = result.applymap(lambda w: 1 if w > 0 else 0)

    if scheme == "log":
        result = result.applymap(lambda w: np.log(1 + w))

    return result
