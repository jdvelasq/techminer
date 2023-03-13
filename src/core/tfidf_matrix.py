import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer


def TFIDF_matrix(
    TF_matrix,
    norm="l2",
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False,
    max_items=3000,
):

    result = (
        TfidfTransformer(
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
        )
        .fit_transform(TF_matrix)
        .toarray()
    )

    result = pd.DataFrame(result, columns=TF_matrix.columns, index=TF_matrix.index)

    if len(result.columns) > max_items:
        terms = result.sum(axis=0)
        terms = terms.sort_values(ascending=False)
        terms = terms.head(max_items)
        result = result.loc[:, terms.index]
        rows = result.sum(axis=1)
        rows = rows[rows > 0]
        result = result.loc[rows.index, :]

    return result
