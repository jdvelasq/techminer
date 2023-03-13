import numpy as np


def normalize_network(X, normalization=None):
    """
    """
    X = X.copy()

    if isinstance(normalization, str) and normalization == "None":
        normalization = None

    if normalization is None:
        X = X.applymap(lambda w: int(w))
    else:
        X = X.applymap(lambda w: float(w))

    M = X.copy()

    if normalization == "Jaccard":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] / (
                    M.loc[row, row] + M.at[col, col] - M.at[row, col]
                )

    if normalization == "Dice":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] / (
                    M.loc[row, row] + M.at[col, col] + 2 * M.at[row, col]
                )

    if normalization == "Salton/Cosine":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] / np.sqrt(
                    (M.loc[row, row] * M.at[col, col])
                )

    if normalization == "Equivalence":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] ** 2 / (
                    M.loc[row, row] * M.at[col, col]
                )

    ## inclusion
    if normalization == "Inclusion":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] / min(M.loc[row, row], M.at[col, col])

    if normalization == "Mutual Information":
        N = len(M.columns)
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = np.log(
                    M.at[row, col] / (N * M.loc[row, row] * M.at[col, col])
                )

    if normalization == "Association":
        for col in M.columns:
            for row in M.index:
                X.at[row, col] = M.at[row, col] / (M.loc[row, row] * M.at[col, col])

    return X
