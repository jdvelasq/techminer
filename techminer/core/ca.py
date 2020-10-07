import numpy as np
import pandas as pd


class CA:
    def __init__(self):
        pass

    def fit(self, X):

        X = X.copy()
        #
        # 1.-- Observed proportions (P)
        #
        P = X / X.sum().sum()

        #
        # 2.-- Row and column masses
        #
        row_masses = P.sum(axis=1)
        col_masses = P.sum(axis=0)

        #
        # 3.-- Expected proportions (E)
        #
        E = np.outer(row_masses, col_masses)

        #
        # 4.-- Residuals: R = P - E
        #
        R = P - E

        #
        # 5.-- Indexed residuals I = R / E
        #
        I = R / E

        #
        # 6.-- Standarized residuals Z = I * sqrt(E)
        #
        Z = I * np.sqrt(E)

        #
        # 7.-- u, s, vh = SVD(Z)
        #        * u: left singular vectors
        #        * d: vectors of singular values
        #        * v: matrix right singular vectors
        #
        u, d, v = np.linalg.svd(Z, full_matrices=False)
        K = min(len(X.columns), len(X.index))
        u = pd.DataFrame(
            u, columns=["Dim-{}".format(i) for i in range(K)], index=X.index,
        )
        d = pd.Series(d, index=["Dim-{}".format(i) for i in range(K)])
        v = pd.DataFrame(
            v.T, columns=["Dim-{}".format(i) for i in range(K)], index=X.columns,
        )

        #
        # 8.-- eigenvalues d^2
        #
        self.eigenvalues_ = np.power(d, 2)

        #
        # 9.-- explained variance
        #
        self.explained_variance_ = self.eigenvalues_ / sum(self.eigenvalues_)

        #
        # 10.-- standard coordinates:
        #         standard coordiantes rows:  u[row, :] = sqrt(row_masses[row])
        #         standard coordiantes cols:  v[:, col] = sqrt(col_masses[col])
        #
        standard_coordinates_rows = u.copy()
        for i in standard_coordinates_rows.columns:
            standard_coordinates_rows[i] = standard_coordinates_rows[
                i
            ] / row_masses.map(np.sqrt)
        self.standard_coordinates_rows_ = standard_coordinates_rows

        standard_coordinates_cols = v.copy()
        for i in standard_coordinates_cols.columns:
            standard_coordinates_cols[i] = standard_coordinates_cols[
                i
            ] / col_masses.map(np.sqrt)
        self.standard_coordinates_cols_ = standard_coordinates_cols

        #
        # 11.-- principal coordinates: standard coordinates * eigenvalues
        #
        principal_coordinates_rows = standard_coordinates_rows.copy()
        for i in standard_coordinates_rows.index:
            principal_coordinates_rows.at[i, :] = (
                principal_coordinates_rows.loc[i, :] * d
            )
        self.principal_coordinates_rows_ = principal_coordinates_rows

        principal_coordinates_cols = standard_coordinates_cols.copy()
        for i in standard_coordinates_cols.index:
            principal_coordinates_cols.at[i, :] = (
                principal_coordinates_cols.loc[i, :] * d
            )
        self.principal_coordinates_cols_ = principal_coordinates_cols

        #
        # 12.-- Scores
        #
        scores = self.principal_coordinates_rows_.applymap(lambda w: np.power(w, 2))
        sum_scores = scores.sum(axis=1)
        for i in scores.index:
            scores.at[i, :] = scores.loc[i, :] / sum_scores[i]
        self.row_scores_ = scores

        scores = self.principal_coordinates_cols_.applymap(lambda w: np.power(w, 2))
        sum_scores = scores.sum(axis=1)
        for i in scores.index:
            scores.at[i, :] = scores.loc[i, :] / sum_scores[i]
        self.col_scores_ = scores
