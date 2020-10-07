import numpy as np

from techminer.core.explode import explode


def add_counters_to_axis(X, axis, data, column):

    X = X.copy()
    data = data.copy()
    data["Num_Documents"] = 1
    m = (
        explode(data[[column, "Num_Documents", "Global_Citations", "ID"]], column)
        .groupby(column, as_index=True)
        .agg(
            {
                "Num_Documents": np.sum,
                "Global_Citations": np.sum,
            }
        )
    )
    n_Num_Documents = int(np.log10(m["Num_Documents"].max())) + 1
    if m["Global_Citations"].max() > 0:
        n_Global_Citations = int(np.log10(m["Global_Citations"].max())) + 1
    else:
        n_Global_Citations = 1
    fmt = "{} {:0" + str(n_Num_Documents) + "d}:{:0" + str(n_Global_Citations) + "d}"
    new_names = {
        key: fmt.format(key, int(nd), int(tc))
        for key, nd, tc in zip(m.index, m.Num_Documents, m.Global_Citations)
    }
    if axis == 0:
        X.index = [new_names[t] for t in X.index]
    elif axis == 1:
        X.columns = [new_names[t] for t in X.columns]
    else:
        raise NameError("Invalid axis value:" + str(axis))

    return X
