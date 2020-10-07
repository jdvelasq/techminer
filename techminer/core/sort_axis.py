import pandas as pd


def sort_axis(data, num_documents, axis, ascending):
    data = data.copy()
    if axis == 0:
        x = data.index.tolist()
    elif axis == 1:
        x = data.columns.tolist()
    else:
        raise NameError("Invalid axis value:" + str(axis))
    if num_documents is True:
        x = sorted(x, key=lambda w: w.split(" ")[-1], reverse=not ascending)
    else:
        x = sorted(
            x,
            key=lambda w: ":".join(w.split(" ")[-1].split(":")[::-1]),
            reverse=not ascending,
        )
    if isinstance(data, pd.DataFrame):
        if axis == 0:
            data = data.loc[x, :]
        else:
            data = data.loc[:, x]
    else:
        data = data[x]
    return data
