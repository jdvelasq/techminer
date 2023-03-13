from techminer.core.sort_axis import sort_axis


def sort_by_axis(data, sort_by, ascending, axis):

    X = data.copy()
    #  sort_by = sort_by.replace(' ', '_').replace('-','_').replace('/','_').replace('(', '').replace(')', '')

    axis_to_sort = {0: [0], 1: [1], 2: [0, 1],}[axis]

    if sort_by == "Alphabetic":

        for m in axis_to_sort:
            X = X.sort_index(axis=m, ascending=ascending).sort_index(
                axis=m, ascending=ascending
            )

    elif (
        sort_by == "Num Documents"
        or sort_by == "Global Citations"
        or sort_by == "Num_Documents"
        or sort_by == "Global_Citations"
    ):

        for m in axis_to_sort:
            X = sort_axis(
                data=X,
                num_documents=(sort_by == "Num_Documents")
                or (sort_by == "Num Documents"),
                axis=m,
                ascending=ascending,
            )

    elif sort_by == "Data":

        for m in axis_to_sort:
            if m == 0:
                t = X.max(axis=1)
                X = X.loc[t.sort_values(ascending=ascending).index, :]
            else:
                t = X.max(axis=0)
                X = X.loc[:, t.sort_values(ascending=ascending).index]
    else:

        raise NameError("Invalid 'Sort by' value:" + sort_by)

    return X
