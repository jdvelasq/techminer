def limit_to_exclude(data, axis, column, limit_to, exclude):

    data = data.copy()

    if axis == 0:
        new_axis = data.index
    elif axis == 1:
        new_axis = data.columns
    else:
        raise NameError("Invalid axis value:" + str(axis))

    #
    # Limit to
    #
    if isinstance(limit_to, dict):
        if column in limit_to.keys():
            limit_to = limit_to[column]
        else:
            limit_to = None

    if limit_to is not None:
        new_axis = [w for w in new_axis if w in limit_to]

    #
    # Exclude
    #
    if isinstance(exclude, dict):
        if column in exclude.keys():
            exclude = exclude[column]
        else:
            exclude = None

    if exclude is not None:
        new_axis = [w for w in new_axis if w not in exclude]

    if axis == 0:
        data = data.loc[new_axis, :]
    else:
        data = data.loc[:, new_axis]

    return data

