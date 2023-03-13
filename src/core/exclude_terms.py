import json


def exclude_terms(data, axis):

    #
    # Loads terms to exclude
    #
    with open("filters.json", "r") as f:
        dict_ = json.load(f)
    if dict_["excluded_terms"] == "---":
        return data

    with open(dict_["excluded_terms"], "r") as f:
        exclude = f.readlines()
    exclude = [term.replace("\n", "") for term in exclude]
    exclude = [term.strip() for term in exclude]
    exclude = [term for term in exclude if term != ""]
    if len(exclude) == 0:
        return data

    data = data.copy()

    if axis == 0:
        new_axis = data.index
    elif axis == 1:
        new_axis = data.columns
    else:
        raise NameError("Invalid axis value:" + str(axis))

    #
    # Exclude
    #
    if exclude is not None:
        new_axis = [w for w in new_axis if w not in exclude]

    if axis == 0:
        data = data.loc[new_axis, :]
    else:
        data = data.loc[:, new_axis]

    return data
