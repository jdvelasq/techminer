import json


def filter_records(x):

    with open("filters.json", "r") as f:
        dict_ = json.load(f)

    x = x[(x.Year >= dict_["years"][0]) & (x.Year <= dict_["years"][1])]
    x = x[
        (x.Global_Citations >= dict_["citations"][0])
        & (x.Global_Citations <= dict_["citations"][1])
    ]
    x = x[
        x.Bradford_Law_Zone.map(
            lambda w: w <= dict_["bradford_law_zones"], na_action="ignore"
        )
    ]
    x = x[
        x.Document_Type.map(lambda w: w in dict_["selected_types"], na_action="ignore")
    ]

    if dict_["selected_cluster"] != "---":
        IDs = dict_[dict_["selected_cluster"]]
        x = x[x.ID.map(lambda w: w in IDs)]

    return x
