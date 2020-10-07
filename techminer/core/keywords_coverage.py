import pandas as pd
from techminer.core import explode


def keywords_coverage(data, column, keywords_list):

    data = data[[column, "ID"]].dropna()
    num_documents = len(data)
    x = pd.DataFrame({column: keywords_list})
    x["Cum Coverage (Cum Num Documents)"] = 0

    data[column] = data[column].map(lambda w: w.split(";"))
    data["SELECTED"] = False
    keywords_list = [" ".join(keyword.split(" ")[:-1]) for keyword in keywords_list]
    x.index = keywords_list
    for keyword in keywords_list:
        data["SELECTED"] = data.SELECTED | data[column].map(lambda w: keyword in w)
        selected = data[data.SELECTED][["ID"]].drop_duplicates()
        x.loc[keyword, "Cum Coverage (Cum Num Documents)"] = len(selected)

    x["Cum Coverage (%)"] = x["Cum Coverage (Cum Num Documents)"].map(
        lambda w: str(round(100 * w / num_documents, 1)) + " %"
    )
    x = x.reset_index()
    x.pop("index")
    return x
