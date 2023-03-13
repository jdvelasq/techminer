import pandas as pd

from techminer.core.extract_words import extract_words
from techminer.core.map import map_
from techminer.core.thesaurus import read_textfile


def extract_abstract_words(
    input_file="techminer.csv",
    thesaurus_file="thesaurus-title-words-cleaned.txt",
    output_file="techminer.csv",
):

    data = pd.read_csv(input_file)
    data["Abstract_words"] = extract_words(data=data, text=data.Abstract)
    th = read_textfile(thesaurus_file)
    th = th.compile_as_dict()
    data["Abstract_words"] = map_(data, "Abstract_words", th.apply_as_dict)
    data.to_csv(output_file, index=False)

