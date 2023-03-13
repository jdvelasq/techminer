import pandas as pd

# from techminer.core.logging_info import logging_info
from techminer.core.map import map_
from techminer.core.thesaurus import read_textfile


def apply_institutions_thesaurus(
    input_file="corpus.csv",
    thesaurus_file="TH_institutions.txt",
    output_file="corpus.csv",
    logging_info=None,
):

    data = pd.read_csv(input_file)

    ##
    ## Loads the thesaurus
    ##
    th = read_textfile(thesaurus_file)
    th = th.compile_as_dict()

    ##
    ## Copy affiliations to institutions
    ##
    data["Institutions"] = data.Affiliations.map(
        lambda w: w.lower().strip(), na_action="ignore"
    )

    ##
    ## Cleaning
    ##
    logging_info("Extract and cleaning institutions.")
    data["Institutions"] = map_(
        data, "Institutions", lambda w: th.apply_as_dict(w, strict=True)
    )

    logging_info("Extracting institution of first author ...")
    data["Institution_1st_Author"] = data.Institutions.map(
        lambda w: w.split(";")[0] if isinstance(w, str) else w
    )

    ##
    ## Finish!
    ##
    data.to_csv(output_file, index=False)
    logging_info("Finished!!!")
