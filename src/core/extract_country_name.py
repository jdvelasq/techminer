import json
import re
from os.path import dirname, join

import pandas as pd

from techminer.core.thesaurus import load_file_as_dict

NAMES = [
    ("universidad del norte", "colombia"),
    ("universidad nacional de colombia", "colombia"),
    ("universidad de antioquia", "colombia"),
    ("universidad industrial de santander", "colombia"),
    ("universidad del valle", "colombia"),
    ("universidad del cauca", "colombia"),
    ("instituto tecnologico metropolitano", "colombia"),
]


def extract_country_name(x):
    #
    if pd.isna(x) or x is None:
        return pd.NA
    ##
    ## List of standardized country names
    ##
    module_path = dirname(__file__)
    filename = join(module_path, "../data/country_codes.data")
    country_codes = load_file_as_dict(filename)
    country_names = list(country_codes.values())
    country_names = [name.lower() for w in country_names for name in w]

    ##
    ## Replace administrative regions by country names
    ## for the current string
    ##
    x = x.lower()
    x = x.strip()
    for a, b in [
        ("bosnia and herzegovina", "bosnia and herz."),
        ("brasil", "brazil"),
        ("czech republic", "czechia"),
        ("espana", "spain"),
        ("hong kong", "china"),
        ("macao", "china"),
        ("macau", "china"),
        ("peoples r china", "china"),
        ("rusia", "russia"),
        ("russian federation", "russia"),
        ("united states of america", "united states"),
        ("usa", "united states"),
    ]:
        x = re.sub(a, b, x)

    ##
    ## Name search in the affiliation (x)
    ##
    for z in reversed(x.split(",")):

        z = z.strip()

        ##
        ## Exact match in list of stadardized country names
        ##
        if z.lower() in country_names:
            return z.lower()

        ##
        ## Discard problems of multiple blank spaces
        ##
        z = " ".join([w.strip() for w in z.lower().split(" ")])
        if z in country_names:
            return z.lower()

    ##
    ## Repair country name from institution name
    ##
    for institution, country in NAMES:
        if institution in x:
            return country

    for country_name in country_names:
        if country_name in x:
            return country_name

    ##
    ## Country not found
    ##
    return pd.NA
