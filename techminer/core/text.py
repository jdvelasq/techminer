"""
Text manipulation
==================================================================================================

This module contains functions for manipulating texts.


Functions in this module
----------------------------------------------------------------------------------------

"""

import json
import re
import string
from os.path import dirname, join

import pandas as pd
from nltk.stem import PorterStemmer, SnowballStemmer


def find_string(
    patterns, x, ignore_case=True, full_match=False, use_re=False, explode=True
):
    r"""Find patterns in the elements of a list.

    >>> x = ['aa;b', 'c;d', 'A', 'e', None]
    >>> find_string('a', x)
    ['A', 'aa']

    >>> find_string('a', x, ignore_case=False)
    ['aa']

    >>> find_string('a', x, full_match=True)
    ['A']

    """
    #
    if explode is True:
        x = [z for e in x if isinstance(e, str) for z in e.split(";")]
    x = list(set(x))
    #
    if isinstance(patterns, str):
        patterns = [patterns]
    #
    results = []
    if use_re is False:
        patterns = [re.escape(pattern) for pattern in patterns]
    if full_match is True:
        patterns = ["^" + pattern + "$" for pattern in patterns]
    if ignore_case is True:
        patterns = [re.compile(pattern, re.I) for pattern in patterns]
    else:
        patterns = [re.compile(pattern) for pattern in patterns]
    for term in x:
        for pattern in patterns:
            result = pattern.findall(term)
            if len(result):
                results.append(term)
    return sorted(set(results))


def stemming_AND(patterns, x, stemmer="porter", explode=True):
    """

    >>> x = [
    ...    'Computer vision',
    ...    'Computer simulation',
    ...    'computer theory',
    ...    'control systems',
    ...    'Computer control systems',
    ...    'hardware',
    ... ]
    >>> stemming_AND('control systems', x)
    ['Computer control systems', 'control systems']

    """

    def prepare(term):
        term = remove_accents(term)
        term = term.lower()
        term = term.split()
        term = [w.strip() for w in term]
        term = [stemmer(w) for w in term]
        term = sorted(set(term))
        term = " ".join(term)
        return term

    if stemmer == "porter":
        stemmer = stemmer_porter
    else:
        stemmer = stemmer_snowball

    if explode is True:
        x = [z for e in x if isinstance(e, str) for z in e.split(";")]

    x = {prepare(t): t for t in x}

    if not isinstance(patterns, list):
        patterns = [patterns]
    patterns = [prepare(pattern) for pattern in patterns]

    results = []
    for key in x.keys():
        for pattern in patterns:
            if pattern in key:
                results.append(x[key])
                continue

    return sorted(set(results))


def stemming_OR(patterns, x, stemmer="porter", explode=True):
    """

    >>> x = [
    ...    'Computer vision',
    ...    'Computer simulation',
    ...    'computer theory',
    ...    'control systems',
    ...    'Computer control systems',
    ...    'hardware',
    ... ]
    >>> stemming_OR('computer software', x)
    ['Computer control systems', 'Computer simulation', 'Computer vision', 'computer theory']
    
    """

    def prepare(term):
        term = remove_accents(term)
        term = term.lower()
        term = term.split()
        term = [w.strip() for w in term]
        term = [stemmer(w) for w in term]
        term = sorted(set(term))
        term = " ".join(term)
        return term

    if stemmer == "porter":
        stemmer = stemmer_porter
    else:
        stemmer = stemmer_snowball

    if explode is True:
        x = [z for e in x if isinstance(e, str) for z in e.split(";")]

    x = {prepare(t): t for t in x}

    if not isinstance(patterns, list):
        patterns = [patterns]
    patterns = [prepare(pattern) for pattern in patterns]

    results = []
    for key in x.keys():
        for pattern in patterns:
            found = None
            for word_key in key.split():
                for word_pattern in pattern.split():
                    if word_pattern in word_key:
                        found = x[key]
                        continue
                if found is not None:
                    results.append(x[key])
                    continue

    return sorted(set(results))


def steamming(pattern, text, stemmer=""):
    """

    Examples
    ----------------------------------------------------------------------------------------------


    """
    text = remove_accents(text)
    pattern = remove_accents(pattern)

    text = text.strip().lower()
    pattern = pattern.strip().lower()

    porter = PorterStemmer()

    pattern = [porter.stem(w) for w in pattern.split()]
    text = [porter.stem(w) for w in text.split()]

    return [m in text for m in pattern]


def steamming_all(pattern, text):
    """

    Examples
    ----------------------------------------------------------------------------------------------

    >>> steamming_all('computers cars', 'car computing')
    True

    >>> steamming_all('computers cars', 'car houses')
    False

    """
    return all(steamming(pattern, text))


def steamming_any(pattern, text):
    """

    Examples
    ----------------------------------------------------------------------------------------------

    >>> steamming_any('computers cars', 'car computing')
    True

    >>> steamming_any('computers cars', 'computing house')
    True

    >>> steamming_all('computers cars', 'tree houses')
    False

    """
    return any(steamming(pattern, text))


# # def replace_string(
# #     pattern, x, repl=None, ignore_case=True, full_match=False, use_re=False
# # ):
# #     """Replace pattern in string.

# #     Args:
# #         pattern (string)
# #         x (string)
# #         repl (string, None)
# #         ignore_case (bool)
# #         full_match (bool)
# #         use_re (bool)

# #     Returns:
# #         string or []

# #     """

# #     if use_re is False:
# #         pattern = re.escape(pattern)

# #     if full_match is True:
# #         pattern = "^" + pattern + "$"

# #     if ignore_case is True:
# #         return re.sub(pattern, repl, x, re.I)
# #     return re.sub(pattern, repl, x)


def one_gram(x):
    """Computes the 1-gram representation of string x.

    See https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth

    Args:
        x (string): string to convert.

    Returns:
        string.


    Examples
    ----------------------------------------------------------------------------------------------

    >>> one_gram('neural net')
    'aelnrtu'


    """
    if x is None:
        return None
    x = x.strip().lower()
    x = re.sub("-", " ", x)
    x = re.sub("[" + string.punctuation + "]", "", x)
    x = remove_accents(x)
    x = x.replace(" ", "")
    x = sorted(list(set(x)))
    return "".join(x)


def two_gram(x):
    """Computes the 2-gram representation of string x.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> two_gram('neural net')
    'aleteulnneraur'


    """
    if x is None:
        return None
    x = x.strip().lower()
    x = re.sub("-", " ", x)
    x = re.sub("[" + string.punctuation + "]", "", x)
    x = remove_accents(x)
    x = x.replace(" ", "")
    x = list(x)
    x = ["".join([x[i], x[i + 1]]) for i in range(len(x) - 1)]
    x = sorted(list(set(x)))
    return "".join(x)


def stemmer_porter(x):
    """Computes the stemmer transformation of string x.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> stemmer_porter('neural net')
    'net neural'


    """
    if x is None:
        return None
    x = x.strip().lower()
    x = re.sub("-", " ", x)
    x = re.sub("[" + string.punctuation + "]", "", x)
    x = remove_accents(x)
    s = PorterStemmer()
    x = sorted(set([s.stem(w) for w in x.split()]))
    return " ".join(x)


def stemmer_snowball(x):
    """Computes the stemmer transformation of string x.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> stemmer_snowball('neural net')
    'net neural'


    """
    if x is None:
        return None
    x = x.strip().lower()
    x = re.sub("-", " ", x)
    x = re.sub("[" + string.punctuation + "]", "", x)
    x = remove_accents(x)
    s = SnowballStemmer("english")
    x = sorted(set([s.stem(w) for w in x.split()]))
    return " ".join(x)


def fingerprint(x):
    """Computes 'fingerprint' representation of string x.

    See https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth

    Args:
        x (string): string to convert.

    Returns:
        string.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> fingerprint('a A b')
    'a b'
    >>> fingerprint('b a a')
    'a b'
    >>> fingerprint(None) is None
    True
    >>> fingerprint('b c')
    'b c'
    >>> fingerprint(' c b ')
    'b c'


    """
    if x is None:
        return None
    x = x.strip().lower()
    x = re.sub("-", " ", x)
    x = re.sub("[" + string.punctuation + "]", "", x)
    x = remove_accents(x)
    x = sorted(set(w for w in x.split()))
    return " ".join(x)


def remove_accents(text):
    """Translate non-ascii charaters to ascii equivalent. Based on Google Open Refine.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> remove_accents('áéíóúñÁÉÍÓÚÑ')
    'aeiounAEIOUN'

    """

    def translate(c):

        if c in [
            "\u0100",
            "\u0102",
            "\u00C5",
            "\u0104",
            "\u00C0",
            "\u00C1",
            "\u00C2",
            "\u00C3",
            "\u00C4",
        ]:
            return "A"

        if c in [
            "\u00E0",
            "\u00E1",
            "\u00E2",
            "\u00E3",
            "\u00E4",
            "\u0103",
            "\u0105",
            "\u00E5",
            "\u0101",
        ]:
            return "a"

        if c in [
            "\u00C7",
            "\u0106",
            "\u0108",
            "\u010A",
            "\u010C",
        ]:
            return "C"

        if c in [
            "\u010D",
            "\u00E7",
            "\u0107",
            "\u010B",
            "\u0109",
        ]:
            return "c"

        if c in [
            "\u00D0",
            "\u010E",
            "\u0110",
        ]:
            return "D"

        if c in [
            "\u0111",
            "\u00F0",
            "\u010F",
        ]:
            return "d"

        if c in [
            "\u00C8",
            "\u00C9",
            "\u00CA",
            "\u00CB",
            "\u0112",
            "\u0114",
            "\u0116",
            "\u0118",
            "\u011A",
        ]:
            return "E"

        if c in [
            "\u011B",
            "\u0119",
            "\u00E8",
            "\u00E9",
            "\u00EA",
            "\u00EB",
            "\u0113",
            "\u0115",
            "\u0117",
        ]:
            return "e"

        if c in [
            "\u011C",
            "\u011E",
            "\u0120",
            "\u0122",
        ]:
            return "G"

        if c in [
            "\u0123",
            "\u011D",
            "\u011F",
            "\u0121",
        ]:
            return "g"

        if c in [
            "\u0124",
            "\u0126",
        ]:
            return "H"

        if c in [
            "\u0127",
            "\u0125",
        ]:
            return "h"

        if c in [
            "\u00CC",
            "\u00CD",
            "\u00CE",
            "\u00CF",
            "\u0128",
            "\u012A",
            "\u012C",
            "\u012E",
            "\u0130",
        ]:
            return "I"

        if c in [
            "\u0131",
            "\u012F",
            "\u012D",
            "\u00EC",
            "\u012B",
            "\u0129",
            "\u00EF",
            "\u00EE",
            "\u00ED",
            "\u017F",
        ]:
            return "i"

        if c in [
            "\u0134",
        ]:
            return "J"
        if c in [
            "\u0135",
        ]:
            return "j"

        if c in [
            "\u0136",
        ]:
            return "K"

        if c in [
            "\u0137",
            "\u0138",
        ]:
            return "k"

        if c in [
            "\u0139",
            "\u013B",
            "\u013D",
            "\u013F",
            "\u0141",
        ]:
            return "L"

        if c in [
            "\u0142",
            "\u013A",
            "\u013C",
            "\u013E",
            "\u0140",
        ]:
            return "l"

        if c in [
            "\u00D1",
            "\u0143",
            "\u0145",
            "\u0147",
        ]:
            return "N"

        if c in [
            "\u014B",
            "\u014A",
            "\u0149",
            "\u0148",
            "\u0146",
            "\u0144",
            "\u00F1",
        ]:
            return "n"

        if c in [
            "\u00D2",
            "\u00D3",
            "\u00D4",
            "\u00D5",
            "\u00D6",
            "\u00D8",
            "\u014C",
            "\u014E",
            "\u0150",
        ]:
            return "O"

        if c in [
            "\u0151",
            "\u00F2",
            "\u00F3",
            "\u00F4",
            "\u00F5",
            "\u00F6",
            "\u00F8",
            "\u014F",
            "\u014D",
        ]:
            return "o"

        if c in [
            "\u0154",
            "\u0156",
            "\u0158",
        ]:
            return "R"

        if c in [
            "\u0159",
            "\u0155",
            "\u0157",
        ]:
            return "r"

        if c in [
            "\u015A",
            "\u015C",
            "\u015E",
            "\u0160",
        ]:
            return "S"

        if c in [
            "\u0161",
            "\u015B",
            "\u015F",
            "\u015D",
        ]:
            return "s"

        if c in [
            "\u0162",
            "\u0164",
            "\u0166",
        ]:
            return "T"

        if c in [
            "\u0167",
            "\u0163",
            "\u0165",
        ]:
            return "t"

        if c in [
            "\u00D9",
            "\u00DA",
            "\u00DB",
            "\u00DC",
            "\u0168",
            "\u016A",
            "\u016E",
            "\u0170",
            "\u0172",
            "\u016C",
        ]:
            return "U"

        if c in [
            "\u0173",
            "\u00F9",
            "\u00FA",
            "\u00FB",
            "\u00FC",
            "\u0169",
            "\u016B",
            "\u016D",
            "\u0171",
            "\u016F",
        ]:
            return "u"

        if c in [
            "\u0174",
        ]:
            return "W"

        if c in [
            "\u0175",
        ]:
            return "w"

        if c in [
            "\u0178",
            "\u00DD",
            "\u0176",
        ]:
            return "Y"

        if c in [
            "\u0177",
            "\u00FD",
            "\u00FF",
        ]:
            return "y"

        if c in [
            "\u0179",
            "\u017B",
            "\u017D",
        ]:
            return "Z"

        if c in [
            "\u017E",
            "\u017A",
            "\u017C",
        ]:
            return "z"

        return c

    return "".join([translate(c) for c in text])


if __name__ == "__main__":

    import doctest

    doctest.testmod()
