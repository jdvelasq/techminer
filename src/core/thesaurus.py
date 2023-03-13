import json
import re
from os.path import dirname, join

import pandas as pd

from techminer.core.text import (
    find_string,
    fingerprint,
    one_gram,
    stemmer_porter,
    stemmer_snowball,
    two_gram,
)


def text_clustering(x, name_strategy="mostfrequent", key="porter", transformer=None):
    #
    def remove_parenthesis(x):
        if "(" in x:
            text_to_remove = x[x.find("(") : x.find(")") + 1]
            x = x.replace(text_to_remove, "")
            x = " ".join([w.strip() for w in x.split()])
        return x

    def remove_brackets(x):
        if "[" in x:
            text_to_remove = x[x.find("[") : x.find("]") + 1]
            x = x.replace(text_to_remove, "")
            x = " ".join([w.strip() for w in x.split()])
        return x

    def translate(x):
        z = [bg2am_[z][0] if z in bg2am_.keys() else z for z in x.split()]
        return " ".join(z)

    ##
    ## Preprocessing
    ##
    x = x.dropna()
    x = x.map(lambda w: w.split(";"))
    x = x.explode()
    x = x.map(lambda w: w.strip())
    x = x.unique()

    ##
    ## Creates a dataframe
    ##
    x = pd.DataFrame({"word": x.tolist()})

    ##
    ## Delete terms between '(' and ')' or '[' and ']'
    ## Repace & by and
    ## Delete 'of'
    ##
    x["word_alt"] = x["word"].copy()
    x["word_alt"] = x["word_alt"].map(lambda w: remove_brackets(w))
    x["word_alt"] = x["word_alt"].map(lambda w: remove_parenthesis(w))
    x["word_alt"] = x["word_alt"].map(lambda w: w.replace("&", "and"))
    x["word_alt"] = x["word_alt"].map(lambda w: w.replace(" of ", ""))

    ##
    ## Search for joined terms
    ##
    keywords_with_2_words = x.word_alt[x.word_alt.map(lambda w: len(w) == 2)]
    keywords_with_1_word = keywords_with_2_words.map(lambda w: w.replace(" ", ""))
    for w1, w2 in zip(keywords_with_1_word, keywords_with_2_words):
        if w1 in x.word_alt.tolist():
            x["word_alt"] = x["word_alt"].map(lambda w: w.replace(w1, w2))

    ##
    ## British to american english
    ##
    module_path = dirname(__file__)
    filename = join(module_path, "../data/bg2am.data")
    bg2am_ = load_file_as_dict(filename)
    x["word_alt"] = x["word_alt"].map(lambda w: translate(w))

    ##
    ## key computation
    ##
    if key == "fingerprint":
        f = fingerprint
    elif key == "1-gram":
        f = one_gram
    elif key == "2-gram":
        f = two_gram
    elif key == "porter":
        f = stemmer_porter
    else:
        f = stemmer_snowball
    x["key"] = x.word_alt.map(f)

    ##
    ## groupsby key
    ##
    grp = x.groupby(by="key").agg({"word": list})

    ##
    ## group name selection
    ##
    grp["listlen"] = grp.word.map(len)
    grp_isolated = grp[grp.listlen.map(lambda w: w == 1)]
    grp = grp[grp.listlen.map(lambda w: w > 1)]
    grp["word"] = grp.word.map(lambda w: pd.Series(w))
    grp["groupname"] = None
    if name_strategy is None:
        name_strategy = "mostfrequent"
    if name_strategy == "mostfrequent":
        grp["groupname"] = grp.word.map(
            lambda w: w.value_counts()[w.value_counts() == w.value_counts().max()]
            .sort_index()
            .index[0]
        )
    if name_strategy == "longest":
        grp["groupname"] = grp.word.map(
            lambda w: sorted(w.tolist(), key=len, reverse=True)[0]
        )
    if name_strategy == "shortest":
        grp["groupname"] = grp.word.map(
            lambda w: sorted(w.tolist(), key=len, reverse=False)[0]
        )

    ##
    ## Preffer names with '-'
    ##
    names_with_hyphen = grp.word.map(lambda w: [i for i in w if "-" in i])
    names_with_hyphen = names_with_hyphen.map(lambda w: w[0] if len(w) > 0 else None)
    grp["groupname"] = [
        hyphen_name if hyphen_name is not None else original_name
        for original_name, hyphen_name in zip(grp["groupname"], names_with_hyphen)
    ]

    ##
    ## Transformer
    ##
    if transformer is not None:
        grp["groupname"] = grp.groupname.map(transformer)

    ##
    ## Thesaurus building
    ##
    result = {
        key: sorted(value.tolist()) for key, value in zip(grp.groupname, grp.word)
    }
    result = {**result, **{value[0]: value for value in grp_isolated.word}}
    return Thesaurus(result, ignore_case=False, full_match=True, use_re=False)


def load_file_as_dict(filename):
    #
    dic = {}
    key = None
    values = None
    #
    file = open(filename, "r", encoding="utf-8")
    for word in file:
        word = word.replace("\n", "")
        if len(word.strip()) == 0:
            continue
        if len(word) > 0:
            if word[0] != " ":
                if key is not None:
                    if values == []:
                        raise Exception(
                            "Key '"
                            + key
                            + "' in file '"
                            + filename
                            + "' without values associated"
                        )
                    dic[key] = values
                key = word.strip()
                values = []
            else:
                if values is not None and len(word.strip()) > 0:
                    values.append(word.strip())
    if key not in dic.keys():
        if values == []:
            raise Exception(
                "Key '" + key + "' in file '" + filename + "' without values associated"
            )
        dic[key] = values
    return dic


def read_textfile(filename):
    dict_ = load_file_as_dict(filename)
    th = Thesaurus(x=dict_, ignore_case=True, full_match=False, use_re=False)
    return th


class Thesaurus:
    def __init__(self, x={}, ignore_case=True, full_match=False, use_re=False):
        self._thesaurus = x
        self._ignore_case = ignore_case
        self._full_match = full_match
        self._use_re = use_re
        self._dict = None
        self._compiled = None
        return None

    @property
    def thesaurus(self):
        return self._thesaurus

    def to_textfile(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            for key in sorted(self._thesaurus.keys()):
                file.write(key + "\n")
                for item in self._thesaurus[key]:
                    file.write("    " + item + "\n")

    def find_key(
        self, patterns, ignore_case=True, full_match=False, use_re=False, explode=True
    ):
        return find_string(
            patterns=patterns,
            x=list(self._thesaurus.keys()),
            ignore_case=ignore_case,
            full_match=full_match,
            use_re=use_re,
            explode=explode,
        )

    def __repr__(self):
        """Returns a json representation of the Thesaurus."""
        text = json.dumps(self._thesaurus, indent=2, sort_keys=True)
        text += "\nignore_case={}, full_match={}, use_re={}, compiled={}".format(
            self._ignore_case.__repr__(),
            self._full_match.__repr__(),
            self._use_re.__repr__(),
            self._compiled is not None,
        )
        return text

    def __str__(self):
        return self.__repr__()

    def compile(self):
        self._compiled = {}
        for key in self._thesaurus:
            patterns = self._thesaurus[key]
            if self._use_re is False:
                patterns = [re.escape(pattern) for pattern in patterns]
            if self._full_match is True:
                patterns = ["^" + pattern + "$" for pattern in patterns]
            if self._ignore_case is True:
                patterns = [re.compile(pattern, re.I) for pattern in patterns]
            else:
                patterns = [re.compile(pattern) for pattern in patterns]
            self._compiled[key] = patterns

        return self

    def compile_as_dict(self):
        self._dict = {}
        for key in self._thesaurus:
            for value in self._thesaurus[key]:
                self._dict[value] = key
        return self

    def apply(self, x):
        """Apply a thesaurus to a string x.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> x = pd.Series(
        ...   [
        ...     'aaa', 'bbb', 'ccc aaa', 'ccc bbb', 'ddd eee', 'ddd fff',  None, 'zzz'
        ...   ]
        ... )
        >>> x # doctest: +NORMALIZE_WHITESPACE
        0        aaa
        1        bbb
        2    ccc aaa
        3    ccc bbb
        4    ddd eee
        5    ddd fff
        6       None
        7        zzz
        dtype: object

        >>> patterns = {'aaa':['aaa', 'bbb', 'eee', 'fff'],  '1':['000']}
        >>> thesaurus = Thesaurus(patterns)
        >>> thesaurus = thesaurus.compile()
        >>> thesaurus
        {
          "1": [
            "000"
          ],
          "aaa": [
            "aaa",
            "bbb",
            "eee",
            "fff"
          ]
        }
        ignore_case=True, full_match=False, use_re=False, compiled=True

        >>> x.map(lambda w: thesaurus.apply(w))
        0     aaa
        1     aaa
        2     aaa
        3     aaa
        4     aaa
        5     aaa
        6    <NA>
        7     zzz
        dtype: object

        >>> import pandas as pd
        >>> x = pd.Series(
        ...   [
        ...     '0', '1', '2', '3', None, '4', '5', '6', '7', '8', '9'
        ...   ]
        ... )
        >>> x # doctest: +NORMALIZE_WHITESPACE
        0        0
        1        1
        2        2
        3        3
        4     None
        5        4
        6        5
        7        6
        8        7
        9        8
        10       9
        dtype: object

        >>> patterns = {
        ...     'a':['0', '1', '2'],
        ...     'b':['4', '5', '6'],
        ...     'c':['7', '8', '9']
        ... }
        >>> thesaurus = Thesaurus(patterns, ignore_case=False, full_match=True)
        >>> thesaurus = thesaurus.compile()
        >>> x.map(lambda w: thesaurus.apply(w)) # doctest: +NORMALIZE_WHITESPACE
        0        a
        1        a
        2        a
        3        3
        4     <NA>
        5        b
        6        b
        7        b
        8        c
        9        c
        10       c
        dtype: object

        """
        if pd.isna(x):
            return pd.NA
        x = x.strip()
        for key in self._compiled:
            for pattern in self._compiled[key]:
                if len(pattern.findall(x)):
                    return key
        return x

    def find_and_replace(self, x):
        """Applies a thesaurus to a string, reemplacing the portion of string
        matching the current pattern with the key.

        Examples
        ----------------------------------------------------------------------------------------------

        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...    'f': ['AAA', 'BBB', 'ccc AAA', 'ccc BBB', 'ddd EEE', 'ddd FFF',  None, 'zzz'],
        ... })
        >>> df # doctest: +NORMALIZE_WHITESPACE
                 f
        0      AAA
        1      BBB
        2  ccc AAA
        3  ccc BBB
        4  ddd EEE
        5  ddd FFF
        6     None
        7      zzz
        >>> patterns = {'aaa':['AAA', 'BBB', 'EEE', 'FFF'],  '1':['000']}
        >>> thesaurus = Thesaurus(patterns)
        >>> thesaurus = thesaurus.compile()
        >>> df.f.map(lambda x: thesaurus.find_and_replace(x))
        0        aaa
        1        aaa
        2    ccc aaa
        3    ccc aaa
        4    ddd aaa
        5    ddd aaa
        6       <NA>
        7        zzz
        Name: f, dtype: object

        """
        if pd.isna(x):
            return pd.NA
        x = x.strip()
        for key in self._compiled:
            for pattern in self._compiled[key]:
                w = pattern.sub(key, x)
                if x != w:
                    return w
        return x

    def merge_keys(self, key, popkey):
        """Adds the strings associated to popkey to key and delete popkey."""
        if isinstance(popkey, list):
            for k in popkey:
                self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[k]
                self._thesaurus.pop(k)
        else:
            self._thesaurus[key] = self._thesaurus[key] + self._thesaurus[popkey]
            self._thesaurus.pop(popkey)

    def merge_keys_from_textfile(self, filename):
        dict_ = load_file_as_dict(filename)
        for key in dict_.keys():
            self.merge_keys(key, dict_[key])

    def pop_key(self, key):
        """Deletes key from thesaurus."""
        self._thesaurus.pop(key)

    def change_key(self, current_key, new_key):
        self._thesaurus[new_key] = self._thesaurus[current_key]
        self._thesaurus.popkey(current_key)

    def to_dict(self):
        result = {}
        for key in self._thesaurus.keys():
            for value in self._thesaurus[key]:
                result[value] = key
        return result

    def apply_as_dict(self, x, strict=False):

        if x is pd.NA:
            return pd.NA
        if strict is True and x not in self._dict.keys():
            return pd.NA
        return self._dict[x] if x in self._dict.keys() else x


# #  def text_nesting(
# #     x, search_strategy="fingerprint", sep=None, transformer=None, max_distance=None
# # ):
# #     """

# #     Examples
# #     ----------------------------------------------------------------------------------------------

# #     >>> import pandas as pd
# #     >>> df = pd.DataFrame({
# #     ...    'f': ['a',
# #     ...          'a b',
# #     ...          'a b c',
# #     ...          'a b c d',
# #     ...          'a e',
# #     ...          'a f',
# #     ...          'a b e',
# #     ...          'a b e f',
# #     ...          'a b e f g'],
# #     ... })
# #     >>> df # doctest: +NORMALIZE_WHITESPACE
# #                f
# #     0          a
# #     1        a b
# #     2      a b c
# #     3    a b c d
# #     4        a e
# #     5        a f
# #     6      a b e
# #     7    a b e f
# #     8  a b e f g

# #     >>> text_nesting(df.f, sep=',') # doctest: +NORMALIZE_WHITESPACE
# #     {
# #       "a": [
# #         "a",
# #         "a e",
# #         "a f"
# #       ],
# #       "a b": [
# #         "a b",
# #         "a b e"
# #       ],
# #       "a b c": [
# #         "a b c",
# #         "a b c d"
# #       ],
# #       "a b e f": [
# #         "a b e f",
# #         "a b e f g"
# #       ]
# #     }
# #     ignore_case=False, full_match=True, use_re=False, compiled=False

# #     >>> df = pd.DataFrame({
# #     ...    'f': ['neural networks; Artificial Neural Networks']
# #     ... })
# #     >>> df # doctest: +NORMALIZE_WHITESPACE
# #                                                  f
# #     0  neural networks; Artificial Neural Networks
# #     >>> text_nesting(df.f, sep=';', max_distance=1) # doctest: +NORMALIZE_WHITESPACE
# #     {
# #       "neural networks": [
# #         "Artificial Neural Networks",
# #         "neural networks"
# #       ]
# #     }
# #     ignore_case=False, full_match=True, use_re=False, compiled=False

# #     """

# #     x = x.dropna()

# #     if sep is not None:
# #         x = pd.Series([z.strip() for y in x for z in y.split(sep)])

# #     result = {}
# #     selected = {text: False for text in x.tolist()}

# #     max_text_len = max([len(text) for text in x])
# #     sorted_x = []
# #     for text_len in range(max_text_len, -1, -1):
# #         texts = x[[True if len(w) == text_len else False for w in x]]
# #         texts = sorted(texts)
# #         sorted_x += texts
# #     x = sorted_x

# #     # for pattern in x:
# #     for iter in tqdm(range(len(x))):

# #         pattern = x[iter]

# #         if pattern == "":
# #             continue

# #         if selected[pattern] is True:
# #             continue

# #         nested_texts = [
# #             text
# #             for text in x
# #             if selected[text] is False and steamming_all(pattern, text)
# #         ]

# #         if max_distance is not None:
# #             nested_texts = [
# #                 z
# #                 for z in nested_texts
# #                 if abs(len(pattern.split()) - len(z.split())) <= max_distance
# #             ]

# #         if len(nested_texts) > 1:
# #             nested_texts = sorted(list(set(nested_texts)))

# #         if len(nested_texts) > 1:

# #             if transformer is not None:
# #                 pattern = transformer(pattern)
# #             if pattern in result.keys():
# #                 result[pattern] += nested_texts
# #             else:
# #                 result[pattern] = nested_texts
# #             for txt in nested_texts:
# #                 selected[txt] = True

# #     return Thesaurus(result, ignore_case=False, full_match=True, use_re=False)


if __name__ == "__main__":
    #
    import doctest

    doctest.testmod()
