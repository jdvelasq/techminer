import re
from os.path import dirname, join

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import Word

from techminer.core.thesaurus import load_file_as_dict


def bigram_finder(text):
    bigrams = nltk.Text(word_tokenize(text)).collocation_list()
    if len(bigrams) > 0:
        bigrams = [a + " " + b for a, b in bigrams]
        for bigram in bigrams:
            text = text.replace(bigram, bigram.replace(" ", "_"))
    return text


def extract_words(data, text):
    #
    def translate_bg2am(x):
        z = [bg2am_[z] if z in bg2am_.keys() else z for z in x.split()]
        return " ".join(z)

    def translate_am2bg(x):
        z = [am2bg_[z] if z in am2bg_.keys() else z for z in x.split()]
        return " ".join(z)

    STOPWORDS = stopwords.words("english")

    #
    # Keyword list preparation
    #
    keywords = pd.Series(data.Author_Keywords.tolist() + data.Index_Keywords.tolist())
    keywords = keywords.dropna()
    keywords = keywords.tolist()
    keywords = [w for k in keywords for w in k.split(";")]
    keywords = sorted(list(set(keywords)))

    #
    # Select compound keywords
    #
    candidates = [word.split() for word in keywords]
    candidates = [word for word in candidates if len(word) > 1]

    #
    # Load dictionaries
    #
    module_path = dirname(__file__)
    filename = join(module_path, "../data/bg2am.data")
    bg2am_ = load_file_as_dict(filename)
    bg2am_ = {key: bg2am_[key][0] for key in bg2am_}
    am2bg_ = {value: key for key in bg2am_.keys() for value in bg2am_[key]}

    #
    # British to American spelling
    #
    candidates_bg2am = [[translate_bg2am(w) for w in word] for word in candidates]
    candidates_bg2am = [" ".join(word) for word in candidates_bg2am]
    candidates_bg2am = [word for word in candidates_bg2am if word not in keywords]
    if len(candidates_bg2am) > 0:
        keywords += candidates_bg2am

    #
    # American to British spelling
    #
    candidates_am2bg = [[translate_am2bg(w) for w in word] for word in candidates]
    candidates_am2bg = [" ".join(word) for word in candidates_am2bg]
    candidates_am2bg = [word for word in candidates_am2bg if word not in keywords]
    if len(candidates_am2bg) > 0:
        keywords += candidates_am2bg

    #
    # Text normalization -- lower case
    #
    text = text.map(lambda w: w.lower(), na_action="ignore")
    text = text.map(
        lambda w: re.sub(r"[\s+]", " ", w),
        na_action="ignore",
    )

    compound_keywords = [keyword for keyword in keywords if len(keyword.split()) > 1]
    compound_keywords_ = [
        keyword.replace(" ", "_").replace("-", "_") for keyword in compound_keywords
    ]
    for a, b in zip(compound_keywords, compound_keywords_):
        text = text.map(lambda w: w.replace(a, b))

    #
    # Collocations
    #
    text = text.map(bigram_finder, na_action="ignore")

    #
    # Remove typical phrases
    #
    module_path = dirname(__file__)
    filename = join(module_path, "../data/phrases.data")
    with open(filename, "r") as f:
        phrases = f.readlines()
    phrases = [w.replace("\n", "") for w in phrases]
    pattern = "|".join(phrases)
    text = text.map(lambda w: re.sub(pattern, "", w), na_action="ignore")

    #
    # Replace chars
    #
    for index in [8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223]:
        text = text.map(lambda w: w.replace(chr(index), ""), na_action="ignore")
    text = text.map(lambda w: w.replace(" - ", ""), na_action="ignore")

    #
    # Keywords extraction
    #
    text = text.map(lambda w: word_tokenize(w), na_action="ignore")
    text = text.map(
        lambda w: [re.sub(r"[^a-zA-z_\-\s]", "", z) for z in w if z != ""],
        na_action="ignore",
    )
    text = text.map(lambda w: [word for word in w if word not in STOPWORDS])

    #
    # Word tagging and selection
    #
    #   import nltk
    #   nltk.download('tagsets')
    #   nltk.help.upenn_tagset()
    #
    # nouns:
    #   NN:  noun, common, singular or mass
    #   NNS: noun, common, plural
    # verbs:
    #   VB:  verb, base form
    #   VBG: verb, present participle or gerund
    # adjectives:
    #   JJ:  adjective or numeral, ordinal
    #   JJR: adjective, comparative
    #   JJS: adjective, superlative
    # adverbs
    #   RB:  adverb
    #   RBR: adverb, comparative
    #   RBS: adverb, superlative
    #
    text = text.map(lambda w: [word for word in w if word != ""], na_action="ignore")
    text = text.map(lambda w: nltk.pos_tag(w), na_action="ignore")
    text = text.map(
        lambda w: [
            (Word(z[0]).singularize(), "NN")
            if z[1] == "NNS" and z[0] not in keywords
            else z
            for z in w
        ]
    )
    wordnet_lemmatizer = WordNetLemmatizer()
    text = text.map(
        lambda w: [
            (wordnet_lemmatizer.lemmatize(z[0], "v"), "VB")
            if z[1] == "VBG" and z[0] not in keywords
            else z
            for z in w
        ]
    )

    text = text.map(
        lambda w: [
            z[0]
            for z in w
            if z[1] in ["NN", "NNS", "VB", "VBG", "JJ", "JJR", "JJS", "RBR", "RBS"]
            or "_" in z[0]
            or z[0] in keywords
        ]
    )

    #
    # Drop duplicates
    #
    # text = text.map(lambda w: list(set(w)))
    #
    # Checks:
    #   Replace '_' by ' '
    #
    text = text.map(lambda w: [a.replace("_", " ") for a in w])

    result = pd.Series([[] for i in range(len(set(text.index.tolist())))])

    for index in set(text.index.tolist()):

        t = text[index]
        if isinstance(t, list):
            result[index] += t
        else:
            for m in t:
                result[index] += m

    #
    # Verification
    #
    #  print(result)
    result = [";".join(sorted([a.strip() for a in w])) for w in result]
    result = [w for w in result if len(w) > 0]
    return result
