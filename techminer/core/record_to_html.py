import os.path
import re
import textwrap

import pandas as pd

from techminer.core.thesaurus import load_file_as_dict

colors = ["#FF6433", "#2E86C1", "#2EA405"] * 10

#### SoftwareX


def record_to_HTML(x, only_abstract=False, keywords_to_highlight=None):
    """"""

    thesaurus = None
    if os.path.isfile("keywords_thesaurus.txt"):
        thesaurus = load_file_as_dict("keywords_thesaurus.txt")
        for keyword in keywords_to_highlight.copy():
            if keyword in thesaurus.keys():
                for word in thesaurus[keyword]:
                    keywords_to_highlight.append(word)

    HTML = ""

    column_list = ["Title_HL" if "Title_HL" in x.index else "Title"]
    column_list += [
        "Year",
        "Authors",
        "Global_Citations",
    ]
    column_list += ["Abstract_HL" if "Abstract_HL" in x.index else "Abstract"]
    column_list += [
        "Author_Keywords_CL" if "Author_Keywords_CL" in x.index else "Author_Keywords"
    ]
    column_list += [
        "Source_title",
    ]

    if only_abstract is False:
        column_list += [
            "Author_Keywords_CL",
            "Index_Keywords_CL",
            "Title_words",
            "Title_words_CL",
            "Abstract_words",
            "Abstract_words_CL",
            "Countries",
            "Institutions",
        ]

    for f in column_list:
        if f not in x.index:
            continue
        z = x[f]
        if pd.isna(z) is True:
            continue

        if f in [
            "Authors",
            "Author_Keywords",
            "Index_Keywords",
            "Author_Keywords_CL",
            "Index_Keywords_CL",
            "Countries",
            "Institutions",
            "Source_title",
            "Abstract_words",
            "Abstract_words_CL",
            "Title_words",
            "Title_words_CL",
        ]:

            #
            # Highlight keywords
            #
            if keywords_to_highlight is not None and f in [
                "Author_Keywords",
                "Index_Keywords",
                "Author_Keywords_CL",
                "Index_Keywords_CL",
            ]:
                for keyword in keywords_to_highlight:
                    if isinstance(keyword, str) and keyword.lower() in z.lower():
                        pattern = re.compile(r"\b" + keyword + r"\b", re.IGNORECASE)
                        z = pattern.sub("<b>" + keyword.upper() + "</b>", z)

            v = z.split(";")
            v = [a.strip() if isinstance(a, str) else a for a in v]
            HTML += "{:>18}: {}<br>".format(f, v[0])
            for m in v[1:]:
                HTML += " " * 20 + "{}<br>".format(m)
        else:
            if f == "Title" or f == "Abstract" or f == "Title_HL" or f == "Abstract_HL":

                #
                # Keywords to upper case
                #
                for keyword in keywords_to_highlight:
                    if isinstance(keyword, str) and keyword.lower() in z.lower():
                        pattern = re.compile(keyword, re.IGNORECASE)
                        z = pattern.sub("<b>" + keyword.upper() + "</b>", z)

                if f == "Abstract":
                    phrases = z.split(". ")
                    z = []
                    for i_phrase, phrase in enumerate(phrases):
                        for keyword in keywords_to_highlight:
                            if (
                                isinstance(keyword, str)
                                and keyword.lower() in phrase.lower()
                            ):
                                phrase = (
                                    '<b_style="color:{}">'.format(colors[i_phrase])
                                    + phrase
                                    + "</b>"
                                )
                                break
                        z.append(phrase)
                    z = ". ".join(z)

                s = textwrap.wrap(z, 80)
                HTML += "{:>18}: {}<br>".format(f, s[0])
                for t in s[1:]:
                    HTML += "{}<br>".format(textwrap.indent(t, " " * 20))
            elif f == "Global_Citations":
                HTML += "{:>18}: {}<br>".format(f, int(z))
            else:
                HTML += "{:>18}: {}<br>".format(f, z)

    HTML = HTML.replace("<b_style", "<b style")
    return "<pre>" + HTML + "</pre>"
