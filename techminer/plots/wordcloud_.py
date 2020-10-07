import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

TEXTLEN = 40


def wordcloud_(
    x,
    darkness=None,
    figsize=(6, 6),
    font_path=None,
    width=400,
    height=200,
    margin=2,
    ranks_only=None,
    prefer_horizontal=0.9,
    mask=None,
    scale=1,
    max_words=200,
    min_font_size=4,
    stopwords=None,
    random_state=None,
    background_color="white",
    max_font_size=None,
    font_step=1,
    mode="RGB",
    relative_scaling="auto",
    regexp=None,
    collocations=True,
    cmap="Blues",
    normalize_plurals=True,
    contour_width=0,
    contour_color="black",
    repeat=False,
    #  include_numbers=False,
    #  min_word_length=0,
):
    """Plots a wordcloud from a dataframe.

    Examples
    ----------------------------------------------------------------------------------------------

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "Num_Documents": [10, 5, 2, 1],
    ...         "Global_Citations": [4, 3, 2, 0],
    ...     },
    ...     index = "author 3,author 1,author 0,author 2".split(","),
    ... )
    >>> df
              Num_Documents  Global_Citations
    author 3             10            4
    author 1              5            3
    author 0              2            2
    author 2              1            0
    >>> fig = wordcloud(x=df['Num_Documents'], darkness=df['Global_Citations'])
    >>> fig.savefig('/workspaces/techminer/sphinx/images/wordcloud.png')

    .. image:: images/wordcloud.png
        :width: 400px
        :align: center

    """

    def color_func(word, font_size, position, orientation, font_path, random_state):
        return color_dic[word]

    darkness = x if darkness is None else darkness

    fig = plt.Figure(figsize=figsize)
    ax = fig.subplots()
    cmap = plt.cm.get_cmap(cmap)

    words = {key: value for key, value in zip(x.index, x)}

    color_dic = {
        key: cmap(
            0.5 + 0.5 * (value - darkness.min()) / (darkness.max() - darkness.min())
        )
        for key, value in zip(darkness.index, darkness)
    }

    for key in color_dic.keys():
        a, b, c, d = color_dic[key]
        color_dic[key] = (
            np.uint8(a * 255),
            np.uint8(b * 255),
            np.uint8(c * 255),
            np.uint8(d * 255),
        )

    wordcloud = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        margin=margin,
        ranks_only=ranks_only,
        prefer_horizontal=prefer_horizontal,
        mask=mask,
        scale=scale,
        color_func=color_func,
        max_words=max_words,
        min_font_size=min_font_size,
        stopwords=stopwords,
        random_state=random_state,
        background_color=background_color,
        max_font_size=max_font_size,
        font_step=font_step,
        mode=mode,
        relative_scaling=relative_scaling,
        regexp=regexp,
        collocations=collocations,
        colormap=cmap,
        normalize_plurals=normalize_plurals,
        contour_width=contour_width,
        contour_color=contour_color,
        repeat=repeat,
        #  include_numbers=include_numbers,
        #  min_word_length=min_word_length,
    )
    wordcloud.generate_from_frequencies(words)
    ax.imshow(wordcloud, interpolation="bilinear")
    #
    ax.spines["bottom"].set_color("lightgray")
    ax.spines["top"].set_color("lightgray")
    ax.spines["right"].set_color("lightgray")
    ax.spines["left"].set_color("lightgray")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.set_tight_layout(True)

    return fig

