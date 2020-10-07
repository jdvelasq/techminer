import textwrap

TEXTLEN = 40


def shorten_ticklabels(labels):

    numbers = [text.split(" ")[-1] for text in labels]
    descriptions = [" ".join(text.split(" ")[:-1]) for text in labels]
    descriptions = [
        textwrap.shorten(text=text, width=TEXTLEN - len(numbers[0]))
        for text in descriptions
    ]
    return [
        description + " " + number for description, number in zip(descriptions, numbers)
    ]

