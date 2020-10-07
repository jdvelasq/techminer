FROM jdvelasq/python3:vscode-minimal

RUN python3 -m pip install \
    cdlib \
    leidenalg \
    matplotlib \
    networkx \
    nltk \
    pandas \
    python-igraph \
    pywaffle \
    sklearn \
    squarify \
    wordcloud \
    pyvis \
    jsonpickle \
    textblob


RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

EXPOSE   8888  5000
WORKDIR /datalake
ENTRYPOINT /etc/init.d/ssh start && jupyter lab
