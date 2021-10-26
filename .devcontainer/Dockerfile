ARG VARIANT="3.7"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

RUN apt-get update \
    && apt-get -yq --no-install-recommends install  pandoc  make \
    && apt-get clean \
    && apt-get autoremove -yq \
    && rm -rf /var/lib/apt/lists/* 

RUN pip3 --disable-pip-version-check --no-cache-dir install \
    bandit \
    black \
    flake8 \
    isort \
    jedi \
    mypy \
    nose \
    poetry \
    prospector \
    pycodestyle \
    pylama \
    pylint \
    pytest \
    rope \
    yapf

RUN pip3 --disable-pip-version-check --no-cache-dir install \
    twine \
    setuptools \
    wheel 



RUN pip3 --disable-pip-version-check --no-cache-dir install \
    sphinx==4.0.3 \
    sphinx-intl \
    nbsphinx \
    sphinx-rtd-theme \ 
    sphinx_copybutton \    
    pygments 

RUN pip3 --disable-pip-version-check --no-cache-dir install \
    cdlib \
    graphviz \
    ipykernel \
    ipywidgets  \
    jsonpickle \
    leidenalg \
    matplotlib \
    networkx \
    nltk \
    numpy \
    pandas \
    python-igraph \
    pyvis \
    pywaffle \
    sklearn \
    squarify \
    textblob \
    wordcloud 



RUN pip3 --disable-pip-version-check --no-cache-dir install \
    jupyterlab

RUN apt update \
    && apt -yq --no-install-recommends install  graphviz \
    && apt clean \
    && apt autoremove -yq \
    && rm -rf /var/lib/apt/lists/* 





# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1









 