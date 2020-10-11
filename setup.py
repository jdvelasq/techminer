from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyCommand(build_py):
    def run(self):
        import nltk

        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("punkt")

        build_py.run(self)


setup(
    cmdclass={"build_py": BuildPyCommand},
    name="techminer",
    version="0.0.0",
    author="Juan D. Velasquez",
    author_email="jdvelasq@unal.edu.co",
    license="MIT",
    url="http://github.com/jdvelasq/techminer",
    description="Tech Mining of Bibliograpy",
    long_description="Tech Mining of Bibliograpy",
    keywords="bibliograpy",
    platforms="any",
    provides=["techminer"],
    install_requires=[
        "squarify",
        "nltk==3.5",
        "cdlib",
        "pyvis",
        "networkx",
        "ipywidgets",
        "textblob",
    ],
    packages=["techminer", "techminer.plots", "techminer.core", "techminer.gui"],
    package_dir={"techminer": "techminer"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
