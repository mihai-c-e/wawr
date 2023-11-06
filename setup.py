from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Evola Constellations'
LONG_DESCRIPTION = ''

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="evola-constellations",
    version=VERSION,
    author="Mihai Ermaliuc",
    author_email="mihai.ermaliuc@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["pandas", "openai", "langchain", "neo4j", "py2neo", "tiktoken", "scikit-learn"],

    keywords=['knowledge graphs', ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)