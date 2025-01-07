import os
from setuptools import setup, find_packages


path = os.path.abspath(os.path.dirname(__file__))

setup(
    name="nlql",
    version="0.1.3",
    keywords=("natural language", "sql", "llm", "ai", "rag", "nlql", "nlp"),
    description="NLQL (Natural Language Query Language) is a tool that helps you search through text using simple commands that look like SQL. Just like how SQL helps you find information in databases, NLQL helps you find information in regular text.",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    python_requires=">=3.9.0",
    license="MIT Licence",


    url="https://github.com/natural-language-query-language/nlql-python",
    author="Okysu",
    author_email="yby@ecanse.com",

    packages=find_packages(),

    install_requires=[
        'numpy'
    ],
)
