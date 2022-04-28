# setup.py for sprucfluo

import os
import site
import sys
import textwrap

import setuptools
from pip._internal.req import parse_requirements
from setuptools import setup
from setuptools.command.develop import develop as develop_orig

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# hacky, but https://stackoverflow.com/questions/14399534/reference-requirements-txt-for-the-install-requires-kwarg-in-setuptools-setup-py
install_reqs = parse_requirements('requirements.txt', session=None)

reqs = [str(ir.requirement) for ir in install_reqs]

setup(
    name="sprucfluo",
    version="0.1.0",
    author="David Hall",
    author_email="dlwh@stanford.edu",
    description="A package preprocessing data for language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.github.com/stanford-crfm/sprucfluo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=reqs
)