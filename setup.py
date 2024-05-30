#!/usr/bin/env python3
from setuptools import find_packages, setup
setup(
    name="GMLLMfinetune",
    version="1.0",
    author="NLPR, UCAS",
    url="https://github.com/HSDai/GraphMLLM",
    packages=find_packages(),
    python_requires=">=3.7",
    extras_require={"dev": ["flake8", "isort", "black"]},
)
