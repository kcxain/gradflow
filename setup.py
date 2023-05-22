import os
import re

from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as file:
    long_description = file.read()

with open("requirements.txt", encoding="utf8") as file:
    requirements = file.readlines()


def find_version(*filepath):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, *filepath)) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name="gradflow",
    version=find_version("gradflow/__init__.py"),
    author="kcxain",
    author_email="kcxain@gmail.com",
    license="Apache License 2.0",
    description=
    "A simple, scalable and efficient deep learning framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["auto-gradient", "deep-learning"],
    url="https://github.com/kcxain/gradflow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
)