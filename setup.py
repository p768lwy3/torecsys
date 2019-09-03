from io import open
from os import path
from setuptools import setup, find_packages

# set rootdir to the repository root directory
rootdir = path.abspath(path.dirname(__file__))

# read readme.md to long_description
with open(path.join(rootdir, "README.md"), encoding="utf-8") as readme:
    long_description = readme.read()


setup(
    # Required: project name
    name    = "torecsys",
    # Required: tag
    version = "0.0.1",
    # Optional: short description
    description="Pure PyTorch Recommender System Module",
    # Optional: long description
    long_description=long_description,
    # Optional: long description type
    long_description_content_type = "text/markdown",
    # Optional: project url
    url = "https://github.com/p768lwy3/torecsys",
    # Optional: author
    author = "Jasper Li",
    # Optional: author email
    author_email = "jasper.li.wy@gmail.com",
    # Classifier
    classifiers = [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7"
    ],
    # Optional: keywords
    keywords = "recommendationsystem machinelearning research",
    # Required: packages
    packages = find_packages(exclude=["contrib", "docs", "tests"]),
    # Optional: install_required
    install_required = ["torch"],
    # Optional: extras_required 
    # extras_required = {},
    # Optional: extra project url
    projects_url = {
        "Bug Reports": "https://github.com/p768lwy3/torecsys/issues",
        "Documentation": "https://github.com/p768lwy3/torecsys",
        "Source": "https://github.com/p768lwy3/torecsys"
    }
)
