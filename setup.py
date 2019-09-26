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
    version = "0.0.5.dev1",
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
    # Optional: install_requires
    install_requires = [
        "matplotlib>=3.1.1", 
        "numpy>=1.17.0",
        "pandas>=0.25.0",
        "scipy>=1.3.1",
        "scikit-learn>=0.21.3",
        "sqlalchemy>=1.3.6",
        "tensorboard==1.14.0",
        "texttable>=1.6.2",
        "torch==1.2.0",
        "torchaudio==0.3.0",
        "torchtext==0.4.0",
        "torchvision==0.4.0",
        "tqdm>=4.33.0"
    ],
    # Optional: python_requires,
    python_requires  = ">=3.7",
    # Optional: extras_required 
    # extras_required = {},
    # Optional: extra project url
    projects_url = {
        "Bug Reports": "https://github.com/p768lwy3/torecsys/issues",
        "Documentation": "https://torecsys.readthedocs.io/en/latest/",
        "Source": "https://github.com/p768lwy3/torecsys"
    }
)
