from io import open
from os import path

from setuptools import setup, find_packages

# set rootdir to the repository root directory
rootdir = path.abspath(path.dirname(__file__))

# read readme.md to long_description
with open(path.join(rootdir, "README.md"), encoding="utf-8") as readme:
    long_description = readme.read()

setup(
    name="torecsys",
    version="dev",
    description=(
        "ToR[e]cSys is a PyTorch Framework to implement recommendation system algorithms, "
        "including but not limited to click-through-rate (CTR) prediction, learning-to-ranking "
        "(LTR), and Matrix/Tensor Embedding. The project objective is to develop a ecosystem "
        "to experiment, share, reproduce, and deploy in real world in a smooth and easy way."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7"
    ],
    keywords=(
        "recsys recommendersystem recommendationsystem "
        "machinelearning deeplearning research ctr "
        "clickthroughrate"
    ),
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.7",
    install_requires=[
        "click==7.0",
        "matplotlib>=3.1.1",
        "numpy>=1.17.0",
        "pandas>=0.25.0",
        "scipy>=1.3.1",
        "scikit-learn>=0.21.3",
        "sqlalchemy>=1.3.6",
        "tensorboard==1.14.0",
        "texttable>=1.6.2",
        "tqdm==4.33.0"
    ],
    entry_points={
        'console_scripts': ["torecsys=torecsys.cli:main"]
    },

    url="https://github.com/p768lwy3/torecsys",
    projects_url={
        "Bug Reports": "https://github.com/p768lwy3/torecsys/issues",
        "Documentation": "https://torecsys.readthedocs.io/en/latest/",
        "Source": "https://github.com/p768lwy3/torecsys"
    },
    author="Jasper Li",
    author_email="jasper_liwaiyin@protonmail.com"
)
